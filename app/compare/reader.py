"""Tam hizda pencere okuma -- `plan.WindowReader`'in ffmpeg uygulamasi.

Sozlesme: `reader(start, count)` donen dizi, dosyanin BASTAN tam cozumunun
`[start : start + count]` dilimiyle ornek ornek AYNI olmalidir. Hizalama
plani gecikmeyi bu pencerelerden olcer ve ana gecis dosyayi bastan cozer; iki
zaman cizgisi arasindaki her fark dogrudan gecikme hatasi olur.

`-ss` ile arama bu sozlesmeyi HER formatta saglamiyor. OLCULEN (0.5 s on okuma
payiyla, tam cozumle karsilastirma):

    WAV/PCM, FLAC, Ogg Opus, MP3 (Xing'li ve Xing'siz VBR, 25 dk)   bit-exact
    M4A/AAC        sapma KONUMA GORE degisiyor: 70..820 ornek
    Ogg Vorbis     sabit -128 ornek
    WebM Opus      sabit -48 ornek   (Matroska zaman damgasi ms hassasiyetinde)
    MKA FLAC       +-1 ornek titresim

Bu yuzden hizli yol bir BEYAZ listedir: yalnizca olculmus (konteyner, codec)
ciftleri. Geri kalan her sey -- bilinmeyenler dahil -- dosyayi bastan cozer ve
pencereye kadar atar. Yavas ama tanim geregi dogru; bellek yine sabit, cunku
atlanan bloklar tutulmaz.

Ikinci tuzak yeniden orneklemede: arama noktasi cikis hizinin ornek
izgarasina denk gelmezse soxr farkli bir fazdan baslar ve pencere tam cozumle
eslesmez (olculen: 44.1->48 kHz'de 13.3712 s'den okumada r = 0.9796). Arama
noktasi bu yuzden iki hizin ortak izgarasina (`1/gcd(giris, cikis)` s)
yuvarlanir ve fark kirpilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, PcmStream, ResampleCfg, open_pcm

# (ffprobe format_name'in bir parcasi, codec) -> -ss ile ornek-dogru arama.
# Yalnizca OLCULMUS ciftler. Liste genisletilecekse once `tests/test_reader.py`
# icindeki sozlesme testine yeni format eklenmeli.
_SEEK_EXACT_CONTAINERS: dict[str, frozenset[str]] = {
    "wav": frozenset({"pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_u8"}),
    "flac": frozenset({"flac"}),
    "ogg": frozenset({"opus"}),
    "mp3": frozenset({"mp3"}),
}

# Aramada hedefin bu kadar oncesinden cozmeye baslanir. Kodlayicilarin
# arama sonrasi isinma bolgesi (MP3 bit rezervuari, Opus 80 ms pre-roll,
# soxr filtre gecikmesi) bunun icinde kalir. 0.5 s ile tum beyaz liste
# bit-exact olculdu.
PREROLL_S = 0.5

_BLOCK_FRAMES = 1 << 16


def seek_exact(container: str, codec: str) -> bool:
    """Bu (konteyner, codec) cifti icin `-ss` ornek-dogru mu (olculmus mu)?"""
    names = {part.strip() for part in container.split(",")}
    return any(codec in _SEEK_EXACT_CONTAINERS.get(name, frozenset()) for name in names)


@dataclass(frozen=True)
class Source:
    """Pencere okunacak ses izinin tanimi."""

    path: Path
    stream_index: int
    channels: int
    # Izin kendi ornekleme hizi.
    source_rate: int
    # Okuma bu hizda yapilir; None ise kaynak hizinda (yeniden ornekleme yok).
    target_rate: int | None = None
    # `seek_exact` sonucu. Varsayilan GUVENLI yol: bastan coz.
    exact_seek: bool = False

    @property
    def rate(self) -> int:
        return self.target_rate if self.target_rate is not None else self.source_rate


class FFmpegWindowReader:
    """`plan.WindowReader` sozlesmesini ffmpeg ile saglar."""

    def __init__(
        self,
        ffmpeg: Path,
        source: Source,
        *,
        resample: ResampleCfg = DEFAULT_RESAMPLE,
        cancel: CancelToken | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg
        self._source = source
        self._resample = resample
        self._cancel = cancel
        rate = source.rate
        # Arama noktasinin oturmasi gereken izgara, cikis cercevesi cinsinden.
        self._grid = rate // math.gcd(source.source_rate, rate)

    @property
    def source(self) -> Source:
        return self._source

    def __call__(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0:
            raise ValueError("start ve count negatif olamaz")
        channels = self._source.channels
        if count == 0:
            return np.zeros((0, channels), dtype=np.float32)
        if self._source.exact_seek:
            return self._read_seeking(start, count)
        return self._read_from_start(start, count)

    def _open(self, begin_frame: int, frames: int | None) -> PcmStream:
        rate = self._source.rate
        return open_pcm(
            self._ffmpeg,
            self._source.path,
            sample_rate=rate,
            channels=self._source.channels,
            stream_index=self._source.stream_index,
            rate=self._source.target_rate,
            start=begin_frame / rate if begin_frame > 0 else None,
            duration=frames / rate if frames is not None else None,
            resample=self._resample,
            cancel=self._cancel,
        )

    def _read_seeking(self, start: int, count: int) -> np.ndarray:
        preroll = int(PREROLL_S * self._source.rate)
        begin = max(0, start - preroll)
        begin -= begin % self._grid
        # Sonda bir izgara payi: -t'nin yuvarlamasi son cerceveyi kesmesin.
        return self._collect(begin, start, count, frames=start - begin + count + self._grid)

    def _read_from_start(self, start: int, count: int) -> np.ndarray:
        return self._collect(0, start, count, frames=start + count + self._grid)

    def _collect(self, begin: int, start: int, count: int, *, frames: int) -> np.ndarray:
        """`begin`den akitip `[start, start+count)` araligini toplar."""
        out = np.empty((count, self._source.channels), dtype=np.float32)
        filled = 0
        position = begin
        with self._open(begin, frames) as stream:
            for block in stream.blocks(_BLOCK_FRAMES):
                end = position + block.shape[0]
                lo = max(start, position)
                hi = min(start + count, end)
                if hi > lo:
                    # Blok yeniden kullanilan bir tampona bakiyor; kopyalanmali.
                    out[lo - start : hi - start] = block[lo - position : hi - position]
                    filled = hi - start
                position = end
                if position >= start + count:
                    break
        return out[:filled]
