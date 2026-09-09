"""ffmpeg'den ham PCM akisi cekmenin TEK yolu.

Bu modulun varlik sebebi tek bir olcumdur. 600 s / 44.1k stereo f32le (211 MB)
bir akisi Windows'ta ffmpeg borusundan okumak:

    stdout.read(256 KB) ....  2.06 s
    stdout.read(2 MB)   ....  7.12 s
    stdout.read(16 MB)  .... 117.60 s     <-- 57 kat yavas
    readinto(memoryview) ...  2.1 s       <-- her tampon boyutunda ayni

Sezgisel "buyuk parca daha hizlidir" varsayimi burada 57 kat yanlis. Kok neden
dogrulanmadi (muhtemelen kismi okuma sonrasi bytes yeniden boyutlandirmasi),
ama davranis tekrarlanabilir. Bu yuzden:

    KURAL: bu depoda ffmpeg borusundan `read(n)` ile okunmaz. Her zaman
    onceden ayrilmis bir bytearray'e `readinto()` yapilir.

Kurali tests/test_ffmpeg_stream.py'deki verim testi kilitler.

Ikinci kural yeniden orneklemeyle ilgili. ffmpeg'in varsayilan soxr'i, hangi
yone gidilirse gidilsin dusuk Nyquist'in ~%95'inde sessizce bir duvar orer:
44.1 -> 48 kHz YUKARI orneklemede bile (bilgi kaybi matematiksel olarak
gereksizken) 21.5 kHz'de -30 dB bastirma olculdu. Varsayilana birakilirsa
aracin kendi transcode dedektoru temiz bir FLAC'i "20.9 kHz'de kesilmis" diye
isaretler. Bu yuzden `cutoff` her zaman acikca verilir.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

import numpy as np

from app.core.errors import FFmpegFailedError
from app.core.ffmpeg_runner import CancelToken, StderrCollector, kill_tree, spawn

# Okuma tamponu. Olculen egride 256 KB - 4 MB araligi duz; 1 MB hem sistem
# cagrisi sayisini dusuk tutar hem 10 s'lik bloklarla dogal ortusur.
DEFAULT_BLOCK_BYTES = 1 << 20

# ffmpeg ornek formati -> numpy dtype. Yalnizca analiz icin anlamli olanlar.
_DTYPES: dict[str, np.dtype[np.generic]] = {
    "f32le": np.dtype(np.float32),
    "s16le": np.dtype(np.int16),
    "s32le": np.dtype(np.int32),
}


@dataclass(frozen=True)
class ResampleCfg:
    """Yeniden ornekleme ayarlari.

    `cutoff` varsayilana BIRAKILMAZ: modul docstring'indeki olcume gore
    ffmpeg'in varsayilani tepe frekanslari sessizce kirpiyor. 0.99 ile 44.1/48
    ciftinde 21.8 kHz'e kadar duz yanit olculdu; 0.995 ve uzeri ek kazanc
    vermedi (soxr iceride bir sinir uyguluyor gorunuyor).
    """

    engine: str = "soxr"
    precision: int = 28
    cutoff: float = 0.99

    def filter_expr(self, rate: int) -> str:
        return (
            f"aresample={rate}:resampler={self.engine}"
            f":precision={self.precision}:cutoff={self.cutoff}"
        )


DEFAULT_RESAMPLE = ResampleCfg()


class _SupportsReadInto(Protocol):
    """Onceden ayrilmis bir tampona okuyabilen akis.

    Bu modulun ffmpeg borusundan bekledigi tek sozlesme budur; `read(n)`
    bilincli olarak yok (modul docstring'indeki olcume bak).
    """

    def readinto(self, buffer: memoryview) -> int | None: ...


def pcm_args(
    path: Path,
    *,
    stream_index: int = 0,
    rate: int | None = None,
    channels: int | None = None,
    fmt: str = "f32le",
    start: float | None = None,
    duration: float | None = None,
    af_chain: Sequence[str] = (),
    resample: ResampleCfg = DEFAULT_RESAMPLE,
) -> list[str]:
    """Ham PCM ureten ffmpeg arguman listesini kurar.

    `-vn -sn -dn` ve `-map 0:a:<i>` her cagrida zorunlu: 20 GB'lik bir MKV
    verildiginde video paketine dokunulmamasini garanti eden sey budur.

    `rate=None` ise HIC yeniden ornekleme yapilmaz. Dosya-ici olcumler
    (kesim frekansi, transcode kaniti, bit entropisi) her zaman bu yolu
    kullanmalidir; resampler'in kendi kesimi olcumu kirletir.

    `-ss` girdiden ONCE verilir (input seek): FLAC/M4A'da ucuzdur, cikti
    tarafinda arama tum dosyayi cozmek demektir.
    """
    if fmt not in _DTYPES:
        raise ValueError(f"desteklenmeyen ornek formati: {fmt}")

    args: list[str] = ["-nostdin", "-hide_banner", "-v", "error"]
    if start is not None:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    args += ["-vn", "-sn", "-dn", "-map", f"0:a:{stream_index}"]

    chain = list(af_chain)
    if rate is not None:
        chain.append(resample.filter_expr(rate))
    if chain:
        args += ["-af", ",".join(chain)]
    if channels is not None:
        args += ["-ac", str(channels)]

    args += ["-f", fmt, "-"]
    return args


class PcmStream:
    """ffmpeg'den akan ham PCM. Context manager olarak kullanilir.

    `blocks()` her adimda AYNI tamponun uzerine oturan bir numpy view dondurur.
    Kopyasiz olmasi bilincli: 10 s'lik bloklar saniyede yuzlerce MB akar ve her
    blok icin yeni dizi ayirmak GC'yi bosuna calistirir. Bir blogu saklamak
    isteyen taraf KENDISI kopyalar.
    """

    def __init__(
        self,
        exe: Path,
        args: Sequence[str],
        *,
        sample_rate: int,
        channels: int,
        fmt: str = "f32le",
        cancel: CancelToken | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = _DTYPES[fmt]
        self.frame_bytes = self.dtype.itemsize * channels
        self._args = (str(exe), *args)
        self._cancel = cancel
        self._eof = False
        self._proc: subprocess.Popen[bytes] = spawn(exe, args)
        self._stderr = StderrCollector(self._proc.stderr)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> PcmStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Tuketici dongudan erken cikarsa ffmpeg dolu bir boruya yazmaya
        # calisirken sonsuza kadar bloke olur; bu yuzden EOF'a varilmadiysa
        # surec agaci oldurulur.
        self.close(kill=not self._eof)

    # -- okuma --------------------------------------------------------------

    def blocks(self, n_frames: int) -> Iterator[np.ndarray]:
        """`n_frames` cerceve buyuklugunde bloklar uretir.

        Son blok kisa olabilir. Uretilen dizi sekli (frames, channels).
        """
        if n_frames <= 0:
            raise ValueError("n_frames pozitif olmali")
        block_bytes = n_frames * self.frame_bytes
        buf = bytearray(block_bytes)
        view = memoryview(buf)
        raw = self._proc.stdout
        if raw is None:  # pragma: no cover - spawn her zaman PIPE veriyor
            raise RuntimeError("stdout borusu yok")
        # subprocess.Popen.stdout typeshed'de IO[bytes] olarak gorunur ve orada
        # readinto beyan edilmez. Gercekte spawn() bufsize=0 verdigi icin nesne
        # ham bir FileIO'dur; asagidaki Protocol o sozlesmeyi acikca yazar.
        stream = cast("_SupportsReadInto", raw)

        while True:
            if self._cancel is not None:
                self._cancel.raise_if_cancelled()
            filled = 0
            while filled < block_bytes:
                # KURAL: read(n) degil, readinto. Modul docstring'ine bak.
                got = stream.readinto(view[filled:])
                if not got:
                    break
                filled += got
            if filled == 0:
                self._eof = True
                break
            frames = filled // self.frame_bytes
            if frames == 0:
                # Tam cerceveden az veri: bozuk akis, sessizce yutulmaz.
                self._eof = True
                break
            yield np.frombuffer(buf, dtype=self.dtype, count=frames * self.channels).reshape(
                frames, self.channels
            )
            if filled < block_bytes:
                self._eof = True
                break

        self._finish()

    def _finish(self) -> None:
        """Surecin duzgun bittigini dogrular, aksi halde hata firlatir."""
        self._proc.wait(timeout=30)
        self._stderr.join()
        if self._proc.returncode not in (0, None):
            raise FFmpegFailedError(self._args, self._proc.returncode, self._stderr.tail())

    @property
    def stderr_tail(self) -> str:
        return self._stderr.tail()

    def close(self, *, kill: bool = False) -> None:
        if kill:
            kill_tree(self._proc)
        if self._proc.stdout is not None:
            self._proc.stdout.close()
        self._stderr.join(timeout=1.0)


def open_pcm(
    ffmpeg: Path,
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    stream_index: int = 0,
    rate: int | None = None,
    fmt: str = "f32le",
    start: float | None = None,
    duration: float | None = None,
    af_chain: Sequence[str] = (),
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    cancel: CancelToken | None = None,
) -> PcmStream:
    """PcmStream acar.

    `sample_rate` ve `channels`, akisin GERCEK cikti parametreleridir: cagiran
    taraf bunlari ffprobe sonucundan ve istedigi donusumden bilir. ffmpeg ham
    PCM'de basik yazmadigi icin akistan okunamazlar, bu yuzden zorunlu.
    """
    args = pcm_args(
        path,
        stream_index=stream_index,
        rate=rate,
        channels=channels,
        fmt=fmt,
        start=start,
        duration=duration,
        af_chain=af_chain,
        resample=resample,
    )
    return PcmStream(
        ffmpeg,
        args,
        sample_rate=sample_rate,
        channels=channels,
        fmt=fmt,
        cancel=cancel,
    )
