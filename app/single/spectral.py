"""Tek dosyadan spektral kanit -- referans olmadan "kayipli kaynakla tutarli mi".

Dosya KENDI hizinda cozulur, yeniden ornekleme YOK: soxr'in kendi kesimi
(plan olcumu: 21.5 kHz'de -30 dB) tam da aranan izi taklit eder ve temiz bir
FLAC'i "transcode" diye isaretletirdi.

Olculen ozellikler ve neden bunlar (gercek CD kesiti + ffmpeg transcode'lari,
ilk bakis; dagilimlar `tools/calibrate_transcode.py` ile cikarilir):

- **Diz** (`knee_hz`, `knee_drop_db`): uzun donem spektrumun 500 Hz icinde en
  cok dustugu yer ve dusus miktari. DIKKAT: gercek bir CD master'i da dik bir
  diz gosterebilir (olculen: 19.6 kHz'de 500 Hz'de ~5 dB, ki dB/oktava
  cevrilince -110 gorunuyordu). Planin "dogal roll-off 6-18 dB/okt" varsayimi
  bu master icin yanlisti. Dik diz TEK BASINA kanit degildir.
- **Diz sonrasi taban** (`floor_rel_db`): dizin ustunde kalan seviye, 1-4 kHz
  referansina gore. En guclu tekil ayirici: gercek CD'de icerik devam eder
  (-27..-41 dB), codec'te hicbir sey kalmaz (MP3/Opus/Vorbis -63..-115 dB).
  AAC'de -65: ffmpeg'in `aac_pns` sentezlenmis gurultu birakiyor.
- **Cerceve basina kesim IQR** (`cutoff_iqr_hz`): codec'in kesimi her cercevede
  ayni yerdedir; dogal materyalde cerceveden cerceveye kayar.
- **Side kanali** (`side_hf_rel_db`): joint-stereo'nun yuksek frekansta side'i
  cokertip cokertmedigi. Ilk bakista ffmpeg kodlayicilari cokertmiyor;
  kalibrasyon karar verecek.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import open_pcm
from app.dsp.stft import StreamingStft, hann
from app.single import thresholds

FFT_SIZE = 4096
# Referans bandi: seviye normalizasyonu icin. Muzigin enerjisi buradadir.
_REF_LO_HZ, _REF_HI_HZ = 1000.0, 4000.0
# Diz aramasi bu frekansin ustunde; alti kesim degil icerik.
_KNEE_SEARCH_FROM_HZ = 8000.0
# Kesim medyani dusukse arama kesimin bu kadar altindan baslar, en az bu Hz'ten.
_KNEE_BELOW_CUTOFF_HZ = 1500.0
_KNEE_SEARCH_MIN_HZ = 2000.0
# Dusus penceresi (Hz): dizin egimi bu genislikte olculur.
_DROP_WINDOW_HZ = 500.0
# Taban bolgesi dizin bu kadar ustunden Nyquist'in bu kesrine kadar.
_FLOOR_GAP_HZ = 400.0
_FLOOR_TOP_FRACTION = 0.98
# Cerceve basina kesim: cercevenin referansindan bu kadar asagisi "yok".
_FRAME_CUTOFF_DROP_DB = 55.0
# Sessiz cerceveler (referans bandi bu dBFS/bin altinda) sayilmaz.
_SILENT_FRAME_DB = -80.0
_SMOOTH_BINS = 5
# Yeniden orneklenmis analizde bunun ustu (Nyquist kesri) kullanilmaz: soxr
# `cutoff=0.99` gecis bandi 0.99'da basliyor ve kendi dik duvari diz aramasinda
# gercek dizi golgeliyordu (96 kHz'e buyutulmus MP3 V0'da 21.3 kHz'teki yumusak
# diz yerine 21.8 kHz'teki resampler duvari secildi).
_RESAMPLED_TOP_FRACTION = 0.985
# Anti-alias duvarinin altinda bulunan diz, kesim medyanina bu kadar yakinsa
# kullanilir (bkz. `_Accumulator.finish`).
_ANTIALIAS_KNEE_PROXIMITY_HZ = 2000.0


@dataclass(frozen=True)
class SpectralEvidence:
    sample_rate: int
    frames: int
    # 1-4 kHz referans seviyesi, dBFS/bin (tam olcekli sinusun bant enerjisi 0 dB).
    reference_db: float
    knee_hz: float
    # Dizdeki dusus, 500 Hz'lik pencerede dB. dB/oktav DEGIL: 20 kHz'de 500 Hz
    # 0.035 oktavdir ve bolme 5 dB'lik dogal bir dususu 140 dB/okt gosteriyordu.
    knee_drop_db: float
    # Diz ustunde kalan seviye, referansa gore (dB). Diz Nyquist'e cok yakinsa NaN.
    floor_rel_db: float
    cutoff_median_hz: float
    cutoff_iqr_hz: float
    # Side'in mid'e gore seviyesi: [8 kHz, diz) eksi [1, 4 kHz]. Mono'da NaN.
    side_hf_rel_db: float
    # Uzun donem spektrum (dB, referansa gore) ve frekans ekseni; rapor icin.
    ltas_rel_db: np.ndarray
    freqs_hz: np.ndarray
    # Nyquist'e yakin dik duvar (kaydin anti-alias filtresi) bulunduysa konumu
    # ve dususu; diz ve taban bu durumda duvarin ALTINDA aranir. Yoksa NaN.
    antialias_hz: float = math.nan
    antialias_drop_db: float = math.nan
    # Sifira cevrilen sonlu olmayan (NaN/inf) ornek sayisi.
    nonfinite_samples: int = 0

    @property
    def nyquist_hz(self) -> float:
        return self.sample_rate / 2.0


def _smooth(rows: np.ndarray, width: int) -> np.ndarray:
    """Son eksen boyunca kutu yumusatma; kenarlar KIRPILIR, sifirla doldurulmaz.

    "same" modu kenari sifir dB ile doldurup Nyquist yakinini yapay olarak
    yukseltiyordu (olculdu: her cercevede kesim Nyquist cikti).
    """
    kernel = np.ones(width) / width
    out = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), -1, rows)
    result: np.ndarray = out
    return result


class _Accumulator:
    """Spektral kaniti AKISLI biriktirir: tum kesitin STFT'si hic tutulmaz.

    Kanit yalnizca uzun donem ortalamalara (mid ve side gucu) ve kare basina
    tek bir kesim degerine ihtiyac duyar. Once tum spektrumlar bellekte
    tutuluyordu: 30 s'lik kesit 96 kHz'te 353 MB, 192 kHz'te 705 MB (denetim
    D5). Sonuclar tek parca hesaplamayla ayni.
    """

    def __init__(self, sample_rate: int, *, top_hz: float | None, stereo: bool) -> None:
        self.sample_rate = sample_rate
        self.top_hz = top_hz
        self._stereo = stereo
        self.restart()
        self._scale = 4.0 / (FFT_SIZE * float(np.sum(hann(FFT_SIZE) ** 2)))
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)
        self._ref_band = (self.freqs >= _REF_LO_HZ) & (self.freqs < _REF_HI_HZ)
        half = _SMOOTH_BINS // 2
        self._smooth_freqs = self.freqs[half : self.freqs.size - half]
        self._drop_bins = max(2, round(_DROP_WINDOW_HZ / (sample_rate / FFT_SIZE)))
        limit = self._smooth_freqs.size - self._drop_bins  # en ust pencere disarida
        if top_hz is not None:
            limit = min(limit, int(np.searchsorted(self._smooth_freqs, top_hz, side="right")))
        self._cutoff_limit = limit
        self._power_sum = np.zeros(self.freqs.size)
        self._side_sum = np.zeros(self.freqs.size) if stereo else None
        self._active = 0
        self._cutoffs: list[np.ndarray] = []

    def restart(self) -> None:
        """Yeni bir kesit basliyor: STFT durumu sifirlanir, birikenler kalir.

        Aksi halde iki kesitin sinirinda ikisinden birden olusan yapay bir
        cerceve dogardi.
        """
        self._mid_stft = StreamingStft(FFT_SIZE)
        self._side_stft = StreamingStft(FFT_SIZE) if self._stereo else None

    def push(self, mid: np.ndarray, side: np.ndarray | None) -> None:
        spectra = self._mid_stft.push(mid.astype(np.float64))
        side_spectra = (
            self._side_stft.push(side.astype(np.float64))
            if self._side_stft is not None and side is not None
            else None
        )
        if not spectra.shape[0]:
            return
        power = (spectra.real**2 + spectra.imag**2) * self._scale
        frame_ref_db = 10.0 * np.log10(np.maximum(power[:, self._ref_band].mean(axis=1), 1e-30))
        active = frame_ref_db > _SILENT_FRAME_DB
        if not active.any():
            return
        power, frame_ref_db = power[active], frame_ref_db[active]
        self._active += int(power.shape[0])
        self._power_sum += power.sum(axis=0)
        if self._side_sum is not None and side_spectra is not None:
            self._side_sum += (np.abs(side_spectra[active]) ** 2 * self._scale).sum(axis=0)
        frame_db = _smooth(10.0 * np.log10(np.maximum(power, 1e-30)), _SMOOTH_BINS)
        cutoffs = np.empty(frame_db.shape[0])
        for i, row in enumerate(frame_db):
            above = np.nonzero(row[: self._cutoff_limit] > frame_ref_db[i] - _FRAME_CUTOFF_DROP_DB)[
                0
            ]
            cutoffs[i] = self._smooth_freqs[above[-1]] if above.size else 0.0
        self._cutoffs.append(cutoffs)

    def finish(self) -> SpectralEvidence:
        sample_rate, freqs = self.sample_rate, self.freqs
        empty = SpectralEvidence(
            sample_rate,
            0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            np.zeros(0),
            freqs,
        )
        if self._active < 4:
            return empty

        # -- uzun donem spektrum ve diz --------------------------------------------
        ltas_db = 10.0 * np.log10(np.maximum(self._power_sum / self._active, 1e-30))
        reference_db = float(ltas_db[self._ref_band].mean())
        rel = ltas_db - reference_db
        smooth = _smooth(ltas_db[None, :], _SMOOTH_BINS)[0]
        smooth_freqs = self._smooth_freqs
        drop_bins = self._drop_bins
        # -- cerceve basina kesim -------------------------------------------------
        cutoffs = np.concatenate(self._cutoffs)
        q25, q50, q75 = np.percentile(cutoffs, [25, 50, 75])
        # Diz aramasi 8 kHz'ten basliyordu; 32/48 kbps MP3'un 4-8 kHz'teki
        # duvari hic gorulmuyor ve 10 dosyanin 5'i "belirsiz" cikiyordu (denetim
        # D12). Kesim daha asagidaysa arama kesimin biraz altindan baslar.
        search_from = float(
            np.clip(q50 - _KNEE_BELOW_CUTOFF_HZ, _KNEE_SEARCH_MIN_HZ, _KNEE_SEARCH_FROM_HZ)
        )
        start = int(np.searchsorted(smooth_freqs, search_from))
        stop = smooth_freqs.size - drop_bins
        if self.top_hz is not None:
            stop = min(
                stop, int(np.searchsorted(smooth_freqs, self.top_hz, side="right")) - drop_bins
            )
        if stop <= start:
            return empty

        def knee(stop: int) -> tuple[int, float, float]:
            drops = smooth[start + drop_bins : stop + drop_bins] - smooth[start:stop]
            at = int(np.argmin(drops)) + start
            return at, float(smooth_freqs[at] + _DROP_WINDOW_HZ / 2.0), float(-drops[at - start])

        at, knee_hz, knee_drop = knee(stop)
        floor_hi = _FLOOR_TOP_FRACTION * sample_rate / 2.0
        if self.top_hz is not None:
            floor_hi = min(floor_hi, self.top_hz)
        antialias_hz = antialias_drop = math.nan
        nyquist = sample_rate / 2.0
        if (
            knee_drop > thresholds.BRICKWALL_DROP_DB
            and knee_hz >= thresholds.MAX_LOSSY_CUTOFF_NYQUIST_FRACTION * nyquist
            and at - drop_bins + 1 > start
        ):
            # Nyquist'e yakin dik duvar kaydin anti-alias filtresidir; ustu
            # filtrenin sondurme bandi. Diz orada aranirsa karanlik ama kayipsiz
            # bir kaydin yumusak dogal inisi gorunmuyor, sondurme bandi da "diz
            # ustunde hicbir sey yok" diye kanit sayiliyordu (denetim D11).
            below = knee(at - drop_bins + 1)
            # Duvarin altindaki diz ancak icerigin bittigi yere (kesim medyani)
            # yakinsa o bitisi anlatir. Uzaksa (MP3 V0: kesim 16.0, diz 21.7 kHz;
            # AAC: kesim 20.0, diz 11.2 kHz) bir sey soylemez ve iki kayipli
            # dosyayi "belirsiz"e cekiyordu; o durumda eski davranis kalir.
            # DOGRULANMADI: 2 kHz'lik yakinlik sentetik karanlik kayittan
            # (1.55 kHz); gercek karanlik + anti-alias kayit sette yok.
            if abs(below[1] - float(q50)) <= _ANTIALIAS_KNEE_PROXIMITY_HZ:
                antialias_hz, antialias_drop = knee_hz, knee_drop
                floor_hi = min(floor_hi, float(smooth_freqs[at]))
                at, knee_hz, knee_drop = below

        # -- diz sonrasi taban ---------------------------------------------------
        floor_lo = knee_hz + _FLOOR_GAP_HZ
        floor_band = (freqs >= floor_lo) & (freqs <= floor_hi)
        floor_rel = float(rel[floor_band].mean()) if floor_band.sum() >= 4 else math.nan

        # -- side ------------------------------------------------------------------
        side_hf = math.nan
        if self._side_sum is not None:
            side_db = 10.0 * np.log10(np.maximum(self._side_sum / self._active, 1e-30))
            hf = (freqs >= _KNEE_SEARCH_FROM_HZ) & (freqs < knee_hz)
            if hf.sum() >= 4:
                ref_band = self._ref_band
                side_hf = float(
                    (side_db[hf] - ltas_db[hf]).mean()
                    - (side_db[ref_band] - ltas_db[ref_band]).mean()
                )

        return SpectralEvidence(
            sample_rate=sample_rate,
            frames=self._active,
            reference_db=reference_db,
            knee_hz=knee_hz,
            knee_drop_db=knee_drop,
            floor_rel_db=floor_rel,
            cutoff_median_hz=float(q50),
            cutoff_iqr_hz=float(q75 - q25),
            side_hf_rel_db=side_hf,
            ltas_rel_db=rel.astype(np.float32),
            freqs_hz=freqs,
            antialias_hz=antialias_hz,
            antialias_drop_db=antialias_drop,
        )


def analyse_samples(
    mid: np.ndarray,
    side: np.ndarray | None,
    sample_rate: int,
    *,
    top_hz: float | None = None,
) -> SpectralEvidence:
    """Ham mid/side orneklerinden kanit cikarir.

    `top_hz`: bunun ustundeki spektrum dosyaya degil araya giren resampler'a
    aittir; diz, taban ve cerceve kesimi bu sinirin altinda aranir.
    """
    accumulator = _Accumulator(sample_rate, top_hz=top_hz, stereo=side is not None)
    accumulator.push(mid, side)
    return accumulator.finish()


def analyse(
    ffmpeg: Path,
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    stream_index: int = 0,
    start: float | None = None,
    duration: float | None = None,
    rate: int | None = None,
    extra: Sequence[tuple[float, float]] = (),
    cancel: CancelToken | None = None,
) -> SpectralEvidence:
    """Dosyanin bir kesitini cozup kanit cikarir; bellek kesit boyundan bagimsiz.

    `extra`: ayni kanita eklenecek ek (baslangic, sure) kesitleri. Toplu tarama
    belirsiz bir dosyada iki kesit daha ekleyip kanit tabanini genisletir.

    Varsayilan native hiz: resampler'in kendi kesimi olcumu kirletir. `rate`
    yalnizca icerigin zaten o hizin Nyquist'inin altinda bittigi bilinen
    durumda verilir (bkz. `verdict.verify`).
    """
    if rate is not None:
        sample_rate = rate
    top_hz = _RESAMPLED_TOP_FRACTION * sample_rate / 2.0 if rate is not None else None
    accumulator = _Accumulator(sample_rate, top_hz=top_hz, stereo=channels >= 2)
    nonfinite = 0
    segments: list[tuple[float | None, float | None]] = [(start, duration), *extra]
    for segment_start, segment_duration in segments:
        accumulator.restart()
        with open_pcm(
            ffmpeg,
            path,
            sample_rate=sample_rate,
            channels=channels,
            stream_index=stream_index,
            rate=rate,
            start=segment_start,
            duration=segment_duration,
            cancel=cancel,
        ) as stream:
            for block in stream.blocks(1 << 16):
                pcm = block.astype(np.float64)
                if pcm.shape[1] >= 2:
                    accumulator.push((pcm[:, 0] + pcm[:, 1]) * 0.5, (pcm[:, 0] - pcm[:, 1]) * 0.5)
                else:
                    accumulator.push(pcm[:, 0], None)
            nonfinite += stream.nonfinite
    evidence = accumulator.finish()
    return replace(evidence, nonfinite_samples=nonfinite) if nonfinite else evidence
