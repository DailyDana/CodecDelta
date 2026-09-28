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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import open_pcm
from app.dsp.stft import StreamingStft, hann

FFT_SIZE = 4096
# Referans bandi: seviye normalizasyonu icin. Muzigin enerjisi buradadir.
_REF_LO_HZ, _REF_HI_HZ = 1000.0, 4000.0
# Diz aramasi bu frekansin ustunde; alti kesim degil icerik.
_KNEE_SEARCH_FROM_HZ = 8000.0
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


def analyse_samples(mid: np.ndarray, side: np.ndarray | None, sample_rate: int) -> SpectralEvidence:
    """Ham (native hizda) mid/side orneklerinden kanit cikarir."""
    stft = StreamingStft(FFT_SIZE)
    spectra = stft.push(mid.astype(np.float64))
    scale = 4.0 / (FFT_SIZE * float(np.sum(hann(FFT_SIZE) ** 2)))
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)
    power = (spectra.real**2 + spectra.imag**2) * scale
    ref_band = (freqs >= _REF_LO_HZ) & (freqs < _REF_HI_HZ)
    frame_ref_db = 10.0 * np.log10(np.maximum(power[:, ref_band].mean(axis=1), 1e-30))
    active = frame_ref_db > _SILENT_FRAME_DB
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
    if active.sum() < 4:
        return empty
    power = power[active]
    frame_ref_db = frame_ref_db[active]

    # -- uzun donem spektrum ve diz --------------------------------------------
    ltas_db = 10.0 * np.log10(np.maximum(power.mean(axis=0), 1e-30))
    reference_db = float(ltas_db[ref_band].mean())
    rel = ltas_db - reference_db
    half = _SMOOTH_BINS // 2
    smooth = _smooth(ltas_db[None, :], _SMOOTH_BINS)[0]
    smooth_freqs = freqs[half : freqs.size - half]
    bin_hz = sample_rate / FFT_SIZE
    drop_bins = max(2, round(_DROP_WINDOW_HZ / bin_hz))
    start = int(np.searchsorted(smooth_freqs, _KNEE_SEARCH_FROM_HZ))
    stop = smooth_freqs.size - drop_bins
    if stop <= start:
        return empty
    drops = smooth[start + drop_bins : stop + drop_bins] - smooth[start:stop]
    at = int(np.argmin(drops)) + start
    knee_hz = float(smooth_freqs[at] + _DROP_WINDOW_HZ / 2.0)
    knee_drop = float(-drops[at - start])

    # -- diz sonrasi taban ---------------------------------------------------
    floor_lo = knee_hz + _FLOOR_GAP_HZ
    floor_hi = _FLOOR_TOP_FRACTION * sample_rate / 2.0
    floor_band = (freqs >= floor_lo) & (freqs <= floor_hi)
    floor_rel = float(rel[floor_band].mean()) if floor_band.sum() >= 4 else math.nan

    # -- cerceve basina kesim -------------------------------------------------
    frame_db = _smooth(10.0 * np.log10(np.maximum(power, 1e-30)), _SMOOTH_BINS)
    frame_freqs = smooth_freqs
    limit = frame_freqs.size - drop_bins  # en ust pencere disarida
    cutoffs = np.empty(frame_db.shape[0])
    for i, row in enumerate(frame_db):
        above = np.nonzero(row[:limit] > frame_ref_db[i] - _FRAME_CUTOFF_DROP_DB)[0]
        cutoffs[i] = frame_freqs[above[-1]] if above.size else 0.0
    q25, q50, q75 = np.percentile(cutoffs, [25, 50, 75])

    # -- side ------------------------------------------------------------------
    side_hf = math.nan
    if side is not None and side.size:
        side_power = np.abs(StreamingStft(FFT_SIZE).push(side.astype(np.float64))) ** 2 * scale
        side_power = side_power[active]
        side_db = 10.0 * np.log10(np.maximum(side_power.mean(axis=0), 1e-30))
        hf = (freqs >= _KNEE_SEARCH_FROM_HZ) & (freqs < knee_hz)
        if hf.sum() >= 4:
            side_hf = float(
                (side_db[hf] - ltas_db[hf]).mean() - (side_db[ref_band] - ltas_db[ref_band]).mean()
            )

    return SpectralEvidence(
        sample_rate=sample_rate,
        frames=int(power.shape[0]),
        reference_db=reference_db,
        knee_hz=knee_hz,
        knee_drop_db=knee_drop,
        floor_rel_db=floor_rel,
        cutoff_median_hz=float(q50),
        cutoff_iqr_hz=float(q75 - q25),
        side_hf_rel_db=side_hf,
        ltas_rel_db=rel.astype(np.float32),
        freqs_hz=freqs,
    )


def analyse(
    ffmpeg: Path,
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    stream_index: int = 0,
    start: float | None = None,
    duration: float | None = None,
    cancel: CancelToken | None = None,
) -> SpectralEvidence:
    """Dosyanin bir kesitini native hizda cozup kanit cikarir."""
    blocks = []
    with open_pcm(
        ffmpeg,
        path,
        sample_rate=sample_rate,
        channels=channels,
        stream_index=stream_index,
        start=start,
        duration=duration,
        cancel=cancel,
    ) as stream:
        for block in stream.blocks(1 << 16):
            blocks.append(block.astype(np.float64))
    if not blocks:
        return analyse_samples(np.zeros(0), None, sample_rate)
    pcm = np.concatenate(blocks)
    if pcm.shape[1] >= 2:
        mid = (pcm[:, 0] + pcm[:, 1]) * 0.5
        side: np.ndarray | None = (pcm[:, 0] - pcm[:, 1]) * 0.5
    else:
        mid, side = pcm[:, 0], None
    return analyse_samples(mid, side, sample_rate)
