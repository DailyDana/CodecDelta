"""Olcum tabani -- zincirin kendisinin ne kadar fark urettigi.

Iki dosya farkli hizdaysa biri yeniden orneklenir; kesirli bir gecikme varsa
spektrumda faz rampasiyla telafi edilir. Ikisi de fark uretir ve bu fark codec
gurultusunden ayirt edilemez. Plan olcumu: 44.1->48->44.1 gidis-donusunde
soxr, 20 kHz altinda 118..148 dB S/N verirken 20-21 kHz'de 50 dB'e iniyor.
Elle yapilan ilk analizdeki "20 kHz ustunde S/N 0.02 dB" sayisi codec hakkinda
hicbir sey soylemiyordu; neredeyse tamami resampler'in kendi kesimiydi.

Bu yuzden her karsilastirmada taban OLCULUR, varsayilmaz:

- Yeniden ornekleme: referansin bir kesiti karsi dosyanin hizina gidip geri
  doner ve kendisiyle karsilastirilir. Gidis-donus hatayi iki kez uygular,
  yani taban muhafazakardir (gercekten biraz daha kotu gosterir).
- Kesirli gecikme: ayni kesit zaman alaninda kesirli kaydirilir, spektrumda
  faz rampasiyla geri alinir; artik rampanin hatasidir.

Iki hata bagimsiz gurultu gibi toplanir.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, ResampleCfg, open_pcm
from app.dsp.accum import CrossSpectrum, hz_to_bin
from app.dsp.stft import StreamingStft, phase_shift
from app.dsp.transforms import fractional_shift, to_mono

DEFAULT_EXCERPT_S = 30.0


def _decode_mono(
    ffmpeg: Path,
    path: Path,
    *,
    stream_index: int,
    channels: int,
    sample_rate: int,
    start: float | None,
    duration: float,
    af_chain: Sequence[str] = (),
    cancel: CancelToken | None,
) -> np.ndarray:
    blocks = []
    with open_pcm(
        ffmpeg,
        path,
        sample_rate=sample_rate,
        channels=channels,
        stream_index=stream_index,
        start=start,
        duration=duration,
        af_chain=af_chain,
        cancel=cancel,
    ) as stream:
        for block in stream.blocks(1 << 16):
            blocks.append(to_mono(block.astype(np.float64)))
    return np.concatenate(blocks) if blocks else np.zeros(0)


def _band_snr(
    a: np.ndarray, b: np.ndarray, size: int, bands: Sequence[tuple[int, int]]
) -> list[float]:
    """a ile b arasindaki farkin bant basina INKOHERENT S/N'i (dB)."""
    n = min(a.size, b.size)
    return _spectral_snr(StreamingStft(size).push(a[:n]), StreamingStft(size).push(b[:n]), bands)


def _spectral_snr(
    spectra_a: np.ndarray, spectra_b: np.ndarray, bands: Sequence[tuple[int, int]]
) -> list[float]:
    """Bant basina inkoherent S/N -- ana olcumle AYNI metrik.

    Taban, karsilastirildigi sayiyla ayni sekilde olculmeli. Ana olcumun
    mansetteki S/N'i inkoherenttir: resampler'in gecis bandi dalgalanmasi gibi
    dogrusal etkiler oradan zaten ayiklanir. Tabani duz farkla olcmek onu
    oldugundan kotu gosteriyordu (18-20 kHz: duz 75.5 dB, inkoherent 118 dB)
    ve olculebilir bantlari "olculemez" diye isaretliyordu.
    """
    # Kenar cercevelerinde filtre gecis bolgeleri var; disarida birakilir.
    core = slice(2, -2) if spectra_a.shape[0] > 8 else slice(None)
    accumulator = CrossSpectrum(spectra_a.shape[1])
    accumulator.add(spectra_a[core], spectra_b[core])
    out = []
    for lo, hi in bands:
        stats = accumulator.band(lo, max(hi, lo + 1))
        out.append(stats.snr_db if stats.reference_power > 0.0 else math.nan)
    return out


def combine_floors(*floors_db: float) -> float:
    """Bagimsiz hata kaynaklarinin tabanlarini birlestirir (guc toplami)."""
    total = 0.0
    for floor in floors_db:
        if math.isnan(floor):
            return math.nan
        if math.isinf(floor):
            continue
        total += 10.0 ** (-floor / 10.0)
    return math.inf if total == 0.0 else -10.0 * math.log10(total)


def measure_floor(
    ffmpeg: Path,
    path: Path,
    *,
    stream_index: int,
    channels: int,
    source_rate: int,
    other_rate: int,
    fractional_delay: float,
    bands_hz: Sequence[tuple[float, float]],
    size: int,
    start: float | None = None,
    excerpt_s: float = DEFAULT_EXCERPT_S,
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    cancel: CancelToken | None = None,
) -> list[float]:
    """Bant basina olcum tabani (dB). `bands_hz` kaynak hizinda degerlendirilir."""
    bands = [
        (hz_to_bin(lo, source_rate, size), hz_to_bin(hi, source_rate, size)) for lo, hi in bands_hz
    ]
    clean = _decode_mono(
        ffmpeg,
        path,
        stream_index=stream_index,
        channels=channels,
        sample_rate=source_rate,
        start=start,
        duration=excerpt_s,
        cancel=cancel,
    )
    floors = [math.inf] * len(bands)
    if clean.size < 4 * size:
        return [math.nan] * len(bands)

    if other_rate != source_rate:
        round_trip = _decode_mono(
            ffmpeg,
            path,
            stream_index=stream_index,
            channels=channels,
            sample_rate=source_rate,
            start=start,
            duration=excerpt_s,
            af_chain=(resample.filter_expr(other_rate), resample.filter_expr(source_rate)),
            cancel=cancel,
        )
        floors = [
            combine_floors(f, r)
            for f, r in zip(floors, _band_snr(clean, round_trip, size, bands), strict=True)
        ]

    if fractional_delay != 0.0:
        stft = StreamingStft(size)
        original = stft.push(clean)
        restored = phase_shift(
            StreamingStft(size).push(fractional_shift(clean, -fractional_delay)),
            fractional_delay,
            size,
        )
        floors = [
            combine_floors(f, r)
            for f, r in zip(floors, _spectral_snr(original, restored, bands), strict=True)
        ]
    return floors
