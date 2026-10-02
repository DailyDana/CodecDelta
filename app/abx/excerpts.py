"""ABX icin hizalanmis ve seviyesi esitlenmis kesit cifti.

Plan kurali: oynatma ASLA orijinal dosyalardan yapilmaz. Ayri ayri cozulen
dosyalarda kodlayici gecikmesi (Opus pre-skip 312 ornek) gecislerde kayma, kazanc
farki (0.5 dB) "yuksek olan orijinaldir" ipucu yaratir. Burada:

- B, karsilastirmanin gecikme modeliyle (kesirli gecikme ve saat kaymasi
  dahil) A'nin zaman ekseninde pencereli sinc ile orneklenir; ayni anda iki
  kaynak ayni ani calar.
- B'nin kazanci ve polaritesi karsilastirmanin olctugu degerle esitlenir.
- Iki taraf AYNI olcekle tepe guvenligine alinir; seviye farki dogmaz.
- Iki dosya ayni hizda okunur (cihazin hizi); ikisi ayni sekilde yeniden
  ornekledigi icin bu bir ipucu degildir.

Qt IMPORT ETMEZ.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.compare import tracking
from app.compare.pipeline import Track
from app.compare.reader import FFmpegWindowReader
from app.compare.result import ComparisonResult
from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, ResampleCfg
from app.dsp import warp

DEFAULT_LENGTH_S = 15.0
# Kesit dosya kenarlarina bu kadar yaklasmaz (s): giris/cikis, kodlayici
# gecikmesi ve warp'in iki yandaki ornek ihtiyaci icin pay.
_EDGE_S = 0.5
# Tepe guvenligi: iki taraf birlikte bu seviyeye olceklenir.
_PEAK = 0.98


@dataclass(frozen=True)
class ExcerptAudio:
    # (cerceve, kanal) float32, ayni uzunlukta ve ayni hizda.
    a: np.ndarray
    b: np.ndarray
    rate: int
    # Referansin zaman ekseninde baslangic (s).
    start_s: float
    # B'ye uygulanan kazanc (dB) ve iki tarafa uygulanan ortak olcek.
    b_gain_db: float
    scale: float

    @property
    def length_s(self) -> float:
        return float(self.a.shape[0]) / self.rate


def valid_range(result: ComparisonResult, length_s: float) -> tuple[float, float] | None:
    """Referans zamaninda kesitin baslayabilecegi aralik; sigmiyorsa None."""
    ref_duration = result.reference.duration
    test_duration = result.test.duration
    if not ref_duration or not test_duration or result.analysis_rate <= 0:
        return None
    delay_s = result.plan.delay_samples / result.analysis_rate
    if math.isnan(delay_s):
        return None
    lo = max(0.0, -delay_s) + _EDGE_S
    hi = min(ref_duration, test_duration - delay_s) - _EDGE_S - length_s
    return (lo, hi) if hi >= lo else None


def random_start(result: ComparisonResult, length_s: float, rng: random.Random) -> float | None:
    span = valid_range(result, length_s)
    if span is None:
        return None
    return rng.uniform(*span)


def critical_start(result: ComparisonResult, length_s: float) -> float | None:
    """NMR haritasinda en kotu `length_s`'lik pencerenin baslangici.

    NMR sutunlari test zaman eksenindedir; referans zamanina gecikmeyle
    cevrilir. Her sutunun en kotu bandi alinir, pencere boyunca ortalanir.
    Duzenlenmis dosyada (hizasiz bloklar olcume girmedi) sutun zamani kayabilir;
    sonuc yine gecerli bir aralikta kalir.
    """
    span = valid_range(result, length_s)
    nmr = result.nmr
    if span is None:
        return None
    if nmr is None or not nmr.grid_db.size or nmr.seconds_per_column <= 0:
        return (span[0] + span[1]) / 2.0
    worst = np.nanmax(nmr.grid_db, axis=0)
    worst = np.where(np.isfinite(worst), worst, -np.inf)
    width = max(1, round(length_s / nmr.seconds_per_column))
    delay_s = result.plan.delay_samples / result.analysis_rate
    best_start, best_score = None, -math.inf
    for column in range(0, max(1, worst.size - width + 1)):
        start_s = column * nmr.seconds_per_column - delay_s
        if not span[0] <= start_s <= span[1]:
            continue
        window = worst[column : column + width]
        finite = window[np.isfinite(window)]
        score = float(finite.mean()) if finite.size else -math.inf
        if score > best_score:
            best_start, best_score = start_s, score
    return best_start if best_start is not None else (span[0] + span[1]) / 2.0


def prepare(
    ffmpeg: Path,
    reference: Track,
    test: Track,
    result: ComparisonResult,
    start_s: float,
    length_s: float = DEFAULT_LENGTH_S,
    *,
    rate: int | None = None,
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    cancel: CancelToken | None = None,
) -> ExcerptAudio:
    """Kesit ciftini cozer, hizalar ve seviyesini esitler.

    `rate`: oynatma hizi (cihazin hizi); verilmezse analiz hizi.
    """
    analysis_rate = result.analysis_rate
    rate = rate or analysis_rate
    model = tracking.from_plan(result.plan, analysis_rate)
    if model is None:
        raise ValueError("karsilastirmanin gecikme modeli yok")
    scale = rate / analysis_rate
    start = round(start_s * rate)
    count = round(length_s * rate)
    if start < 0 or count <= 0:
        raise ValueError("gecersiz kesit")

    read_reference = FFmpegWindowReader(
        ffmpeg, reference.source(rate), resample=resample, cancel=cancel
    )
    read_test = FFmpegWindowReader(ffmpeg, test.source(rate), resample=resample, cancel=cancel)
    a = np.asarray(read_reference(start, count), dtype=np.float64)
    if a.shape[0] < count:
        raise ValueError("kesit referansin sonunu asiyor")

    # Referans ornegi R icin test konumu T: T - d(T) = R  ->  T = R + d(R + d).
    ref_times = start + np.arange(count, dtype=np.float64)
    first_guess = ref_times / scale
    delay = model.at(first_guess + model.at(first_guess)) * scale
    positions = ref_times + delay
    lo = math.floor(float(positions.min())) - warp.HALF_TAPS - 2
    hi = math.ceil(float(positions.max())) + warp.HALF_TAPS + 2
    if lo < 0:
        raise ValueError("kesit testin basindan once basliyor")
    block = np.asarray(read_test(lo, hi - lo), dtype=np.float64)
    if block.shape[0] < hi - lo:
        raise ValueError("kesit testin sonunu asiyor")
    b = np.stack([warp.sample(block[:, c], positions, base=lo) for c in range(block.shape[1])], 1)

    gain = (
        result.polarity * 10.0 ** (result.gain_db / 20.0)
        if result.gain_db == result.gain_db
        else 1.0
    )
    if gain == 0.0:
        gain = 1.0
    b = b / gain
    peak = max(float(np.max(np.abs(a), initial=0.0)), float(np.max(np.abs(b), initial=0.0)))
    common = min(1.0, _PEAK / peak) if peak > 0 else 1.0
    return ExcerptAudio(
        a=(a * common).astype(np.float32),
        b=(b * common).astype(np.float32),
        rate=rate,
        start_s=start / rate,
        b_gain_db=-20.0 * math.log10(abs(gain)),
        scale=common,
    )
