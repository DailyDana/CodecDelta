"""Saat kaymasi izleme -- gecikmenin zamanla degisimini modelleme.

Gecikme dogrusal kabul edilir: `d(t) = intercept + slope * t` (ornek, analiz
hizinda, t test icindeki ornek indeksi). Egim ASIRI hassas olmali: 204 ppm'lik
bir kaymada egimin binde biri kadar hatasi 60 s'de 0.5 ornek eder ve kayipsiz
bir kopyanin S/N'ini 166 dB'den 12 dB'e dusurur (olculdu, sentetik).

Plan izleme noktalarini kisa pencerelerde olcer ve o pencerelerin icinde de
kayma vardir (204 ppm'de 4096 ornekte 0.84 ornek). Kayan bir pencerede olculen
gecikme pencere merkezinin degil, ENERJI AGIRLIKLI konumun gecikmesidir; ses
yuksekligi pencere icinde esit dagilmadigi icin her nokta birkac yuzde ornek
sapar. OLCULEN (kayipsiz, ffmpeg ile 204.08 ppm): plan noktalarindan gecen
dogru -204.126 ppm verdi; kaydin uclarinda 0.12 ornek hata ve izlemeli S/N
ancak 28.4 dB.

Cozum yineleme: mevcut modelle referans her pencerede WARP edilir, kalan
gecikme olculur. Warp sonrasi pencere ici kayma model hatasi kadardir (ppm'in
binde biri mertebesinde), yani artik olcumler yansizdir ve pencereler uzun
tutulabilir. Kalan gecikmelerden gecen dogru modele eklenir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.align import gccphat, refine
from app.align.plan import AlignmentPlan, WindowReader
from app.dsp import warp
from app.dsp.transforms import to_mono

# Iyilestirme penceresi (s) ve sayisi. Uzun pencere daha hassas; warp sonrasi
# pencere ici kayma sorun degil.
_REFINE_WINDOW_S = 1.0
_REFINE_POINTS = 9
_MAX_ITERATIONS = 5
# Bu kadar kucuk bir duzeltmede yinelemeyi birak (ornek).
_CONVERGED = 1e-5


@dataclass(frozen=True)
class DelayModel:
    """Test icindeki konumun fonksiyonu olarak gecikme: d(t) = intercept + slope * t."""

    intercept: float
    slope: float = 0.0
    # Son olcum noktalarinin dogrudan en buyuk sapmasi (ornek). Dogrusal
    # olmayan bir kaymayi (sicaklikla degisen saat) gorunur kilar.
    max_residual: float = 0.0

    def at(self, t: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.intercept + self.slope * t
        return result


def from_plan(alignment: AlignmentPlan, rate: int) -> DelayModel | None:
    """Plandan ilk modeli kurar; izleme gerekiyor ama mumkun degilse None."""
    tracking = alignment.drift is not None and alignment.drift.needs_tracking
    if not tracking:
        return DelayModel(alignment.delay_samples)
    if len(alignment.track) < 2:
        return None
    t = np.array([pos for pos, _ in alignment.track]) * rate
    d = np.array([delay for _, delay in alignment.track])
    slope, intercept = np.polyfit(t, d, 1)
    residual = float(np.max(np.abs(d - (intercept + slope * t))))
    return DelayModel(float(intercept), float(slope), residual)


@dataclass(frozen=True)
class _Window:
    """Bir iyilestirme noktasi icin BIR KEZ okunmus pencere cifti."""

    start: int
    test: np.ndarray
    ref: np.ndarray
    ref_base: int


# Pencereler bir kez okunur, yinelemeler ayni veriyi kullanir. Ilk surum her
# yinelemede her noktayi ffmpeg ile yeniden okuyordu: 90 s'lik dosyada izleme
# 21-31 s surdu. Model yinelemeler arasinda ornegin kucuk bir kesri kadar
# degisir; bu pay fazlasiyla yeter.
_READ_MARGIN = warp.HALF_TAPS + 64


def _read_windows(
    model: DelayModel,
    read_reference: WindowReader,
    read_test: WindowReader,
    rate: int,
    span: tuple[int, int],
) -> list[_Window]:
    length = int(_REFINE_WINDOW_S * rate)
    lo, hi = span
    windows: list[_Window] = []
    if hi - lo < length:
        # Ortusme bir pencereden kisa: `linspace` negatif baslangiclar uretip
        # okuyucuyu `ValueError` ile dusuruyordu (denetim D9). Model oldugu
        # gibi kalir.
        return windows
    for start in np.linspace(lo, hi - length, _REFINE_POINTS).astype(np.int64):
        ends = np.array([start, start + length - 1], dtype=np.float64)
        wanted = ends - model.at(ends)
        first = int(np.floor(wanted[0])) - _READ_MARGIN
        stop = int(np.floor(wanted[1])) + _READ_MARGIN
        if first < 0:
            continue
        ref_block = read_reference(first, stop - first)
        test_block = read_test(int(start), length)
        if ref_block.shape[0] < stop - first or test_block.shape[0] < length:
            continue
        windows.append(
            _Window(
                start=int(start),
                test=to_mono(test_block).astype(np.float64),
                ref=to_mono(ref_block).astype(np.float64),
                ref_base=first,
            )
        )
    return windows


def _residuals(model: DelayModel, windows: list[_Window]) -> tuple[np.ndarray, np.ndarray]:
    """Model uygulandiktan sonra kalan gecikme, her pencerenin ortasinda."""
    positions, residuals = [], []
    for window in windows:
        times = window.start + np.arange(window.test.size, dtype=np.float64)
        wanted = times - model.at(times)
        try:
            warped = warp.sample(window.ref, wanted, base=window.ref_base)
        except ValueError:
            # Model okuma payini asacak kadar degisti: bu noktayi birak.
            continue
        lag = gccphat.estimate(warped, window.test, max_lag=4)
        fine = refine.refine(warped, window.test, lag.lag)
        if fine.status != "ok":
            continue
        positions.append(float(window.start) + window.test.size / 2.0)
        residuals.append(lag.lag + fine.delay)
    return np.array(positions), np.array(residuals)


def refine_model(
    model: DelayModel,
    read_reference: WindowReader,
    read_test: WindowReader,
    rate: int,
    span: tuple[int, int],
) -> DelayModel:
    """Modeli warp edilmis pencerelerde olculen kalan gecikmeyle iyilestirir.

    `span`: test icinde ortusen aralik (ornek). Yeterli nokta bulunamazsa model
    oldugu gibi doner.
    """
    windows = _read_windows(model, read_reference, read_test, rate, span)
    for _ in range(_MAX_ITERATIONS):
        t, e = _residuals(model, windows)
        if t.size < 3:
            return model
        slope, intercept = np.polyfit(t, e, 1)
        model = DelayModel(
            model.intercept + float(intercept),
            model.slope + float(slope),
            float(np.max(np.abs(e - (intercept + slope * t)))),
        )
        correction = np.abs(intercept + slope * np.array([span[0], span[1]]))
        if float(correction.max()) < _CONVERGED:
            break
    return model
