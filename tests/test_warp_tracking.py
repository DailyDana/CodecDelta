"""Surekli yeniden ornekleme ve saat kaymasi izleme testleri."""

from __future__ import annotations

import numpy as np
import pytest

from app.align.envelope import CoarseMatch
from app.align.plan import AlignmentPlan, WindowReader
from app.compare import tracking
from app.compare.tracking import DelayModel
from app.dsp import warp

RNG = np.random.default_rng(0)
_F = RNG.uniform(0.001, 0.45, 300)
_PH = RNG.uniform(0, 2 * np.pi, 300)
_AMP = RNG.uniform(0.2, 1.0, 300)


def analytic(positions: np.ndarray) -> np.ndarray:
    """0.9 Nyquist'e kadar sinuzoid toplami; her konumda TAM degeri bilinir."""
    out = np.zeros(positions.size)
    for i in range(0, _F.size, 50):
        part = slice(i, i + 50)
        out += (
            _AMP[part, None]
            * np.cos(2 * np.pi * _F[part, None] * positions[None, :] + _PH[part, None])
        ).sum(axis=0)
    return out


def _band_error_db(estimate: np.ndarray, truth: np.ndarray, lo: float, hi: float) -> float:
    window = np.hanning(truth.size)
    error = np.abs(np.fft.rfft((estimate - truth) * window)) ** 2
    signal = np.abs(np.fft.rfft(truth * window)) ** 2
    freqs = np.fft.rfftfreq(truth.size)
    band = (freqs >= lo) & (freqs < hi)
    return float(10 * np.log10(error[band].sum() / signal[band].sum()))


def test_warp_accuracy_below_09_nyquist() -> None:
    """Olculen: 8192 faz, 64 tap -> -82..-87 dB (docstring tablosu)."""
    source = analytic(np.arange(120_000, dtype=np.float64))
    times = np.arange(10_000, 110_000, dtype=np.float64)
    positions = times - (3.3 + 200e-6 * times)
    estimate = warp.sample(source, positions)
    truth = analytic(positions)
    for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.35), (0.35, 0.45)):
        assert _band_error_db(estimate, truth, lo, hi) < -80.0


def test_warp_integer_positions_return_the_samples() -> None:
    source = RNG.standard_normal(1000)
    positions = np.arange(100, 900, dtype=np.float64)
    assert np.allclose(warp.sample(source, positions), source[100:900], atol=1e-12)


def test_warp_refuses_to_pad_silently() -> None:
    """Kenarda sifirla doldurmak sahte bir hata uretir; yetersiz tampon hata olmali."""
    source = np.zeros(200)
    with pytest.raises(ValueError, match="tampon yetersiz"):
        warp.sample(source, np.array([5.5]))
    with pytest.raises(ValueError, match="tampon yetersiz"):
        warp.sample(source, np.array([190.5]))


def test_warp_respects_the_base_offset() -> None:
    source = RNG.standard_normal(1000)
    positions = np.array([5100.0, 5200.25])
    shifted = warp.sample(source, positions, base=5000)
    direct = warp.sample(source, positions - 5000)
    assert np.array_equal(shifted, direct)


def _reader(signal: np.ndarray) -> WindowReader:
    stereo = np.stack([signal, signal], axis=1)
    return lambda start, count: stereo[start : start + count]


def test_refine_model_removes_a_biased_initial_slope() -> None:
    """Plan noktalari yanli olabilir (kayan pencere); yineleme gercek egime iner.

    Gercek: 204.08 ppm. Baslangic modeli 204.126 ppm -- ffmpeg'le uretilmis bir
    dosyada plan noktalarindan gercekten boyle cikti.
    """
    rate = 8000
    n = 60 * rate
    reference = analytic(np.arange(n + 5000, dtype=np.float64) * 0.4)
    true = DelayModel(intercept=12.3, slope=-204.08e-6)
    times = np.arange(n, dtype=np.float64)
    # test[T] = ref[T - d(T)]; kenarlarda warp icin pay birakiliyor
    positions = times[200:-200] - true.at(times[200:-200])
    test = np.zeros(n)
    test[200:-200] = warp.sample(reference, positions)

    start = DelayModel(intercept=12.3 + 0.05, slope=-204.126e-6)
    refined = tracking.refine_model(
        start, _reader(reference), _reader(test), rate, (2000, n - 2000)
    )
    worst = np.max(np.abs(refined.at(times[2000:-2000]) - true.at(times[2000:-2000])))
    assert worst < 2e-3
    assert refined.slope == pytest.approx(true.slope, abs=5e-10)


def test_constant_delay_needs_no_tracking() -> None:
    alignment = AlignmentPlan(
        verdict="aligned",
        delay_s=41.25 / 44100,
        delay_samples=41.25,
        position_s=1.0,
        sample_rate=44100,
        envelope=CoarseMatch(lag_s=0.0, lag_frames=0, rho=1.0, overlap_frames=100),
        drift=None,
        lag=None,
        fine=None,
        polarity=1,
        gain_db=0.0,
        channel_map=(0, 1),
        reasons=(),
        track=((1.0, 41.25),),
    )
    assert tracking.from_plan(alignment, 44100) == DelayModel(41.25)


def test_an_overlap_shorter_than_a_window_keeps_the_model() -> None:
    """Kisa klipte negatif pencere baslangici `ValueError` firlatiyordu (D9)."""
    rate = 8000
    signal = analytic(np.arange(rate // 2, dtype=np.float64) * 0.4)
    loose = _reader(signal)

    def strict(start: int, count: int) -> np.ndarray:
        # `FFmpegWindowReader` sozlesmesi: negatif baslangic hatadir.
        if start < 0 or count < 0:
            raise ValueError("start ve count negatif olamaz")
        return loose(start, count)

    # Test referanstan onde (negatif gecikme): referans penceresi pozitif kalirken
    # test baslangici negatife dusuyordu.
    start = DelayModel(intercept=-2000.0, slope=-200e-6)
    refined = tracking.refine_model(start, strict, strict, rate, (100, 3000))
    assert refined == start
