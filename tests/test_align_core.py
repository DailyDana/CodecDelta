"""Hizalama cekirdegi testleri.

Yontem: bilinen bir kaymayi/olcegi/polariteyi sentetik sinyale ENJEKTE et,
sonra geri bulunup bulunmadigina bak. Hizalama katmanindaki hatalar gercek
dosyalarda sessizce yanlis sonuc uretir, bu yuzden dogrulama gercek dosyalarla
degil, cevabi onceden bilinen verilerle yapiliyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.align import gccphat, refine
from app.dsp import transforms


def band_limited_noise(n: int, *, seed: int = 0, high: float = 0.40) -> np.ndarray:
    """Nyquist yakinini bosaltilmis genis bantli gurultu.

    Kesirli kaydirma faz rampasiyla yapildigi icin Nyquist'e dayanan icerik
    kaydirmayi bozar; bant sinirlamak testin OLCTUGU seyi kaydirma hatasi
    yerine algoritmanin dogrulugu yapar.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n) > high] = 0.0
    signal = np.fft.irfft(spectrum, n)
    return signal / np.max(np.abs(signal))


def delayed_pair(
    length: int, delay: int, *, margin: int = 4096, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Gercekci bir cift uretir: ayni kaynaktan kaydirilmis iki pencere.

    Dairesel kaydirma yerine kaynaktan farkli konumlardan dilim alinir; gercek
    dosyalarda durum budur ve kenar etkileri de boylece sinanir.
    """
    source = band_limited_noise(length + 2 * margin, seed=seed)
    reference = source[margin : margin + length]
    start = margin - delay
    test = source[start : start + length]
    return reference, test


# -- transforms -------------------------------------------------------------


def test_fractional_shift_matches_roll_for_integers() -> None:
    x = band_limited_noise(4096)
    shifted = transforms.fractional_shift(x, 5)
    assert np.allclose(shifted, np.roll(x, 5), atol=1e-9)


def test_fractional_shift_sign_delays() -> None:
    """Pozitif delta sinyali GECIKTIRIR: modul boyunca gecerli uzlasim."""
    x = np.zeros(1024)
    x[100] = 1.0
    shifted = transforms.fractional_shift(x, 7)
    assert int(np.argmax(shifted)) == 107


def test_fractional_shift_round_trip() -> None:
    x = band_limited_noise(4096)
    there = transforms.fractional_shift(x, 3.25)
    back = transforms.fractional_shift(there, -3.25)
    assert np.allclose(back, x, atol=1e-9)


def test_fractional_shift_rejects_tiny_arrays() -> None:
    with pytest.raises(ValueError, match="cok kisa"):
        transforms.fractional_shift(np.zeros(4), 1.0)


def test_optimal_gain_is_signed() -> None:
    """Isaret korunmali: negatif kazanc polarite tersligi demek."""
    x = band_limited_noise(4096)
    assert transforms.optimal_gain(x, x * 0.5) == pytest.approx(0.5, abs=1e-9)
    assert transforms.optimal_gain(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_mid_and_side() -> None:
    block = np.array([[1.0, 3.0], [2.0, 0.0]])
    assert np.allclose(transforms.to_mono(block), [2.0, 1.0])
    assert np.allclose(transforms.to_side(block), [-1.0, 1.0])


def test_side_of_mono_is_zero() -> None:
    block = np.array([[1.0], [2.0]])
    assert np.allclose(transforms.to_side(block), [0.0, 0.0])


def test_db_floor_instead_of_negative_infinity() -> None:
    """-inf, ortalama ve persentil hesaplarini zehirler."""
    assert transforms.db(0.0) == -300.0
    assert transforms.db(1.0) == pytest.approx(0.0)


# -- GCC-PHAT ---------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 1, 7, 100, 1000, -1, -7, -100, -1000])
def test_recovers_integer_delay(delay: int) -> None:
    reference, test = delayed_pair(1 << 15, delay)
    estimate = gccphat.estimate(reference, test, max_lag=4000)
    assert estimate.lag == delay
    assert estimate.correlation > 0.99


def test_polarity_inversion_is_found_and_reported() -> None:
    """Ters polariteli kopya bulunmali; isaret korelasyonda tasinmali."""
    reference, test = delayed_pair(1 << 15, 250)
    estimate = gccphat.estimate(reference, -test, max_lag=4000)
    assert estimate.lag == 250
    assert estimate.correlation < -0.99
    assert estimate.polarity == -1
    assert estimate.magnitude > 0.99


def test_gain_change_does_not_move_the_peak() -> None:
    reference, test = delayed_pair(1 << 15, 64)
    quiet = gccphat.estimate(reference, test * 0.01, max_lag=4000)
    assert quiet.lag == 64
    assert quiet.correlation > 0.99


def test_max_lag_bounds_the_search() -> None:
    reference, test = delayed_pair(1 << 15, 900)
    narrow = gccphat.estimate(reference, test, max_lag=100)
    assert abs(narrow.lag) <= 100
    assert narrow.lag != 900


def test_aligned_slices_are_consistent() -> None:
    reference, test = delayed_pair(1 << 14, 300)
    a, b = gccphat.aligned_slices(reference, test, 300)
    assert a.size == b.size > 0
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_empty_input_is_handled() -> None:
    estimate = gccphat.estimate(np.array([]), np.array([]))
    assert estimate.lag == 0
    assert estimate.correlation == 0.0


# -- alt-ornek --------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.1, 0.25, -0.3, 0.5, -0.45])
def test_recovers_fractional_delay(fraction: float) -> None:
    """Kesirli gecikme, tam sayi hizalamadan sonra geri bulunmali."""
    x = band_limited_noise(1 << 15, seed=3)
    shifted = transforms.fractional_shift(x, fraction)
    # Kenarlardaki dairesel tasmayi disarida birak
    a, b = x[2048:-2048], shifted[2048:-2048]

    estimate = refine.refine(a, b, 0)
    assert estimate.delay == pytest.approx(fraction, abs=0.01)
    assert estimate.status == "ok"


def test_combined_integer_and_fractional_delay() -> None:
    x = band_limited_noise(1 << 16, seed=4)
    shifted = transforms.fractional_shift(x, 37.4)
    a, b = x[4096:-4096], shifted[4096:-4096]

    coarse = gccphat.estimate(a, b, max_lag=1000)
    assert coarse.lag == 37
    total, estimate = refine.total_delay(a, b, coarse.lag)
    assert total == pytest.approx(37.4, abs=0.05)
    assert estimate.trustworthy


def test_phase_intercept_reveals_polarity() -> None:
    """Ters polaritede sabit terim pi'ye yakin cikar, egim bozulmaz."""
    x = band_limited_noise(1 << 15, seed=5)
    shifted = transforms.fractional_shift(x, 0.3)
    a, b = x[2048:-2048], -shifted[2048:-2048]

    delay, intercept = refine.phase_slope_delay(a, b)
    assert delay == pytest.approx(0.3, abs=0.02)
    assert abs(abs(intercept) - np.pi) < 0.2


def test_residual_search_is_the_single_estimator() -> None:
    """`refine.delay` dogrudan artik aramasindan gelmeli.

    Ikinci bir tahmin edici ve secim dali bilincli olarak yok; bu test iki
    yolun ayrisamayacagini kilitliyor.
    """
    x = band_limited_noise(1 << 15, seed=7)
    shifted = transforms.fractional_shift(x, -0.37)
    a, b = x[2048:-2048], shifted[2048:-2048]

    direct = refine.residual_min_delay(a, b)
    assert direct == pytest.approx(-0.37, abs=0.01)
    estimate = refine.refine(a, b, 0)
    assert estimate.delay == direct
    assert estimate.trustworthy


def test_residual_search_needs_enough_samples() -> None:
    tiny = np.zeros(64)
    assert np.isnan(refine.residual_min_delay(tiny, tiny))


def test_phase_slope_returns_nan_on_silence() -> None:
    silence = np.zeros(4096)
    delay, intercept = refine.phase_slope_delay(silence, silence)
    assert np.isnan(delay)
    assert np.isnan(intercept)


def test_polarity_helper() -> None:
    reference, test = delayed_pair(1 << 14, 10)
    assert refine.polarity_of(reference, test, 10) == 1
    assert refine.polarity_of(reference, -test, 10) == -1
