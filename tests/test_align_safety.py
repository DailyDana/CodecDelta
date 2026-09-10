"""Hizalama secim guvenligi testleri.

Bu dosyanin tamami adversarial denetimde bulunan hatalarin regresyon kilididir.
Her test, denetimde OLCULEN bir basarisizligi temsil ediyor; gecmesi "kod
calisiyor" degil, "o hata geri gelmedi" demek.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.align import gccphat, refine
from app.align.refine import SubSampleEstimate, _select
from app.dsp import transforms


def analytic_pair(
    n: int, delay: float, seed: int, partials: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """Sinuzoid toplami ve KAPALI FORMDA gecikmis hali.

    Hicbir kaydirma islemi kullanilmaz: gecikme her bilesenin fazina dogrudan
    islenir. Seyrek/tonal spektrum uretir -- faz egiminin coktugu sinif.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    freqs = rng.uniform(0.02, 0.45, partials)
    amps = rng.uniform(0.2, 1.0, partials)
    phases = rng.uniform(0, 2 * np.pi, partials)
    a = np.zeros(n)
    b = np.zeros(n)
    for f, amp, ph in zip(freqs, amps, phases, strict=True):
        omega = 2 * np.pi * f
        a += amp * np.sin(omega * t + ph)
        b += amp * np.sin(omega * (t - delay) + ph)
    peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
    return a / peak, b / peak


# -- secim sozlesmesi -------------------------------------------------------


def test_residual_is_returned_even_when_methods_agree() -> None:
    """Uyusma durumunda bile artik dondurulur.

    "Uyusuyorlarsa faz'i al" gibi bir dal birakilmadi: o dal, denetimdeki P0
    hatasinin geri gelebilecegi tek yer.
    """
    delay, method, status, _ = _select(phase=0.3001, residual=0.3000)
    assert (delay, method, status) == (0.3000, "residual", "ok")


def test_disagreement_returns_residual_not_phase() -> None:
    """Denetimin P0-2 bulgusu: eski kod burada faz'i donduruyordu."""
    delay, method, status, agreement = _select(phase=3.38, residual=0.30)
    assert delay == 0.30
    assert method == "residual"
    assert status == "disagree"
    assert agreement == pytest.approx(3.08)


@pytest.mark.parametrize("bad_phase", [float("inf"), -float("inf"), 50.0, -50.0])
def test_absurd_phase_never_selected(bad_phase: float) -> None:
    """inf ve aralik disi degerler eski kodda `best` olarak donuyordu."""
    delay, method, _, _ = _select(phase=bad_phase, residual=0.30)
    assert delay == 0.30
    assert method == "residual"


def test_only_residual_available_is_unverified() -> None:
    """Tek yontem calistiysa sonuc DOGRULANMAMISTIR.

    Eski `agree` alani bu durumda True donuyordu, yani capraz kontrol
    yapilamadigi halde yapilmis gibi gorunuyordu.
    """
    delay, method, status, agreement = _select(phase=float("nan"), residual=0.30)
    assert (delay, method, status) == (0.30, "residual", "unverified")
    assert np.isnan(agreement)


def test_only_phase_available_is_unverified() -> None:
    delay, method, status, _ = _select(phase=0.30, residual=float("nan"))
    assert (delay, method, status) == (0.30, "phase", "unverified")


def test_nothing_available_is_invalid() -> None:
    delay, method, status, _ = _select(phase=float("nan"), residual=float("nan"))
    assert np.isnan(delay)
    assert (method, status) == ("none", "invalid")


def test_agree_property_is_false_when_unverified() -> None:
    """`agree`, "dogrulandi" demek; "kontrol edemedim" demek DEGIL."""
    verified = SubSampleEstimate(0.3, "residual", "ok", 0.0001, 0.3, 0.3, 0.0)
    unverified = SubSampleEstimate(
        0.3, "residual", "unverified", float("nan"), float("nan"), 0.3, 0.0
    )
    assert verified.agree is True
    assert unverified.agree is False
    assert unverified.usable is True


def test_invalid_is_not_usable() -> None:
    bad = SubSampleEstimate(
        float("nan"), "none", "invalid", float("nan"), float("nan"), float("nan"), float("nan")
    )
    assert bad.usable is False
    assert bad.agree is False


# -- gercek basarisizligin regresyonu ---------------------------------------


def test_sparse_spectrum_regression() -> None:
    """Denetimin minimum yeniden uretim vakasi.

    analytic_pair(16384, 0.30, seed=1): faz egimi GURULTUSUZ veride 1.3298
    donduruyor (gercek 0.30). Eski secim mantigi bu degeri kullanirdi.
    """
    a, b = analytic_pair(1 << 14, 0.30, seed=1)
    a, b = a[2048:-2048], b[2048:-2048]
    est = refine.refine(a, b, 0)

    # Faz gercekten bozuk olmali; degilse test artik dogru seyi olcmuyordur
    assert abs(est.phase_slope - 0.30) > 0.5, "faz beklenenden saglam, test guncellenmeli"
    assert est.status == "disagree"
    assert est.method == "residual"
    assert est.delay == pytest.approx(0.30, abs=0.01)


def test_sparse_spectrum_many_seeds() -> None:
    """Seyrek spektrumda secilen deger her tohumda dogru olmali."""
    errors = []
    for seed in range(25):
        a, b = analytic_pair(1 << 13, -0.37, seed=seed)
        a, b = a[1024:-1024], b[1024:-1024]
        est = refine.refine(a, b, 0)
        assert est.usable
        errors.append(abs(est.delay - (-0.37)))
    assert max(errors) < 0.01, f"en kotu hata {max(errors):.4f}"


# -- bicimsiz girdi ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan, 0.0, 1.0] * 200),
        np.array([np.inf, 0.0, 1.0] * 200),
        np.array([-np.inf, 0.0, 1.0] * 200),
    ],
)
def test_non_finite_input_raises(bad: np.ndarray) -> None:
    """Denetimde bu girdiler `residual=+1.0, agree=True` uretiyordu.

    Bozuk ses, guvenilir gorunen tam olarak 1 ornekli bir gecikmeye donusuyordu.
    """
    good = np.sin(np.arange(600) * 0.1)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        refine.refine(bad, good, 0)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        gccphat.estimate(bad, good)


def test_two_dimensional_input_raises_typed_error() -> None:
    """open_pcm bloklari (frames, channels) verir; bu dogal bir cagri hatasi.

    Eski kod `IndexError: boolean index did not match` veriyordu.
    """
    block = np.zeros((4096, 2))
    with pytest.raises(ValueError, match="tek boyutlu"):
        refine.refine(block, block, 0)
    with pytest.raises(ValueError, match="tek boyutlu"):
        gccphat.estimate(block, block)
    with pytest.raises(ValueError, match="tek boyutlu"):
        transforms.ensure_signal(block)


def test_empty_input_is_allowed() -> None:
    """Bos dizi bicimsiz degil; kontrollu sonuc vermeli."""
    empty = np.array([])
    assert gccphat.estimate(empty, empty).lag == 0
    assert refine.refine(empty, empty, 0).status == "invalid"


# -- bozuk ama sonlu girdi --------------------------------------------------


@pytest.mark.parametrize(
    ("name", "signal"),
    [
        ("tumu sifir", np.zeros(4096)),
        ("sabit DC", np.ones(4096)),
        ("cok kucuk", np.full(4096, 1e-300)),
    ],
)
def test_degenerate_input_is_invalid_not_confident(name: str, signal: np.ndarray) -> None:
    """Enerjisiz girdi hata degil, `invalid` uretmeli.

    Denetimde ucu de `residual=+1.0, agree=True` donduruyordu: downstream
    tam olarak 1 ornekli, dogrulanmis gorunen bir gecikme goruyordu.
    """
    est = refine.refine(signal, signal, 0)
    assert est.status == "invalid", name
    assert np.isnan(est.delay), name
    assert est.usable is False, name


def _band_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n) > 0.4] = 0.0
    return np.fft.irfft(spec, n)


def test_flat_objective_returns_nan_not_boundary() -> None:
    """Hedef fonksiyon duz oldugunda arama sinira surunur; bu dondurulmemeli.

    Denetimde olculdu: enerjisiz girdide altin oran aramasi tam olarak +1.0
    donduruyordu ve `agree=True` ile birlestiginde downstream bunu gecerli bir
    gecikme saniyordu.
    """
    assert np.isnan(refine.residual_min_delay(np.zeros(4096), np.zeros(4096)))
    assert np.isnan(refine.residual_min_delay(np.ones(4096), np.ones(4096)))


def test_out_of_range_delay_is_a_known_gap() -> None:
    """BILINEN SINIR: arama araligi disindaki gecikme yakalanmiyor.

    `residual_min_delay` [-1, +1] arasinda arar. Gercek gecikme disarida
    kalirsa sonuc her zaman sinira yapismaz -- otokorelasyonun yan lobu
    aramanin icinde yerel bir tepe olusturabilir ve makul gorunen ama yanlis
    bir deger doner. Olculen ornek: gercek 1.6 -> -0.192.

    Bu, cagiran tarafin dogru tam sayi hizalama vermesiyle onlenir (sozlesme
    `refine`in docstring'inde). Kilometre tasi 3'te kaba/ince hizalama
    baglandiginda zorlanacak; test o zamana kadar siniri BELGELIYOR.
    """
    x = _band_noise(1 << 13, 3)
    y = transforms.fractional_shift(x, 1.6)
    a, b = x[1024:-1024], y[1024:-1024]
    result = refine.residual_min_delay(a, b)
    assert not np.isnan(result)
    assert abs(result - 1.6) > 1.0, "yakalanamayan durum artik yakalaniyorsa test guncellenmeli"


def test_total_delay_propagates_invalid() -> None:
    total, est = refine.total_delay(np.zeros(4096), np.zeros(4096), 10)
    assert np.isnan(total)
    assert est.status == "invalid"


def test_total_delay_adds_coarse_lag() -> None:
    """Sinyaller gercekten 7.25 ornek kaydirilmis olmali.

    Ilk yazimda yalnizca 0.25 kaydirip coarse_lag=7 verilmisti; o durumda
    `aligned_slices` sinyalleri 7 ornek AYIRIYOR ve kalan gecikme -6.75
    oluyor, yani arama araliginin disina cikiyor.
    """
    x = _band_noise(1 << 14, 9)
    y = transforms.fractional_shift(x, 7.25)
    a, b = x[2048:-2048], y[2048:-2048]
    total, est = refine.total_delay(a, b, 7)
    assert est.usable
    assert est.delay == pytest.approx(0.25, abs=0.01)
    assert total == pytest.approx(7.25, abs=0.01)
