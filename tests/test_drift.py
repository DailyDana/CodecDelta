"""Hiz orani ve surukelenme testleri.

Yontem her yerde ayni: bilinen bir oran ENJEKTE edilir ve geri bulunmasi
beklenir. Isaret uzlasimi kolay karistirilan bir sey oldugu icin ilk testler
tam olarak onu sabitliyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.align import drift
from app.align.drift import Anchor, collect_anchors, snap, theil_sen

RATE = 8000


def music_like(seconds: float, seed: int = 0, rate: int = RATE) -> np.ndarray:
    """Genis bantli, zengin yapili sinyal.

    GCC-PHAT'in tutunabilmesi icin her pencerede genis bir spektrum sart; duz
    bir sinus periyodik belirsizlik uretir ve test gercek davranisi olcmez.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = rng.standard_normal(n)
    t = np.arange(n) / rate
    for _ in range(4):
        x += rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * rng.uniform(80, 3000) * t)
    return x / np.max(np.abs(x))


def resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """`x`'i `ratio` kati HIZLI calan bir kopya uretir (dogrusal interpolasyon).

    `ratio > 1` -> sonuc daha kisa. PAL'de oldugu gibi perde de kayar; burada
    kasten `atempo` degil `asetrate` davranisi taklit ediliyor.
    """
    n = int(x.size / ratio)
    return np.interp(np.arange(n) * ratio, np.arange(x.size), x)


def shifted(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """`x`'in `delay_samples` kadar GECIKMIS kopyasi (pozitif = geride)."""
    if delay_samples >= 0:
        return np.concatenate([np.zeros(delay_samples), x])[: x.size]
    return np.concatenate([x[-delay_samples:], np.zeros(-delay_samples)])


# -- Theil-Sen --------------------------------------------------------------


def test_theil_sen_recovers_a_clean_line() -> None:
    x = np.linspace(0, 100, 40)
    slope, intercept = theil_sen(x, 3.0 + 0.25 * x)
    assert slope == pytest.approx(0.25)
    assert intercept == pytest.approx(3.0)


def test_theil_sen_survives_a_quarter_of_bad_anchors() -> None:
    """Kirilma noktasi %29; %25 bozuk capa egimi bozmamali.

    Ayni veride en kucuk kareler karsilastiriliyor: farki gormek, yontemin
    neden secildigini kayda geciriyor.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0, 100, 40)
    y = 3.0 + 0.25 * x
    corrupt = rng.choice(x.size, size=10, replace=False)
    y[corrupt] = rng.uniform(-500, 500, size=10)

    slope, _ = theil_sen(x, y)
    assert slope == pytest.approx(0.25, abs=0.01)

    least_squares = float(np.polyfit(x, y, 1)[0])
    assert abs(least_squares - 0.25) > 0.1


def test_theil_sen_handles_degenerate_input() -> None:
    assert theil_sen(np.zeros(0), np.zeros(0)) == (0.0, 0.0)
    assert theil_sen(np.array([5.0]), np.array([2.0])) == (0.0, 2.0)
    # Tum x'ler ayni: egim tanimsiz, sifir donmeli
    assert theil_sen(np.full(6, 3.0), np.arange(6.0))[0] == 0.0


def test_theil_sen_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="ayni uzunlukta"):
        theil_sen(np.zeros(4), np.zeros(5))


# -- capa toplama -----------------------------------------------------------


def test_anchor_lag_sign_matches_the_convention() -> None:
    """Pozitif gecikme = test GERIDE. Isaret hatasi tum modulu bozar."""
    source = music_like(20.0, seed=1)
    delay = 400  # ornek
    anchors = collect_anchors(source, shifted(source, delay), RATE, count=8)

    assert len(anchors) >= 6
    measured = np.median([a.lag_s for a in anchors])
    assert measured == pytest.approx(delay / RATE, abs=1e-3)


def test_anchors_are_flat_when_there_is_no_drift() -> None:
    source = music_like(20.0, seed=2)
    anchors = collect_anchors(source, source.copy(), RATE, count=16)
    lags = np.array([a.lag_s for a in anchors])
    assert np.all(np.abs(lags) < 1e-3)


def test_anchors_slope_downwards_when_test_is_faster() -> None:
    """Hizli bir test, ilerledikce reference'a gore ONE gecer -> gecikme AZALIR.

    Bu, `ratio = 1 - slope` turetmesinin ampirik dayanagi; isaret hatasi
    yapmak kolay oldugu icin ayri bir test.
    """
    source = music_like(300.0, seed=3)
    anchors = collect_anchors(source, resample(source, 1.0001), RATE, count=20)
    slope, _ = theil_sen(
        np.array([a.position_s for a in anchors]), np.array([a.lag_s for a in anchors])
    )
    assert slope < 0
    assert slope == pytest.approx(-0.0001, rel=0.05)


def test_anchor_capture_range_is_about_1000_ppm() -> None:
    """Capalarin yakalama araligi bir SINIR; sessizce asilmamali.

    Pencere icinde biriken kayma `window * (ratio - 1)` ornektir. Olculen
    (0.25 s pencere): 1000 ppm'de 1.0 ppm hatayla calisir, 5000 ppm'de hic
    capa bulunamaz. Sinirin ustunde modul "bilmiyorum" demeli, yanlis bir
    oran uydurmamali -- `estimate_from_audio` orada hipotez sinamasina gecer.
    """
    source = music_like(300.0, seed=14)

    inside = collect_anchors(source, resample(source, 1.001), RATE, count=30)
    assert len(inside) >= drift.MIN_ANCHORS
    assert drift.estimate(inside).ratio == pytest.approx(1.001, abs=2e-6)

    outside = collect_anchors(source, resample(source, 1.005), RATE, count=30)
    assert drift.estimate(outside).status == "unreliable"


def test_silent_anchors_are_dropped() -> None:
    """Sessizlige denk gelen pencere anlamsiz gecikme uretir, elenmeli."""
    source = music_like(20.0, seed=4)
    source[int(8 * RATE) : int(14 * RATE)] = 0.0
    anchors = collect_anchors(source, source.copy(), RATE, count=20)
    assert all(not (8.0 < a.position_s < 13.0) for a in anchors)


def test_collect_anchors_validates_input() -> None:
    x = music_like(2.0)
    with pytest.raises(ValueError, match="sample_rate"):
        collect_anchors(x, x, 0)
    with pytest.raises(ValueError, match="en az iki capa"):
        collect_anchors(x, x, RATE, count=1)
    with pytest.raises(ValueError, match="tek boyutlu"):
        collect_anchors(np.zeros((10, 2)), x, RATE)


def test_too_short_input_yields_no_anchors() -> None:
    assert collect_anchors(music_like(0.5), music_like(0.5), RATE, window_s=2.0) == []


# -- oran kestirimi ---------------------------------------------------------


def _anchors_for(ratio: float, offset_s: float = 0.0, count: int = 30) -> list[Anchor]:
    x = np.linspace(0, 600.0, count)
    return [
        Anchor(
            position_s=float(p),
            lag_s=offset_s + (1.0 - ratio) * float(p),
            correlation=0.99,
            psr=30.0,
        )
        for p in x
    ]


def test_identical_speed_reports_none() -> None:
    result = drift.estimate(_anchors_for(1.0, offset_s=0.25))
    assert result.status == "none"
    assert result.ratio == 1.0
    assert result.offset_s == pytest.approx(0.25)
    assert not result.is_resampling


def test_pal_speed_up_is_detected_and_named() -> None:
    """PAL, aracin dogru anlamasi gereken en onemli tek senaryo."""
    result = drift.estimate(_anchors_for(25.0 / 24.0))
    assert result.status == "drift"
    assert result.ratio == pytest.approx(25.0 / 24.0)
    assert result.label is not None and "PAL" in result.label
    assert result.ppm == pytest.approx(41666.7, abs=1.0)
    assert result.is_resampling


def test_pal_slow_down_is_detected() -> None:
    result = drift.estimate(_anchors_for(24.0 / 25.0))
    assert result.ratio == pytest.approx(24.0 / 25.0)
    assert result.label is not None and "PAL" in result.label


def test_ntsc_pulldown_is_detected() -> None:
    result = drift.estimate(_anchors_for(30.0 / 29.97))
    assert result.label is not None and "NTSC" in result.label


def test_measured_ratio_snaps_to_the_exact_rational() -> None:
    """Olculen 1.041663, raporda 1.0416667 olmali.

    Tablo degerleri tam rasyonel sayilardir; olcum gurultusunu raporlamak
    kullaniciya sahte bir hassasiyet gosterir.
    """
    result = drift.estimate(_anchors_for(25.0 / 24.0 - 1e-4))
    assert result.ratio == 25.0 / 24.0


def test_small_clock_drift_is_reported_without_a_label() -> None:
    """50 ppm kristal toleransi gercek ama tabloda yok."""
    result = drift.estimate(_anchors_for(1.00005))
    assert result.status == "drift"
    assert result.label is None
    assert result.ppm == pytest.approx(50.0, abs=1.0)


def test_negligible_drift_is_reported_as_none() -> None:
    """2 ppm, 10 dakikada 1.2 ms. L3'un blok-yerel takibi bunu zaten yutar."""
    result = drift.estimate(_anchors_for(1.000002))
    assert result.status == "none"
    assert result.ratio == 1.0


def test_too_few_anchors_is_unreliable_not_confident() -> None:
    """Az capa "surukelenme yok" DEGIL, "bilmiyorum" demektir."""
    result = drift.estimate(_anchors_for(1.0, count=6))
    assert result.status == "unreliable"
    assert result.anchors == 6
    assert not result.is_resampling


def test_no_anchors_at_all_is_unreliable() -> None:
    result = drift.estimate([])
    assert result.status == "unreliable"
    assert result.anchors == 0
    assert result.offset_s == 0.0


def test_scattered_anchors_are_unreliable_not_a_ratio() -> None:
    """Capalar bir dogru olusturmuyorsa oran uydurulmamali."""
    rng = np.random.default_rng(7)
    anchors = [
        Anchor(position_s=float(p), lag_s=float(rng.uniform(-2, 2)), correlation=0.9, psr=10.0)
        for p in np.linspace(0, 600, 30)
    ]
    result = drift.estimate(anchors)
    assert result.status == "unreliable"
    assert result.residual_ms > 20.0


def test_a_few_bad_anchors_do_not_break_the_fit() -> None:
    anchors = _anchors_for(25.0 / 24.0, count=32)
    rng = np.random.default_rng(11)
    for i in rng.choice(32, size=7, replace=False):
        anchors[i] = Anchor(anchors[i].position_s, float(rng.uniform(-5, 5)), 0.6, 5.0)
    result = drift.estimate(anchors)
    assert result.ratio == pytest.approx(25.0 / 24.0)
    assert result.label is not None


def test_snap_returns_none_for_an_unknown_ratio() -> None:
    assert snap(1.02) is None
    assert snap(1.0) is None
    assert snap(25.0 / 24.0) is not None


# -- uctan uca --------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected_label"),
    [
        (1.0, None),
        (1.00005, None),
        (30.0 / 29.97, "NTSC"),
        (25.0 / 24.0, "PAL"),
        (24.0 / 25.0, "PAL"),
    ],
)
def test_estimate_from_audio_recovers_every_table_ratio(
    ratio: float, expected_label: str | None
) -> None:
    """Modulun asil sinavi: ham sesten oran, PAL dahil.

    PAL (41667 ppm) capalarin yakalama araliginin cok disinda, yani dogrudan
    OLCULEMEZ. Ama surekli bir bilinmeyen de degil -- kisa, ayrik bir tablodan
    gelir, dolayisiyla olculmez SINANIR. Bes durumda da olculen hata 0.0 ppm.
    """
    source = music_like(300.0, seed=12)
    result = drift.estimate_from_audio(source, resample(source, ratio), RATE)

    assert result.ratio == pytest.approx(ratio, abs=2e-6)
    if expected_label is None:
        assert result.label is None
    else:
        assert result.label is not None and expected_label in result.label
    assert result.status == ("none" if ratio == 1.0 else "drift")


def test_estimate_from_audio_reports_a_plain_offset_without_drift() -> None:
    source = music_like(300.0, seed=13)
    result = drift.estimate_from_audio(source, shifted(source, 800), RATE)
    assert result.status == "none"
    assert result.offset_s == pytest.approx(0.1, abs=2e-3)


def test_estimate_from_audio_gives_up_on_unrelated_audio() -> None:
    """Alakasiz ses "surukelenme yok" DEGIL, "bilmiyorum" uretmeli."""
    result = drift.estimate_from_audio(music_like(120.0, seed=20), music_like(120.0, seed=21), RATE)
    assert result.status == "unreliable"
    assert not result.is_resampling


@pytest.mark.parametrize("ratio", [1.0, 25.0 / 24.0])
def test_estimate_from_audio_finds_its_own_coarse_offset(ratio: float) -> None:
    """Buyuk bir baslangic kaymasi + PAL, disaridan gecikme verilmeden.

    Capa arama yaricapi 0.25 s; 3.7 s'lik kaymayi ancak zarf bulabilir. PAL'de
    tek bir disaridan verilen gecikme zaten tanimsiz, o yuzden her hipotez
    kendi telafi edilmis zarfindan bakiyor. Zarfin tutunabilmesi icin
    dinamigi olan bir sinyal kullaniliyor.
    """
    from tests.test_envelope import dynamic_noise

    source = dynamic_noise(200.0, seed=31)
    offset = int(3.7 * RATE)
    test = resample(np.concatenate([np.zeros(offset), source]), ratio)

    result = drift.estimate_from_audio(source, test, RATE)
    assert result.ratio == pytest.approx(ratio, abs=2e-6)
    assert result.offset_s == pytest.approx(3.7, abs=2e-3)


def test_ambiguity_separates_a_tone_from_broadband_content() -> None:
    """Periyodik sinyalde ikinci tepe ana tepeye esit; genis bantli sinyalde degil.

    Capa gibi: uzun sinyalden kesilmis referans dilimi ve test penceresi; test
    tarafinda codec gurultusu yerine BAGIMSIZ gurultu (ortak gurultu gecikmeyi
    gercekten belirler).
    """
    from app.align import gccphat

    rng = np.random.default_rng(3)
    t = np.arange(40_000) / 8000.0

    def pair(x: np.ndarray, noise_db: float) -> float:
        coded = x + 10 ** (noise_db / 20) * np.std(x) * rng.standard_normal(x.size)
        return gccphat.ambiguity(x[:20_000], coded[3000:19_000])

    noise = rng.standard_normal(t.size)
    lowpassed = np.fft.irfft(np.fft.rfft(noise) * (np.arange(20_001) < 3000), t.size)
    assert pair(np.sin(2 * np.pi * 1000.0 * t), -40) > 0.9
    assert pair(np.sin(2 * np.pi * 440.0 * t), -40) > 0.9
    assert pair(noise, -40) < 0.2
    assert pair(lowpassed, -80) < 0.3
    assert gccphat.ambiguity(np.zeros(100), np.zeros(100)) == 0.0


def test_lazy_windows_match_a_full_rescale() -> None:
    """Pencere pencere hiz telafisi, tum diziyi yeniden orneklemekle ayni (D5)."""
    x = (music_like(5.0, seed=4) * 32767).astype(np.int16)
    ratio = 1.0 / (25.0 / 24.0)
    count = int(x.size / ratio)
    full = np.interp(np.arange(count) * ratio, np.arange(x.size), x.astype(np.float64))
    lazy = drift._Lazy(x, ratio)
    assert lazy.size == count
    for start, stop in ((0, 4000), (12_345, 20_000), (count - 500, count + 100)):
        np.testing.assert_allclose(lazy[start:stop], full[start:stop], rtol=0, atol=1e-9)
    assert np.concatenate(list(lazy.chunks(7000))).size == count


def test_int16_samples_give_the_same_drift_as_floats() -> None:
    """Plan artik int16 ornekleri dogrudan veriyor; hiz tahmini olcekten bagimsiz."""
    source = music_like(60.0, seed=6)
    test = resample(source, 1.0002)
    as_float = drift.estimate_from_audio(source, test, RATE)
    as_int = drift.estimate_from_audio(
        (source * 32767).astype(np.int16), (test * 32767).astype(np.int16), RATE
    )
    assert as_int.status == as_float.status
    assert as_int.ppm == pytest.approx(as_float.ppm, abs=1.0)
