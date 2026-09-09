"""ABX istatistigi testleri.

Beklenen degerler el ile hesaplandi; kapali form fonksiyonlarin dogrulugunu
bagimsiz olarak kilitliyorlar.
"""

from __future__ import annotations

import math

import pytest

from app.core.stats import (
    SprtBounds,
    binomial_p,
    min_correct_for_significance,
    power_at,
    sidak_alpha,
    sprt_decision,
    sprt_llr,
    summarise,
    wilson_ci,
)


def test_binomial_p_known_values() -> None:
    # 10 denemede 10 dogru: tek bir sonuc, 2^-10
    assert binomial_p(10, 10) == pytest.approx(1 / 1024)
    # 10 denemede 9 veya daha fazla: (1 + 10) / 1024
    assert binomial_p(10, 9) == pytest.approx(11 / 1024)
    # 16 denemede 12+: plandaki 0.0384 degeri
    assert binomial_p(16, 12) == pytest.approx(0.0384, abs=5e-5)


def test_binomial_p_edges() -> None:
    assert binomial_p(10, 0) == 1.0
    assert binomial_p(10, 11) == 0.0
    assert binomial_p(0, 0) == 1.0


def test_binomial_p_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="negatif"):
        binomial_p(-1, 0)


def test_significance_thresholds_match_plan() -> None:
    """Plandaki tablo: n=16 icin 12 dogru gerekiyor."""
    assert min_correct_for_significance(10) == 9
    assert min_correct_for_significance(16) == 12
    assert min_correct_for_significance(20) == 15
    assert min_correct_for_significance(30) == 20


def test_tiny_n_can_never_reach_significance() -> None:
    """4 denemede hepsini bilmek bile p=0.0625; alfa 0.05'in altina inilemez."""
    assert min_correct_for_significance(4) is None


def test_power_at_sixteen_trials_leaves_a_third_of_real_differences_unfound() -> None:
    """n=16'da guc 0.63 -- gercekten %75 ayirt eden biri ucte bir ihtimalle kalir.

    Bu sayi arayuzde testten ONCE gosterilir; negatif sonucu "fark yok" diye
    okumanin neden yanlis oldugunun sayisal kaniti.
    """
    assert power_at(16, 0.75) == pytest.approx(0.630, abs=5e-3)
    # Daha cok deneme daha cok guc
    assert power_at(30, 0.75) == pytest.approx(0.894, abs=5e-3)
    assert power_at(40, 0.75) > power_at(30, 0.75)
    # Fark buyukse yakalamak kolay
    assert power_at(16, 0.95) > 0.9


def test_power_is_not_monotonic_in_n() -> None:
    """n=20'nin gucu n=16'dan dusuk; bu bir hata degil.

    Binom esigi tamsayidir: n=16'da 12/16 (%75) yeterken n=20'de 15/20 (%75)
    gerekiyor ama dagilimin ayrikligi n=20'yi bir tik daha zorlu yapiyor.
    Arayuz deneme sayisi onerirken bunu bilmeli.
    """
    assert power_at(20, 0.75) < power_at(16, 0.75)


def test_power_at_chance_is_about_alpha() -> None:
    """Gercek oran %50 ise "anlamli" cikma olasiligi alfayi asmamali."""
    assert power_at(16, 0.5) < 0.05


def test_wilson_ci_contains_point_estimate() -> None:
    lo, hi = wilson_ci(13, 16)
    assert lo < 13 / 16 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_is_not_degenerate_at_extremes() -> None:
    """Wald araligi k=n'de sifir genislik verir; Wilson vermez."""
    lo, hi = wilson_ci(16, 16)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo < 1.0
    assert lo > 0.75


def test_wilson_ci_of_chance_result_straddles_half() -> None:
    """10/20 sonucu %50'yi iceren bir aralik verir -> BELIRSIZ."""
    lo, hi = wilson_ci(10, 20)
    assert lo < 0.5 < hi


def test_sidak_alpha_shrinks_with_more_comparisons() -> None:
    assert sidak_alpha(0.05, 1) == 0.05
    assert sidak_alpha(0.05, 7) < 0.05
    # 7 kesit denendiyse etkin esik kabaca %0.7
    assert sidak_alpha(0.05, 7) == pytest.approx(0.00730, abs=1e-4)


def test_sprt_bounds_match_plan() -> None:
    b = SprtBounds.build()
    assert b.upper == pytest.approx(2.8904, abs=1e-4)
    assert b.lower == pytest.approx(-2.2513, abs=1e-4)
    assert b.step_correct == pytest.approx(math.log(1.5), abs=1e-9)
    assert b.step_wrong == pytest.approx(math.log(0.5), abs=1e-9)


def test_sprt_accepts_h1_after_eight_straight_correct() -> None:
    """Plandaki en hizli pozitif: 8 ardisik dogru."""
    assert sprt_decision(7, 0) == "continue"
    assert sprt_decision(8, 0) == "accept_h1"


def test_sprt_keeps_going_while_evidence_is_thin() -> None:
    """5 dogru 5 yanlis LLR'yi -1.44'e getirir; alt esik -2.25, henuz karar yok."""
    assert sprt_llr(5, 5) == pytest.approx(-1.438, abs=5e-3)
    assert sprt_decision(5, 5) == "continue"


def test_sprt_accepts_h0_when_guessing() -> None:
    """6 dogru 10 yanlis -> LLR -4.50, alt esigin altinda."""
    assert sprt_decision(6, 10) == "accept_h0"
    assert sprt_llr(6, 10) < SprtBounds.build().lower


def test_sprt_truncates_at_n_max() -> None:
    """Kesme sinirinda karar verilmemis sayilir; p uydurulmaz."""
    # Iki esik arasinda kalan bir dizi
    assert sprt_decision(30, 20, n_max=50) == "truncated"


def test_summarise_positive_result() -> None:
    out = summarise(16, 13)
    assert out.significant
    assert out.p_value < 0.05
    assert out.accuracy == pytest.approx(13 / 16)


def test_summarise_chance_result_is_not_significant() -> None:
    out = summarise(20, 10)
    assert not out.significant
    assert out.ci_low < 0.5 < out.ci_high


def test_negative_result_is_reported_as_an_upper_bound() -> None:
    """Sansa esit skor p=0.5'i DISLAYAMAZ; soylenebilen tek sey ust sinirdir.

    200/400 gibi ideal bir "ayirt edemedi" sonucunda bile guven araligi %50'yi
    icerir. Bu yuzden `summarise` "sonuc kesin mi" diye bir boolean uretmez;
    negatif sonuc daima `max_discrimination` ile ifade edilir. Daha cok deneme
    o siniri daraltir -- ama asla sifira indirmez.
    """
    few = summarise(20, 10)
    many = summarise(400, 200)
    assert not few.significant and not many.significant
    assert many.ci_low < 0.5 < many.ci_high
    # Ust sinir daralir: 20 denemede ~%73, 400 denemede ~%55
    assert many.max_discrimination < few.max_discrimination
    assert many.max_discrimination == pytest.approx(0.549, abs=5e-3)
