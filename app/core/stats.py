"""ABX icin gereken istatistik. Kapali form, saf `math`, scipy gerektirmez.

Buradaki fonksiyonlarin hepsi bir tek amaca hizmet ediyor: dinleme testinin
sonucunu DURUSTCE ifade etmek. Ozellikle iki yaygin hata koda gomulu olarak
engellenir:

1. "Gecemedim, demek ki fark yok." Yanlis. Bir testi gecememek bir kanit degil,
   kanit yoklugudur. Negatif sonuc ASLA "fark yok" diye degil, bir UST SINIR
   olarak raporlanir: `TestOutcome.max_discrimination` "ayirt etme orani en
   fazla %X" der. `power_at` ise testin ne kadar guclu oldugunu testten ONCE
   soyler.

2. "p<0.05 gorunce durdum." Yanlis. Sansli bir seri her zaman gelir; naif erken
   durdurma gercek alfayi ucurur. Erken durmanin dogru yolu SPRT'dir ve o da
   p degil bir KARAR uretir. Iki mod bilincli olarak ayri tutulur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# SPRT varsayilanlari. H0: denek tahmin ediyor (p=0.5).
# H1: %75 oraninda ayirt ediyor -- ABX literaturunde alisildik "gercek ama
# zorlanilan fark" duzeyi.
SPRT_P0 = 0.5
SPRT_P1 = 0.75
SPRT_ALPHA = 0.05
SPRT_BETA = 0.10


def binomial_p(n: int, k: int) -> float:
    """P(X >= k | n deneme, p=0.5). Tek yonlu tam binom testi.

    Yaklasim yok: `math.comb` tam tamsayi aritmetigi kullanir, bu yuzden
    kucuk n'lerde normal yaklasiminin verdigi hata olusmaz.
    """
    if n < 0:
        raise ValueError("n negatif olamaz")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # 1 << n, 2**n ile ayni ama tamsayi kalir (tur denetimi icin onemli).
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (1 << n)


def min_correct_for_significance(n: int, alpha: float = 0.05) -> int | None:
    """Verilen n icin p < alpha saglayan en kucuk dogru sayisi.

    Arayuz bunu testten ONCE gosterir: "16 denemede 12 dogru gerekiyor" bilgisi,
    denegin bekledigi cabayi gercekci kilar.
    """
    for k in range(n + 1):
        if binomial_p(n, k) < alpha:
            return k
    return None


def power_at(n: int, p_true: float, alpha: float = 0.05) -> float:
    """Gercek ayirt etme orani `p_true` iken testin gucu.

    Guc, "fark gercekten varsa onu yakalayabilme olasiligi"dir. Bu makinede
    hesaplanan degerler (p_true = 0.75, alpha = 0.05):

        n = 16  esik 12  guc 0.630
        n = 20  esik 15  guc 0.617
        n = 24  esik 17  guc 0.766
        n = 30  esik 20  guc 0.894
        n = 40  esik 26  guc 0.946

    n=16'da guc 0.63: gercekten %75 oraninda ayirt eden biri bile denemelerin
    yaklasik ucte birinde testi GECEMEZ. Negatif sonucu "fark yok" diye okumanin
    neden yanlis oldugunu tek basina bu sayi anlatiyor. (n=20'nin n=16'dan daha
    dusuk guc vermesi bir hata degil: binom esigi tamsayidir, 15/20 orani
    12/16'dan daha zorlu.)
    """
    threshold = min_correct_for_significance(n, alpha)
    if threshold is None:
        return 0.0
    total = 0.0
    for i in range(threshold, n + 1):
        total += math.comb(n, i) * float(p_true**i) * float((1.0 - p_true) ** (n - i))
    return total


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Dogruluk oraninin Wilson skor guven araligi (varsayilan %95).

    Wald araligi degil: k=n veya k=0 gibi uc degerlerde Wald sifir genislikte
    bir aralik verir, ki bu sacmadir. Wilson kucuk orneklemde de anlamlidir.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def sidak_alpha(alpha: float, comparisons: int) -> float:
    """Coklu karsilastirma icin duzeltilmis alfa.

    Kullanici ayni cift icin 7 farkli kesit denerse etkin alfa %5 degil ~%30'dur.
    Arayuz denenen kesit sayisini sayar ve duzeltilmis esigi gosterir.
    """
    if comparisons <= 1:
        return alpha
    return 1.0 - float((1.0 - alpha) ** (1.0 / comparisons))


SprtDecision = Literal["continue", "accept_h1", "accept_h0", "truncated"]


@dataclass(frozen=True)
class SprtBounds:
    """Wald sirali olasilik orani testinin esikleri ve adim agirliklari."""

    upper: float
    lower: float
    step_correct: float
    step_wrong: float

    @staticmethod
    def build(
        p0: float = SPRT_P0,
        p1: float = SPRT_P1,
        alpha: float = SPRT_ALPHA,
        beta: float = SPRT_BETA,
    ) -> SprtBounds:
        return SprtBounds(
            upper=math.log((1.0 - beta) / alpha),
            lower=math.log(beta / (1.0 - alpha)),
            step_correct=math.log(p1 / p0),
            step_wrong=math.log((1.0 - p1) / (1.0 - p0)),
        )


def sprt_llr(correct: int, wrong: int, bounds: SprtBounds | None = None) -> float:
    """Birikmis log-olabilirlik orani."""
    b = bounds or SprtBounds.build()
    return correct * b.step_correct + wrong * b.step_wrong


def sprt_decision(
    correct: int,
    wrong: int,
    *,
    bounds: SprtBounds | None = None,
    n_max: int = 50,
) -> SprtDecision:
    """SPRT kararini dondurur.

    `n_max` kesme siniri: sinirsiz SPRT teorik olarak sonlanir ama pratikte
    dinleyici yorulur. Kesilmis SPRT'de n_max'a varildiginda karar verilmemis
    sayilir; bu durumda p degeri UYDURULMAZ, sonuc "belirsiz" olarak raporlanir.
    """
    b = bounds or SprtBounds.build()
    llr = sprt_llr(correct, wrong, b)
    if llr >= b.upper:
        return "accept_h1"
    if llr <= b.lower:
        return "accept_h0"
    if correct + wrong >= n_max:
        return "truncated"
    return "continue"


@dataclass(frozen=True)
class TestOutcome:
    """Bir ABX oturumunun ozeti.

    `max_discrimination` alani bilincli olarak burada: arayuz negatif sonucu
    "fark yok" diye yazamasin diye. Tam sansa esit bir sonuc (orn. 200/400) bile
    p = 0.5'i DISLAYAMAZ; soylenebilecek tek durust sey bir ust sinirdir.
    """

    n: int
    correct: int
    p_value: float
    accuracy: float
    ci_low: float
    ci_high: float
    power: float
    significant: bool
    # Guven araliginin ust ucu. Negatif sonuc bununla ifade edilir:
    # "ayirt etme orani %95 guvenle en fazla %{max_discrimination:.0%}".
    max_discrimination: float


def summarise(n: int, correct: int, *, alpha: float = 0.05, p_true: float = 0.75) -> TestOutcome:
    """Sabit-n bir oturumu ozetler.

    Pozitif sonuc p degeriyle, negatif sonuc UST SINIRLA ifade edilir. Bilincli
    olarak "sonuc kesin mi" diye bir boolean yok: oyle bir bayrak, uydurma bir
    esdegerlik esigi (orn. "%60 altindaysa fark yoktur") gerektirirdi ve bu
    depoda dayanaksiz esik konmuyor.
    """
    p_value = binomial_p(n, correct)
    lo, hi = wilson_ci(correct, n)
    return TestOutcome(
        n=n,
        correct=correct,
        p_value=p_value,
        accuracy=(correct / n) if n else 0.0,
        ci_low=lo,
        ci_high=hi,
        power=power_at(n, p_true, alpha),
        significant=p_value < alpha,
        max_discrimination=hi,
    )
