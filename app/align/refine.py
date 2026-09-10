"""Alt-ornek gecikme kestirimi ve gecerlilik olcusu.

Sozlesme: `docs/SPEC-alignment.md`. Buradaki her sayinin dayanagi
`docs/DECISIONS.md`'deki bir olcumdur.

Tam sayi hizalama yeterli degil. 44.1 kHz'de bir ornek 22.7 us'dir ve yarim
ornek kalan bir kayma, 10 kHz'de 80 dereceye varan faz hatasi demektir; fark
sinyalinde bu, codec gurultusu gibi gorunen buyuk bir artik uretir. Yanlis
gecikme analizi bozmaz, GECERSIZ kilar -- ustelik makul gorunmeye devam ederek.

TEK tahmin edici var: artik enerjisinin en aza indirilmesi. Ikinci bir gecikme
tahmin edicisi ve onu secebilecek bir dal bilincli olarak YOK. Adversarial
denetimde olculdu (12.600 sentetik + gercek muzik + gercek codec cifti): artik
aramasi her veri kumesinde %0.00 basarisiz oldu, faz egimi ise malzemeye gore
%5.67 ile %79.17 arasinda. "Uyusuyorlarsa faz'i al" dali P0 hatasinin ta
kendisiydi; dalin KOSULUNU duzeltmek yerine dali kaldirmak, hatanin geri
gelmesini yapisal olarak imkansiz kiliyor.

Ikinci bir tahmin edici yerine GECERLILIK olculuyor. Gerekce: bir gecikme
tahmin edicisi her zaman bir sayi dondurur. Iliskisiz iki sinyalde iki farkli
tahmin edici iki farkli makul sayi uretir ve ikisi de yanlistir. "Bulunacak bir
gecikme var mi" sorusunu ancak gecerlilik olcusu cevaplayabilir.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.align.gccphat import aligned_slices, correlation_at
from app.align.thresholds import MIN_ALIGNMENT_CORRELATION
from app.dsp.transforms import ensure_signal, fractional_shift

# Faz uydurmasinda kullanilacak normalize frekans bandi (cevrim/ornek, 0..0.5).
DEFAULT_BAND = (0.02, 0.45)

# Dairesel kaydirmanin kenar etkisini disarida birakan pay.
_EDGE_GUARD = 64

# Altin oran aramasi: yakinsama esigi ve ust yineleme siniri.
_SEARCH_TOLERANCE = 1e-4
_SEARCH_MAX_ITER = 80
_INV_PHI = (5.0**0.5 - 1.0) / 2.0

# Arama sinirinin bu kadar yakinina dusen sonuc "kelepcelendi" sayilir.
# Dogru bir tam sayi hizalamadan SONRA kalan gecikme tanim geregi 1 ornekten
# kucuktur; sinira yapisan bir deger ya sozlesme ihlali ya da duz hedef
# fonksiyon (sessizlik, bozuk veri) demektir.
_CLAMP_EPSILON = 1e-3

# Keskinlik olcusunun bakacagi ofset. Yarim ornek: gecikmenin en kucuk anlamli
# belirsizligi kadar.
_SHARPNESS_OFFSET = 0.5

AlignmentStatus = Literal["ok", "weak", "invalid"]


@dataclass(frozen=True)
class SubSampleEstimate:
    """Kalan kesirli gecikme ve ne kadar guvenilebilecegi.

    `delay` tek basina okunmamali. `status` "invalid" iken NaN'dir; "weak" iken
    sayisal olarak dogru olabilir ama uzerine kurulacak bir FARK olcumu anlamli
    olmaz, cunku iki sinyal saf bir zaman kaymasiyla iliskili degildir.
    """

    delay: float
    status: AlignmentStatus
    # Bulunan gecikmede isaretli normalize korelasyon. Gecerli malzemede 0.893
    # ve ustu, iliskisizde 0.485 ve alti olculdu.
    correlation: float
    # artik(d +- 0.5) / artik(d). Yapisi geregi >= 1. RAPORLANIR ama hicbir
    # karar buna baglanmaz: olculen araliklar gecerli/gecersiz arasinda tamamen
    # ortusuyor (bkz. thresholds.py).
    sharpness: float

    @property
    def polarity(self) -> int:
        """Korelasyonun isaretinden polarite."""
        return -1 if self.correlation < 0 else 1

    @property
    def usable(self) -> bool:
        """Bir gecikme hesaplanabildi mi."""
        return self.status != "invalid"

    @property
    def trustworthy(self) -> bool:
        """Uzerine fark olcumu kurulabilir mi.

        "weak" durumunda False: gecikme dogru olabilir ama sinyaller saf bir
        zaman kaymasiyla iliskili degil, yani fark sinyali codec farkini degil
        master farkini olcer.
        """
        return self.status == "ok"


_INVALID = SubSampleEstimate(
    delay=float("nan"), status="invalid", correlation=float("nan"), sharpness=float("nan")
)


def phase_slope_delay(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[float, float]:
    """Capraz spektrumun faz egiminden gecikme. `(gecikme, sabit_terim)` doner.

    `refine` TARAFINDAN CAGRILMIYOR. Modulde kalmasinin sebebi, teknigin ve
    neden kullanilmadiginin kayitli kalmasi.

    Olculen basarisizlik orani (hata > 0.05 ornek):

        bant sinirli gurultu @24 dB        %5.67
        seyrek/tonal spektrum, GURULTUSUZ %31.67
        gercek muzik, GURULTUSUZ          %22.92
        gercek muzik + codec alcak geciren %79.17

    Kok neden `np.unwrap`: bant icindeki tek bir dusuk genlikli bin 2*pi'lik dal
    degisimi yapar ve bu sonraki TUM binlere tasinir. Genlik agirliklandirmasi o
    binin katkisini azaltir ama unwrap zaten iyi binlerin degerini bozmustur.
    Binleri onceden filtrelemek de cozmuyor: bin silmek unwrap'in bitisiklik
    varsayimini bozar ve spektral dropout'ta 150-400 ornek hata verir.
    """
    ensure_signal(reference, "reference")
    ensure_signal(test, "test")
    n = min(reference.size, test.size)
    if n < 16:
        return float("nan"), float("nan")
    a = reference[:n].astype(np.float64, copy=False)
    b = test[:n].astype(np.float64, copy=False)

    cross = np.conj(np.fft.rfft(a)) * np.fft.rfft(b)
    freqs = np.fft.rfftfreq(n)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if int(mask.sum()) < 8:
        return float("nan"), float("nan")

    weights = np.abs(cross[mask])
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return float("nan"), float("nan")

    x = 2.0 * np.pi * freqs[mask]
    y = np.unwrap(np.angle(cross[mask]))

    sum_x = float(np.dot(weights, x))
    sum_y = float(np.dot(weights, y))
    sum_xx = float(np.dot(weights, x * x))
    sum_xy = float(np.dot(weights, x * y))
    denominator = total_weight * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return float("nan"), float("nan")

    slope = (total_weight * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / total_weight
    # angle(conj(A)B) = -omega * gecikme, yani gecikme egimin negatifi.
    return -slope, intercept


def _golden_section_max(objective: Callable[[float], float], low: float, high: float) -> float:
    """[low, high] araliginda `objective`i en buyuten noktayi arar."""
    c = high - _INV_PHI * (high - low)
    d = low + _INV_PHI * (high - low)
    fc, fd = objective(c), objective(d)
    for _ in range(_SEARCH_MAX_ITER):
        if high - low < _SEARCH_TOLERANCE:
            break
        if fc > fd:
            high, d, fd = d, c, fc
            c = high - _INV_PHI * (high - low)
            fc = objective(c)
        else:
            low, c, fc = c, d, fd
            d = low + _INV_PHI * (high - low)
            fd = objective(d)
    return (low + high) / 2.0


def residual_min_delay(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    search: float = 1.0,
) -> float:
    """Artigi en aza indiren kesirli gecikmeyi arar. TEK tahmin edici.

    En iyi kazanc her delta icin ayri secildiginde artik enerjisi
    `||b||^2 - <a_d,b>^2 / ||a_d||^2` olur. Dairesel kaydirma `||a_d||`'yi
    degistirmedigi icin artigi en aza indirmek `|<a_d, b>|`'yi en buyutmekle
    ayni sey -- yani her adimda tek bir ic carpim yeter.

    Arama sinirina yapisan sonuc NaN doner; eski surum sessizce +1.0
    donduruyordu ve bu, bozuk girdiyi guvenilir bir gecikme gibi gosteriyordu.
    """
    ensure_signal(reference, "reference")
    ensure_signal(test, "test")
    n = min(reference.size, test.size)
    if n < 4 * _EDGE_GUARD:
        return float("nan")
    a = reference[:n].astype(np.float64, copy=False)
    b = test[:n].astype(np.float64, copy=False)
    if float(np.dot(a, a)) <= 0.0 or float(np.dot(b, b)) <= 0.0:
        return float("nan")

    core = slice(_EDGE_GUARD, n - _EDGE_GUARD)
    target = b[core]

    def objective(delta: float) -> float:
        return abs(float(np.dot(fractional_shift(a, delta)[core], target)))

    result = _golden_section_max(objective, -search, search)
    if abs(abs(result) - search) <= _CLAMP_EPSILON:
        return float("nan")
    return result


def _residual_energy(shifted: np.ndarray, target: np.ndarray) -> float:
    """En iyi kazanc secildikten sonra kalan enerji."""
    denominator = float(np.dot(shifted, shifted))
    if denominator <= 0.0:
        return float("inf")
    gain = float(np.dot(shifted, target)) / denominator
    return float(np.sum((target - gain * shifted) ** 2))


def validity(reference: np.ndarray, test: np.ndarray, delay: float) -> tuple[float, float]:
    """Bulunan gecikmenin ne kadar iyi oldugunu olcer.

    `(korelasyon, keskinlik)` doner. Korelasyon "bu iki sinyal gercekten
    iliskili mi", keskinlik "bu gecikme ne kadar iyi tanimlanmis" sorusunu
    cevaplar. Karar YALNIZCA korelasyona baglanir; gerekce thresholds.py'de.
    """
    n = min(reference.size, test.size)
    if n < 4 * _EDGE_GUARD or not math.isfinite(delay):
        return float("nan"), float("nan")
    a = reference[:n].astype(np.float64, copy=False)
    b = test[:n].astype(np.float64, copy=False)
    if float(np.dot(a, a)) <= 0.0 or float(np.dot(b, b)) <= 0.0:
        return float("nan"), float("nan")

    core = slice(_EDGE_GUARD, n - _EDGE_GUARD)
    target = b[core]
    aligned = fractional_shift(a, delay)[core]

    correlation = correlation_at(aligned, target, 0)
    at_peak = _residual_energy(aligned, target)
    off_peak = min(
        _residual_energy(fractional_shift(a, delay - _SHARPNESS_OFFSET)[core], target),
        _residual_energy(fractional_shift(a, delay + _SHARPNESS_OFFSET)[core], target),
    )
    sharpness = off_peak / at_peak if at_peak > 0.0 else float("inf")
    return correlation, sharpness


def refine(
    reference: np.ndarray,
    test: np.ndarray,
    coarse_lag: int,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> SubSampleEstimate:
    """Tam sayi hizalamadan sonra kalan kesirli gecikmeyi ve gecerliligini olcer.

    Donen `delay` KALAN gecikmedir; toplam gecikme `coarse_lag + delay`.

    Bicimsiz girdi (2 boyutlu dizi, NaN/inf) `ValueError` firlatir: bu bir veri
    durumu degil, cagri hatasidir. Bozuk ama sonlu girdi (sessizlik, sabit
    sinyal) hata degil `status="invalid"` uretir.

    `band` su an kullanilmiyor -- yalnizca `phase_slope_delay` icin anlamliydi
    ve o karar yolundan cikti. Cagri yerlerini kirmamak icin imzada duruyor.
    """
    ensure_signal(reference, "reference")
    ensure_signal(test, "test")

    a, b = aligned_slices(reference, test, coarse_lag)
    if a.size < 4 * _EDGE_GUARD:
        return _INVALID

    delay = residual_min_delay(a, b)
    if not math.isfinite(delay):
        return _INVALID

    correlation, sharpness = validity(a, b, delay)
    if not math.isfinite(correlation):
        return _INVALID

    status: AlignmentStatus = "ok" if abs(correlation) >= MIN_ALIGNMENT_CORRELATION else "weak"
    return SubSampleEstimate(
        delay=delay, status=status, correlation=correlation, sharpness=sharpness
    )


def total_delay(
    reference: np.ndarray,
    test: np.ndarray,
    coarse_lag: int,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[float, SubSampleEstimate]:
    """Toplam gecikmeyi (tam sayi + kesir) ve ayrintiyi dondurur.

    Kestirim gecersizse toplam NaN olur; cagiran taraf `estimate.status`
    degerine bakmadan sonucu kullanmamali.
    """
    estimate = refine(reference, test, coarse_lag, band=band)
    if not estimate.usable:
        return float("nan"), estimate
    return coarse_lag + estimate.delay, estimate


def polarity_of(reference: np.ndarray, test: np.ndarray, lag: int) -> int:
    """Hizali segmentlerin korelasyon isaretinden polarite."""
    return -1 if correlation_at(reference, test, lag) < 0 else 1
