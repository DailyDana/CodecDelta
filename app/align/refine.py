"""Alt-ornek gecikme kestirimi ve secim guvenligi.

Tam sayi hizalama yeterli degil. 44.1 kHz'de bir ornek 22.7 us'dir ve yarim
ornek kalan bir kayma, 10 kHz'de 80 dereceye varan faz hatasi demektir; fark
sinyalinde bu, codec gurultusu gibi gorunen buyuk bir artik uretir. Yani alt-
ornek duzeltme bir sik olma meselesi degil, olcumun gecerliligi meselesi.

Iki bagimsiz yontem hesaplanir:

  artik en az -- kaydirilmis adayin test ile ic carpimini en buyuten delta'yi
                 altin oran aramasiyla bulur. Dogrudan ILGILENDIGIMIZ seyi
                 (artik enerjisi) en aza indirir. BIRINCIL yontem.
  faz egimi   -- capraz spektrumun fazina agirlikli dogru uydurma. Ucuz ve
                 temiz veride cok dogru, ama guvenilmez (asagi bak). Yalnizca
                 CAPRAZ KONTROL olarak kullanilir.

Neden artik birincil: adversarial denetimde olculdu (12.600 sentetik + gercek
muzik + gercek codec ciftleri). Faz egiminin basarisizlik orani:

    bant sinirli gurultu, S/N 24 dB      %5.67
    seyrek/tonal spektrum, GURULTUSUZ    %31.67
    gercek muzik, GURULTUSUZ             %12.50
    gercek muzik + codec alcak gecireni  %79.17

Ayni kumelerde artik aramasinin basarisizlik orani her seferinde %0.00 oldu.
Kok neden `np.unwrap`: bant icindeki dusuk genlikli tek bir bin 2*pi'lik dal
degisimi yapar ve bu SONRAKI TUM binlere tasinir; genlik agirliklandirmasi o
binin katkisini azaltir ama unwrap zaten iyi binlerin degerini bozmustur.

Secim kurali bu olcumden cikiyor: iki yontem uyussa bile artik dondurulur.
Faz'i tercih etmek, denetimde bulunan P0 hatasinin ta kendisiydi -- `agree`
bayragi basarisizligi %99.97 dogrulukla yakaliyordu ama secim onu yok sayip
basarisiz tahmin ediciyi donduruyordu.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.align.gccphat import aligned_slices, correlation_at
from app.dsp.transforms import ensure_signal, fractional_shift

# Faz uydurmasinda kullanilacak normalize frekans bandi (cevrim/ornek, 0..0.5).
# DC ve Nyquist yakinlari disarida: oralarda faz gurultulu ve tasiyacagi bilgi
# yok. 0.02-0.45, 44.1 kHz'de kabaca 880 Hz - 19.8 kHz'e denk gelir.
DEFAULT_BAND = (0.02, 0.45)

# Iki yontemin ayrisma esigi (ornek). Bu deger KALIBRE EDILMEDI: saglam
# kosullarda olculen tipik ayrisma ~0.00003, yani 0.05 yaklasik 1500 kat genis
# bir pay birakiyor ve kucuk ama gercek ayrismalari kacirir. Faz tahmin edicisi
# duzeltildiginde olculen dagilimdan yeniden turetilmeli.
AGREEMENT_TOLERANCE = 0.05

# Artik aramasinda dairesel kaydirmanin kenar etkisini disarida birakan pay.
_EDGE_GUARD = 64

# Altin oran aramasi: yakinsama esigi ve ust yineleme siniri.
_SEARCH_TOLERANCE = 1e-4
_SEARCH_MAX_ITER = 80
_INV_PHI = (5.0**0.5 - 1.0) / 2.0

# Arama sinirinin bu kadar yakinina dusen sonuc "kelepcelendi" sayilir.
# Gerekce: dogru bir tam sayi hizalamadan SONRA kalan gecikme tanim geregi
# 1 ornekten kucuktur. Sinira yapisan bir deger ya cagiran taraf yanlis coarse
# lag verdigi ya da hedef fonksiyon duz oldugu (sessizlik, bozuk veri) icindir.
# Denetimde olculdu: tumu-sifir ve NaN girdiler tam olarak +1.0 donduruyordu.
_CLAMP_EPSILON = 1e-3

AlignmentStatus = Literal["ok", "disagree", "unverified", "invalid"]
AlignmentMethod = Literal["residual", "phase", "none"]


@dataclass(frozen=True)
class SubSampleEstimate:
    """Alt-ornek kalan gecikme, hangi yontemden geldigi ve ne kadar guvenilir.

    `delay` tek basina okunmamali: `status` "invalid" iken NaN'dir ve
    "disagree" iken iki yontem ayristigi icin downstream'in ilerleyip
    ilerlemeyecegine karar vermesi gerekir.

    Durumlar bilincli olarak dort ayri deger:

        ok         iki yontem de hesaplandi ve uyustu
        disagree   ikisi de hesaplandi ama ayristi -> artik dondurulur
        unverified yalnizca biri hesaplanabildi -> capraz kontrol YOK
        invalid    kullanilabilir sonuc yok (bos/sessiz/kelepcelenmis)

    "unverified" ile "ok"u ayirmak sart: eski surumde tek bir boolean vardi ve
    "dogrulandi" ile "kontrol edilemedi" ayni degeri aliyordu.
    """

    delay: float
    method: AlignmentMethod
    status: AlignmentStatus
    # |faz - artik|. Ikisi de yoksa NaN. Ham fark; esikle karsilastirilmis hali
    # `status` alaninda.
    agreement: float
    phase_slope: float
    residual_min: float
    # Faz uydurmasinin sabit terimi. Sifira yakin olmali; pi'ye yakinsa
    # polarite ters demektir.
    phase_intercept: float

    @property
    def agree(self) -> bool:
        """Iki yontem de hesaplandi VE uyustu.

        "Kontrol edilemedi" durumunda False doner -- eski `agree` alani orada
        True donuyordu ve bu, dogrulanmamis bir sonucu dogrulanmis gosteriyordu.
        """
        return self.status == "ok"

    @property
    def usable(self) -> bool:
        """Downstream bu degeri kullanabilir mi.

        "disagree" de kullanilabilir sayilir: artik aramasi ayrisma vakalarinin
        tamaminda dogru cikti. Ama cagiran taraf durumu raporlamak zorunda.
        """
        return self.status != "invalid"


def _valid(value: float) -> bool:
    return bool(np.isfinite(value))


def phase_slope_delay(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[float, float]:
    """Capraz spektrumun faz egiminden gecikme. `(gecikme, sabit_terim)` doner.

    GUVENILMEZ -- modul basindaki olcume bak. Yalnizca capraz kontrol olarak
    kullanilmali, birincil kestirim olarak degil.

    Uydurma SABIT TERIMLI yapilir. Orijinden gecen bir uydurma daha kesin
    olurdu ama polaritesi ters bir kopyada tum fazlar pi kayar ve orijinden
    gecme zorunlulugu bu kaymayi egime yansitarak yanlis gecikme uretir.

    Yeterli enerjili bin yoksa `(nan, nan)` doner.
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
    """Artigi en aza indiren kesirli gecikmeyi arar. BIRINCIL yontem.

    En iyi kazanc her delta icin ayri secildiginde artik enerjisi
    `||b||^2 - <a_d,b>^2 / ||a_d||^2` olur. Dairesel kaydirma `||a_d||`'yi
    degistirmedigi icin artigi en aza indirmek `|<a_d, b>|`'yi en buyutmekle
    ayni sey -- yani her adimda tek bir ic carpim yeter.

    Arama sinirina yapisan sonuc NaN doner: bu, ya cagiran tarafin yanlis coarse
    lag verdigi ya da hedef fonksiyonun duz oldugu (sessizlik, sabit sinyal)
    anlamina gelir ve iki durumda da deger anlamsizdir. Eski surum sessizce
    +1.0 donduruyordu.
    """
    ensure_signal(reference, "reference")
    ensure_signal(test, "test")
    n = min(reference.size, test.size)
    if n < 4 * _EDGE_GUARD:
        return float("nan")
    a = reference[:n].astype(np.float64, copy=False)
    b = test[:n].astype(np.float64, copy=False)
    # Enerjisiz girdide hedef fonksiyon duzdur; arama sinira surunur.
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


def _select(phase: float, residual: float) -> tuple[float, AlignmentMethod, AlignmentStatus, float]:
    """Iki kestirimden hangisinin dondurulecegine karar verir.

    Kural: artik gecerliyse HER ZAMAN o dondurulur -- uyusma durumunda bile.
    Faz'i tercih etmek denetimdeki P0 hatasiydi; uyusuyorlarsa zaten fark yok,
    uyusmuyorlarsa artik dogru olan. Boylece "hangi kosulda hangisi" diye bir
    dal kalmiyor ve yanlis tahmin edicinin secilmesi yapisal olarak imkansiz.
    """
    phase_ok = _valid(phase)
    residual_ok = _valid(residual)
    agreement = abs(phase - residual) if (phase_ok and residual_ok) else float("nan")

    if phase_ok and residual_ok:
        status: AlignmentStatus = "ok" if agreement <= AGREEMENT_TOLERANCE else "disagree"
        return residual, "residual", status, agreement
    if residual_ok:
        return residual, "residual", "unverified", agreement
    if phase_ok:
        # Artik hesaplanamadi (cok kisa pencere, enerjisiz veya kelepcelenmis).
        # Faz tek basina guvenilmez, bu yuzden "unverified" -- deger verilir ama
        # dogrulanmamis oldugu acikca isaretlenir.
        return phase, "phase", "unverified", agreement
    return float("nan"), "none", "invalid", agreement


def refine(
    reference: np.ndarray,
    test: np.ndarray,
    coarse_lag: int,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> SubSampleEstimate:
    """Tam sayi hizalamadan sonra kalan kesirli gecikmeyi olcer.

    Donen `delay` KALAN gecikmedir; toplam gecikme `coarse_lag + delay`.

    Bicimsiz girdi (2 boyutlu dizi, NaN/inf) `ValueError` firlatir: bu bir veri
    durumu degil, cagri hatasidir. Bozuk ama sonlu girdi (sessizlik, sabit
    sinyal) hata degil `status="invalid"` uretir.
    """
    ensure_signal(reference, "reference")
    ensure_signal(test, "test")

    a, b = aligned_slices(reference, test, coarse_lag)
    if a.size < 16:
        return SubSampleEstimate(
            delay=float("nan"),
            method="none",
            status="invalid",
            agreement=float("nan"),
            phase_slope=float("nan"),
            residual_min=float("nan"),
            phase_intercept=float("nan"),
        )

    phase, intercept = phase_slope_delay(a, b, band=band)
    residual = residual_min_delay(a, b)
    delay, method, status, agreement = _select(phase, residual)
    return SubSampleEstimate(
        delay=delay,
        method=method,
        status=status,
        agreement=agreement,
        phase_slope=phase,
        residual_min=residual,
        phase_intercept=intercept,
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
    if math.isnan(estimate.delay):
        return float("nan"), estimate
    return coarse_lag + estimate.delay, estimate


def polarity_of(reference: np.ndarray, test: np.ndarray, lag: int) -> int:
    """Hizali segmentlerin korelasyon isaretinden polarite."""
    return -1 if correlation_at(reference, test, lag) < 0 else 1
