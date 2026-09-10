"""Alt-ornek gecikme kestirimi.

Tam sayi hizalama yeterli degil. 44.1 kHz'de bir ornek 22.7 us'dir ve yarim
ornek kalan bir kayma, 10 kHz'de 80 dereceye varan faz hatasi demektir; fark
sinyalinde bu, codec gurultusu gibi gorunen buyuk bir artik uretir. Yani alt-
ornek duzeltme bir sik olma meselesi degil, olcumun gecerliligi meselesi.

Iki bagimsiz yontem hesaplanir ve birbirini dogrular:

  faz egimi  -- capraz spektrumun fazina agirlikli dogru uydurma. Tum bandi
                kullanir, kapali form, cok dogru. Faz sarmasina duyarli:
                yalnizca kalan gecikme bir ornekten kucukken guvenilir.
  artik en az -- adayin kaydirilmis halinin test ile ic carpimini en buyuten
                delta'yi altin oran aramasiyla bulur. Yavas ama dogrudan
                ILGILENDIGIMIZ seyi (artik enerjisi) en aza indirir, bu yuzden
                bagimsiz bir ikinci goruse deger.

Parabolik interpolasyon bilincli olarak KULLANILMIYOR. Denendi ve olculdu:
gercek gecikme 0.30 ornekken

    PHAT korelasyonu uzerinde parabolik   -> 0.020
    duz korelasyon uzerinde parabolik     -> 0.216
    faz egimi                             -> 0.29999

Duz korelasyondaki yanlilik ustelik sinyale BAGLI: ayni 0.30 gecikme icin
bant 0.20'ye sinirliyken 0.284, 0.48'e kadar acikken 0.0998 cikti. Sebep,
parabolun sinc benzeri bir tepeye oturmamasi ve tepenin genisliginin bant
genisligiyle degismesi. Yanliligi kalibre edilemeyen bir olcum, bagimsiz
dogrulama gorevi goremez.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.align.gccphat import aligned_slices, correlation_at
from app.dsp.transforms import fractional_shift

# Faz uydurmasinda kullanilacak normalize frekans bandi (cevrim/ornek, 0..0.5).
# DC ve Nyquist yakinlari disarida: oralarda faz gurultulu ve tasiyacagi bilgi
# yok. 0.02-0.45, 44.1 kHz'de kabaca 880 Hz - 19.8 kHz'e denk gelir.
DEFAULT_BAND = (0.02, 0.45)

# Iki yontemin ayrisma esigi (ornek). Ustunde sonuc supheli isaretlenir.
AGREEMENT_TOLERANCE = 0.05

# Artik aramasinda dairesel kaydirmanin kenar etkisini disarida birakan pay.
_EDGE_GUARD = 64

# Altin oran aramasi: yakinsama esigi ve ust yineleme siniri.
_SEARCH_TOLERANCE = 1e-4
_SEARCH_MAX_ITER = 80
_INV_PHI = (5.0**0.5 - 1.0) / 2.0


@dataclass(frozen=True)
class SubSampleEstimate:
    """Alt-ornek kalan gecikme ve iki yontemin uyumu."""

    phase_slope: float
    residual_min: float
    # Faz uydurmasinin sabit terimi. Sifira yakin olmali; pi'ye yakinsa
    # polarite ters demektir ve cagiran taraf bunu bilmeli.
    phase_intercept: float

    @property
    def best(self) -> float:
        """Raporlanacak deger: faz egimi, hesaplanamadiysa artik aramasi."""
        if np.isnan(self.phase_slope):
            return self.residual_min
        return self.phase_slope

    @property
    def agree(self) -> bool:
        """Iki yontem ayni cevabi veriyor mu.

        Ayrisma, sinyallerin gercekten hizalanamadigina veya araya dogrusal
        olmayan bir islem girdigine isaret eder; hukum verilirken bakilmali.
        """
        if np.isnan(self.phase_slope) or np.isnan(self.residual_min):
            return True
        return abs(self.phase_slope - self.residual_min) <= AGREEMENT_TOLERANCE


def phase_slope_delay(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[float, float]:
    """Capraz spektrumun faz egiminden gecikme. `(gecikme, sabit_terim)` doner.

    Uydurma SABIT TERIMLI yapilir. Orijinden gecen bir uydurma daha kesin
    olurdu ama polaritesi ters bir kopyada tum fazlar pi kayar ve orijinden
    gecme zorunlulugu bu kaymayi egime yansitarak yanlis gecikme uretir.
    Sabit terim ayrica dondurulur ki cagiran taraf polariteyi anlayabilsin.

    Yeterli enerjili bin yoksa `(nan, nan)` doner.
    """
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

    # Agirlik olarak capraz spektrumun genligi: dusuk enerjili binlerin fazi
    # gurultuludur ve uydurmayi bozar.
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
    """[low, high] araliginda `objective`i en buyuten noktayi arar.

    Turev gerektirmez ve tek tepeli fonksiyonlarda guvenlidir; artik enerjisi
    minimumun yakininda duzgun ve tek tepeli oldugu icin uygun.
    """
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
    """Artigi en aza indiren kesirli gecikmeyi arar.

    En iyi kazanc her delta icin ayri secildiginde artik enerjisi
    `||b||^2 - <a_d,b>^2 / ||a_d||^2` olur. Dairesel kaydirma `||a_d||`'yi
    degistirmedigi icin artigi en aza indirmek `|<a_d, b>|`'yi en buyutmekle
    ayni sey -- yani her adimda tek bir ic carpim yeter.
    """
    n = min(reference.size, test.size)
    if n < 4 * _EDGE_GUARD:
        return float("nan")
    a = reference[:n].astype(np.float64, copy=False)
    b = test[:n].astype(np.float64, copy=False)
    core = slice(_EDGE_GUARD, n - _EDGE_GUARD)
    target = b[core]

    def objective(delta: float) -> float:
        # Kenar payi, dairesel kaydirmanin bastan sona tasan orneklerini
        # olcumun disinda birakir.
        return abs(float(np.dot(fractional_shift(a, delta)[core], target)))

    return _golden_section_max(objective, -search, search)


def refine(
    reference: np.ndarray,
    test: np.ndarray,
    coarse_lag: int,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> SubSampleEstimate:
    """Tam sayi hizalamadan sonra kalan kesirli gecikmeyi olcer.

    Donen degerler KALAN gecikmedir; toplam gecikme `coarse_lag + best`.
    """
    a, b = aligned_slices(reference, test, coarse_lag)
    if a.size < 16:
        return SubSampleEstimate(float("nan"), float("nan"), float("nan"))

    delay, intercept = phase_slope_delay(a, b, band=band)
    return SubSampleEstimate(
        phase_slope=delay,
        residual_min=residual_min_delay(a, b),
        phase_intercept=intercept,
    )


def total_delay(
    reference: np.ndarray,
    test: np.ndarray,
    coarse_lag: int,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[float, SubSampleEstimate]:
    """Toplam gecikmeyi (tam sayi + kesir) ve ayrintiyi dondurur."""
    estimate = refine(reference, test, coarse_lag, band=band)
    return coarse_lag + estimate.best, estimate


def polarity_of(reference: np.ndarray, test: np.ndarray, lag: int) -> int:
    """Hizali segmentlerin korelasyon isaretinden polarite."""
    return -1 if correlation_at(reference, test, lag) < 0 else 1
