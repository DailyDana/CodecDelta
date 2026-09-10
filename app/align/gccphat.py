"""GCC-PHAT ile tam sayi gecikme kestirimi.

Neden duz capraz korelasyon degil: muzikte enerji birkac oktava yigilir ve duz
korelasyonun tepesi genis, yayvan olur -- bir bas davulun periyoduna denk gelen
yanlis tepeler gercek tepeyle yarisir. PHAT agirliklandirmasi genligi
beyazlatip yalnizca faz bilgisini birakir; sonuc, tek ornek genisliginde keskin
bir tepedir.

Bedeli: PHAT genlik bilgisini yok eder, yani tepe yuksekligi "ne kadar
benziyorlar" sorusunu CEVAPLAMAZ. Bu yuzden modul iki isi ayirir:

    konum  -> PHAT ile bulunur (keskin)
    guven  -> ham sinyallerin hizalanmis Pearson korelasyonundan olculur

Isaret uzlasimi `app.dsp.transforms` ile ayni: pozitif gecikme, `test`in
`reference`e gore GECIKMIS oldugu anlamina gelir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.dsp.transforms import next_fast_len

# PHAT bolmesinde sifira bolmeyi engelleyen taban. Mutlak degil goreli:
# sinyal olcegi degistiginde davranis degismemeli.
_PHAT_FLOOR = 1e-12


@dataclass(frozen=True)
class LagEstimate:
    """Bir gecikme kestirimi ve ne kadar guvenilir oldugu."""

    # test'in reference'e gore gecikmesi, ornek cinsinden. Pozitif = test geride.
    lag: int
    # Hizalanmis segmentlerin Pearson korelasyonu. ISARETLI: negatif deger
    # polarite tersligini gosterir.
    correlation: float
    # Tepe / yan lob medyani. Tepenin gercekten one cikip cikmadigini olcer;
    # korelasyondan bagimsizdir ve "bunlar ayni kayit mi" sorusunun asil olcutu.
    psr: float

    @property
    def polarity(self) -> int:
        return -1 if self.correlation < 0 else 1

    @property
    def magnitude(self) -> float:
        return abs(self.correlation)


def aligned_slices(
    reference: np.ndarray, test: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Verilen gecikmeye gore ortusen bolumleri dondurur.

    `test[t] = reference[t - lag]` varsayimiyla: pozitif lag'de test'in basi
    kirpilir, negatifte reference'in basi.
    """
    n = min(reference.size, test.size)
    if lag >= 0:
        span = n - lag
        if span <= 0:
            return reference[:0], test[:0]
        return reference[:span], test[lag : lag + span]
    span = n + lag
    if span <= 0:
        return reference[:0], test[:0]
    return reference[-lag : -lag + span], test[:span]


def correlation_at(reference: np.ndarray, test: np.ndarray, lag: int) -> float:
    """Belirli bir gecikmede isaretli Pearson korelasyonu."""
    a, b = aligned_slices(reference, test, lag)
    if a.size < 2:
        return 0.0
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    denominator = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(a64, b64) / denominator)


def phat_correlation(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    """PHAT agirlikli capraz korelasyonu dairesel dizi olarak dondurur.

    Indeks k, k ornekli gecikmeye karsilik gelir; negatif gecikmeler dizinin
    sonundan sarar (n-1 = -1 gecikmesi).
    """
    n = next_fast_len(reference.size + test.size)
    spectrum_a = np.fft.rfft(reference, n)
    spectrum_b = np.fft.rfft(test, n)
    # conj(A) * B uzlasimi: tepe dogrudan test'in gecikmesini verir.
    cross = np.conj(spectrum_a) * spectrum_b
    magnitude = np.abs(cross)
    floor = _PHAT_FLOOR * (magnitude.max() if magnitude.size else 1.0)
    cross /= np.maximum(magnitude, floor)
    return np.fft.irfft(cross, n)


def estimate(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    max_lag: int | None = None,
    exclude: int = 0,
) -> LagEstimate:
    """Tam sayi gecikmeyi kestirir.

    `max_lag`: aranacak en buyuk mutlak gecikme. None ise tum aralik.
    `exclude`: PSR hesaplanirken tepenin etrafinda yok sayilacak yariayak
    (ornek). Gercek bir tepe birkac ornek genisligindedir; komsulari yan lob
    sayilirsa PSR yapay olarak dusuk cikar.

    Tepe MUTLAK degere gore aranir; boylece polaritesi ters bir kopya da
    bulunur ve isaret `correlation` alaninda tasinir.
    """
    if reference.size == 0 or test.size == 0:
        return LagEstimate(lag=0, correlation=0.0, psr=0.0)

    correlation = phat_correlation(reference, test)
    n = correlation.size
    limit = min(max_lag if max_lag is not None else n // 2, n // 2)

    # Negatif gecikmeler dizinin sonunda; ikisini tek eksende birlestir.
    positive = correlation[: limit + 1]
    negative = correlation[n - limit :] if limit else correlation[:0]
    window = np.concatenate([negative, positive])
    lags = np.arange(-limit, limit + 1)

    magnitude = np.abs(window)
    peak_index = int(np.argmax(magnitude))
    lag = int(lags[peak_index])

    return LagEstimate(
        lag=lag,
        correlation=correlation_at(reference, test, lag),
        psr=_peak_to_sidelobe(magnitude, peak_index, exclude),
    )


def _peak_to_sidelobe(magnitude: np.ndarray, peak_index: int, exclude: int) -> float:
    """Tepe / yan lob medyani.

    Medyan kullaniliyor cunku ortalama, ikinci bir gercek tepeden (ornegin
    dongusel bir muzik motifi) etkilenir ve olcutu bozar.
    """
    if magnitude.size == 0:
        return 0.0
    peak = float(magnitude[peak_index])
    if peak <= 0.0:
        return 0.0

    low = max(0, peak_index - exclude)
    high = min(magnitude.size, peak_index + exclude + 1)
    sidelobe = np.concatenate([magnitude[:low], magnitude[high:]])
    if sidelobe.size == 0:
        return 0.0
    median = float(np.median(sidelobe))
    if median <= 0.0:
        return float("inf")
    return peak / median
