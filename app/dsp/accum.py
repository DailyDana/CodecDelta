"""Capraz spektrum biriktirme -- tek gecisli karsilastirmanin cekirdegi.

Iki hizali sinyalin STFT cercevelerinden bin basina uc toplam tutulur:

    Saa = sum |A|^2      Sbb = sum |B|^2      Sab = sum conj(A) * B

Gecis bitince fark hakkinda sorulabilecek her sey bunlardan turer; ikinci bir
gecis gerekmez. Optimal kazanc bile: `g = Re(sum Sab) / sum Saa`.

En onemli sonuc farkin IKIYE ayrilmasi. Bin basina en iyi dogrusal filtre
`H = Sab / Saa` ile aciklanabilen kisim DOGRUSALDIR -- seviye, EQ, yeniden
ornekleme egimi, faz. Hicbir dogrusal filtrenin aciklayamadigi kalan ise
INKOHERENT'tir -- codec gurultusu, kuantalama, maskelenmis bantlarin
silinmesi:

    inkoherent  = Sbb - |Sab|^2 / Saa
    koherent    = |Sab|^2 / Saa
    skaler artik (tek kazancla) = Sbb - 2 g Re(Sab) + g^2 Saa
    dogrusal    = skaler artik - inkoherent

"Farkin 18 dB'i EQ, gercek codec gurultusu 5.8 dB" diyebilmek bu ayrimdan
gelir. Mansettaki S/N inkoherent bilesenden hesaplanir.

Yanlilik: `inkoherent`, ayni veriye oturtulmus bir filtrenin artigidir ve az
cercevede gercek gurultuyu OLDUGUNDAN KUCUK gosterir (tek cercevede tam sifir:
her bin tek bir karmasik sayiyla mukemmel aciklanir). Beklenen deger kabaca
`(k-1)/k` katidir; `k/(k-1)` ile duzeltilir. Duzeltmenin yeterliligi testte
olculuyor. Gercek kullanimda k binlerce oldugu icin etkisi ihmal edilebilir,
ama birkac cercevelik bir analiz sonsuz S/N raporlamamali.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Duzeltmenin anlamli olmasi icin en az cerceve. Bunun altinda inkoherent guc
# tanimsiz sayilir (NaN), cunku k=1'de kalan tam sifirdir.
MIN_FRAMES = 2


@dataclass(frozen=True)
class BandStats:
    """Bir frekans bandinin toplu capraz spektrum ozeti. Tum guc degerleri toplamdir."""

    lo_bin: int
    hi_bin: int
    frames: int
    reference_power: float
    test_power: float
    # test'in, bin basina dogrusal bir filtreyle reference'tan aciklanabilen gucu.
    coherent_power: float
    # Hicbir dogrusal filtrenin aciklayamadigi guc (yanlilik duzeltilmis).
    incoherent_power: float
    # Yalnizca tek bir skaler kazancla hizalandiginda kalan fark gucu.
    scalar_residual_power: float

    @property
    def linear_power(self) -> float:
        """Farkin EQ/seviye/faz kismi: skaler artik eksi inkoherent."""
        return max(0.0, self.scalar_residual_power - self.incoherent_power)

    @property
    def snr_db(self) -> float:
        """Codec S/N'i: koherent guc / inkoherent guc. Mansettaki sayi bu."""
        return _ratio_db(self.coherent_power, self.incoherent_power)

    @property
    def scalar_snr_db(self) -> float:
        """Duz S/N: yalnizca kazanc esitlenmis fark. EQ farkini da gurultu sayar."""
        return _ratio_db(self.test_power, self.scalar_residual_power)

    @property
    def coherence(self) -> float:
        """Guc agirlikli ortalama koherans (0..1)."""
        if self.test_power <= 0.0:
            return math.nan
        return self.coherent_power / self.test_power


def _ratio_db(signal: float, noise: float) -> float:
    if math.isnan(signal) or math.isnan(noise):
        return math.nan
    if noise <= 0.0:
        return math.inf if signal > 0.0 else math.nan
    if signal <= 0.0:
        return -math.inf
    return 10.0 * math.log10(signal / noise)


class CrossSpectrum:
    """Bin basina Saa, Sbb, Sab biriktirici."""

    def __init__(self, bins: int) -> None:
        if bins <= 0:
            raise ValueError("bin sayisi pozitif olmali")
        self.bins = bins
        self.saa = np.zeros(bins, dtype=np.float64)
        self.sbb = np.zeros(bins, dtype=np.float64)
        self.sab = np.zeros(bins, dtype=np.complex128)
        self.frames = 0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        """(cerceve, bin) seklindeki iki spektrum yiginini ekler."""
        if a.shape != b.shape:
            raise ValueError(f"spektrum sekilleri farkli: {a.shape} vs {b.shape}")
        if a.ndim != 2 or a.shape[1] != self.bins:
            raise ValueError(f"beklenen (cerceve, {self.bins}) sekli, gelen {a.shape}")
        if a.shape[0] == 0:
            return
        self.saa += np.sum(a.real**2 + a.imag**2, axis=0)
        self.sbb += np.sum(b.real**2 + b.imag**2, axis=0)
        self.sab += np.sum(np.conj(a) * b, axis=0)
        self.frames += a.shape[0]

    def merge(self, other: CrossSpectrum) -> None:
        """Baska bir biriktiriciyi ekler (paralel parcalari birlestirmek icin)."""
        if other.bins != self.bins:
            raise ValueError("bin sayilari farkli")
        self.saa += other.saa
        self.sbb += other.sbb
        self.sab += other.sab
        self.frames += other.frames

    def gain(self, lo_bin: int = 0, hi_bin: int | None = None) -> float:
        """Farki en aza indiren GERCEK skaler kazanc. Isaretli: negatif = ters polarite."""
        band = slice(lo_bin, hi_bin)
        denominator = float(self.saa[band].sum())
        if denominator <= 0.0:
            return 0.0
        return float(self.sab[band].real.sum()) / denominator

    def band(self, lo_bin: int, hi_bin: int, *, gain: float | None = None) -> BandStats:
        """`[lo_bin, hi_bin)` bandinin ozeti.

        `gain` skaler artik icin kullanilacak kazanc; verilmezse bandin kendi
        optimal kazanci. Genis bant karsilastirmada tek bir global kazanc
        verilmeli, yoksa her bant kendi seviyesini duzeltir ve seviye farki
        "dogrusal" bilesenden kaybolur.
        """
        if not 0 <= lo_bin < hi_bin <= self.bins:
            raise ValueError(f"gecersiz bant: [{lo_bin}, {hi_bin}) / {self.bins}")
        sl = slice(lo_bin, hi_bin)
        saa, sbb, sab = self.saa[sl], self.sbb[sl], self.sab[sl]
        g = self.gain(lo_bin, hi_bin) if gain is None else gain

        k = self.frames
        coherent = np.zeros_like(saa)
        np.divide(sab.real**2 + sab.imag**2, saa, out=coherent, where=saa > 0.0)
        raw_incoherent = float(np.maximum(sbb - coherent, 0.0).sum())
        incoherent = math.nan if k < MIN_FRAMES else raw_incoherent * k / (k - 1)
        test_power = float(sbb.sum())
        scalar_residual = float(
            max(0.0, test_power - 2.0 * g * float(sab.real.sum()) + g * g * float(saa.sum()))
        )
        return BandStats(
            lo_bin=lo_bin,
            hi_bin=hi_bin,
            frames=k,
            reference_power=float(saa.sum()),
            test_power=test_power,
            coherent_power=(
                max(0.0, test_power - incoherent) if not math.isnan(incoherent) else math.nan
            ),
            incoherent_power=incoherent,
            scalar_residual_power=scalar_residual,
        )


def hz_to_bin(frequency: float, sample_rate: int, size: int) -> int:
    """Frekansi en yakin STFT binine cevirir (0..size/2)."""
    return max(0, min(size // 2, round(frequency * size / sample_rate)))
