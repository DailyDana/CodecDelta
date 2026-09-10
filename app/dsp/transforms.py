"""Hizalama ve karsilastirma icin temel sinyal donusumleri.

Buradaki her fonksiyon saf: girdi diziler, cikti diziler, durum yok. Bu bilincli
-- hizalama katmanindaki hatalarin ayiklanmasi zor oldugu icin cekirdek islemler
tek tek, sentetik verilerle sinanabilir olmali.

Isaret uzlasimi (tum modul boyunca gecerli):

    delta > 0  =>  sinyal GECIKTIRILIR (saga kayar)

`fractional_shift(x, +3)` cikan dizide x'in 3 ornek sonra basladigi anlamina
gelir. Hizalama katmani "b, a'ya gore ne kadar gecikmis" sorusunu bu isaretle
cevaplar.
"""

from __future__ import annotations

import numpy as np

# Cok kucuk pencerelerde FFT tabanli islemler anlamsizlasir; cagri yerlerinin
# sessizce sacma sonuc uretmemesi icin alt sinir.
MIN_FFT_LENGTH = 8


def next_fast_len(n: int) -> int:
    """FFT icin verimli bir uzunluk secer (2'nin kuvveti).

    numpy'nin FFT'si karma uzunluklarda da calisir ama 2'nin kuvvetlerinde
    belirgin sekilde hizlidir ve capraz korelasyonda uzunluk zaten serbest.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fractional_shift(x: np.ndarray, delta: float) -> np.ndarray:
    """Sinyali `delta` ornek geciktirir. Kesirli deger kabul eder.

    Frekans alaninda faz rampasiyla yapilir; bu, tam-bant sinc interpolasyonuna
    esdegerdir ve dogrusal interpolasyonun yuksek frekanslarda getirdigi
    yumusatmayi yapmaz. Hizalama artiklarini olcerken bu fark onemli: kotu bir
    interpolasyon, codec gurultusu sanilacak bir hata uretir.

    DAIRESELDIR: sondan tasan kisim basa doner. Blok blok kullanimda bloklarin
    kenarlarina pay birakilmali.
    """
    n = x.size
    if n < MIN_FFT_LENGTH:
        raise ValueError(f"dizi cok kisa: {n} < {MIN_FFT_LENGTH}")
    if delta == 0.0:
        return x.copy()
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n)
    spectrum *= np.exp(-2j * np.pi * freqs * delta)
    shifted: np.ndarray = np.fft.irfft(spectrum, n)
    return shifted


def to_mono(block: np.ndarray) -> np.ndarray:
    """(frames, channels) blogunu mid kanala indirger.

    Mid = ortalama. Codec gurultusu mid'de en gorunur oldugu icin varsayilan
    analiz kanali budur; side ayrica olculur (bkz. `to_side`).
    """
    if block.ndim == 1:
        return block
    mid: np.ndarray = block.mean(axis=1)
    return mid


def to_side(block: np.ndarray) -> np.ndarray:
    """Stereo blogun side kanali: (L - R) / 2.

    Joint-stereo hasari burada ortaya cikar. Mono girdide sifir doner.
    """
    if block.ndim == 1 or block.shape[1] < 2:
        return np.zeros(block.shape[0], dtype=block.dtype)
    side: np.ndarray = (block[:, 0] - block[:, 1]) * 0.5
    return side


def normalise_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iki diziyi ortak en kisa uzunluga kirpar."""
    n = min(a.size, b.size)
    return a[:n], b[:n]


def optimal_gain(reference: np.ndarray, test: np.ndarray) -> float:
    """Farki en aza indiren olcek carpani: g = <a,b> / <a,a>.

    ISARETLIDIR. Negatif deger polarite tersligini gosterir ve bu bilgi
    kaybedilmemeli: `abs()` alip gecmek, ters polariteli bir dosyayi sessizce
    "cok bozuk" diye raporlamaya yol acar.
    """
    denominator = float(np.dot(reference, reference))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(reference, test)) / denominator


def apply_window(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Cerceve matrisine (k, n) pencere uygular."""
    windowed: np.ndarray = frames * window
    return windowed


def frame_signal(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Sinyali ortusen cercevelere boler; sonuc (k, frame_length).

    `sliding_window_view` kullanildigi icin KOPYASIZDIR: donen dizi x'in
    uzerine oturur. Cagiran taraf x'i degistirirse cerceveler de degisir.
    """
    if x.size < frame_length:
        return np.empty((0, frame_length), dtype=x.dtype)
    windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(x, frame_length)
    return windows[::hop]


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def db(value: float, floor: float = -300.0) -> float:
    """Genligi dB'ye cevirir. Sifir icin -inf yerine sonlu bir taban doner.

    -inf, ortalama/persentil hesaplarini zehirledigi icin bilincli olarak
    sonlu bir degere kelepceleniyor.
    """
    if value <= 0.0:
        return floor
    return max(floor, 20.0 * float(np.log10(value)))
