"""Enerji zarfi ve kaba eslestirme -- hizalamanin L1 katmani.

Uc katmanli hizalamanin en ustu. Gorevi tek: iki kaydin birbirine gore KABA
konumunu bulmak. Ornek hassasiyeti L2/L3'un isi.

Neden ayri bir katman: 2 saatlik bir konser kaydinda 4 dakikalik bir parcayi
aramak, tam ornekleme hizinda capraz korelasyonla 300 milyon ornekten olusan bir
diziyi taramak demektir. 100 Hz'lik zarf uzerinde ayni arama 720 bin noktadir --
yaklasik 400 kat ucuz.

Ikinci ve daha onemli sebep: zarf, SPEKTRAL farklara dayaniklidir. Farkli master,
farkli EQ, farkli codec, hatta farkli kazanc -- hicbiri enerji zarfinin seklini
belirgin degistirmez, cunku zarf "ne calindigini" degil "ne zaman yuksek
sesliydi"yi olcer. Tam hizdaki korelasyon ise alcak geciren bir kopyada zayiflar.

Zarf, logaritmik enerjidir ve z-skoruna donusturulur. Logaritma dinamigi
sikistirir (tek bir gurultulu vurus tum korelasyonu ele gecirmesin); z-skoru
seviye farkini tumuyle yok eder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import open_pcm
from app.dsp.transforms import ensure_signal

# Zarf icin cozulecek ornekleme hizi. 8 kHz, muzikteki enerji degisimlerini
# yakalamak icin fazlasiyla yeterli ve tam hizin ~%18'i kadar veri uretir.
DEFAULT_RATE = 8000

# Zarf cozunurlugu: 10 ms adim, 20 ms pencere (%50 ortusme). 100 Hz'lik zarf,
# perkusif bir vurusun konumunu ~10 ms'ye kadar belirler; L2 oradan devam eder.
DEFAULT_HOP_MS = 10.0
DEFAULT_WINDOW_MS = 20.0

# Sessiz cerceveler icin taban: en yuksek cercevenin bu kadar altina kelepcelenir.
# Mutlak degil goreli oldugu icin zarf, kaydin seviyesinden bagimsiz kalir.
_FLOOR_DB = 80.0

# Bir gecikmenin degerlendirilebilmesi icin gereken en az ortusme. Kisa zarfin
# yarisi: "kisa dosya uzun dosyanin ICINDE" durumunu dogru modeller ve tek
# cerceve ortusen sahte mukemmel eslesmeleri eler.
_MIN_OVERLAP_FRACTION = 0.5
_MIN_OVERLAP_FRAMES = 50


@dataclass(frozen=True)
class Envelope:
    """Bir kaydin logaritmik enerji zarfi ve (istege bagli) ham ornekleri."""

    # z-skoruna donusturulmus log-enerji, cerceve basina bir deger.
    values: np.ndarray
    hop_hz: float
    sample_rate: int
    duration_s: float
    # Zarfi uretmek icin zaten cozulmus olan mono PCM. int16 saklanir: 2 saatlik
    # bir kayit float32'de 230 MB, int16'da 115 MB tutar ve hizalama icin 96 dB
    # dinamik fazlasiyla yeterlidir. `None` ise cagiran taraf L2 penceresini
    # yeniden cozmek zorundadir.
    samples: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(self.values.size)

    def seconds(self, frames: float) -> float:
        """Cerceve sayisini saniyeye cevirir."""
        return frames / self.hop_hz


@dataclass(frozen=True)
class CoarseMatch:
    """Zarf duzeyinde bulunan kaba hizalama."""

    # test'in reference'e gore gecikmesi, saniye. Pozitif = test geride.
    lag_s: float
    lag_frames: int
    # Bulunan gecikmede normalize edilmis zarf korelasyonu (-1..1). Spektral
    # farklara dayanikli bir "ayni kayit mi" gostergesi.
    rho: float
    # Bu gecikmenin ikinci en iyi ALAKASIZ tepeye gore ustunlugu yok: zarfta
    # boyle bir olcut ise yaramiyor (bkz. docs/DECISIONS.md). Ayirt edici olan
    # tek sey rho'dur.
    # Bu gecikmede ust uste binen cerceve sayisi.
    overlap_frames: int


def _log_energy(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Cerceve basina dB cinsinden enerji.

    Taban en yuksek cerceveye GORE belirlenir; boylece zarf kaydin mutlak
    seviyesinden bagimsizdir ve sessizlik -inf yerine sonlu bir degere oturur.
    """
    if x.size < window:
        return np.zeros(0, dtype=np.float64)
    count = 1 + (x.size - window) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, window)[::hop][:count]
    energy = np.mean(np.square(frames, dtype=np.float64), axis=1)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0.0:
        return np.zeros(count, dtype=np.float64)
    floor = peak * 10.0 ** (-_FLOOR_DB / 10.0)
    result: np.ndarray = 10.0 * np.log10(np.maximum(energy, floor))
    return result


def _zscore(values: np.ndarray) -> np.ndarray:
    """Ortalamayi ve olcegi kaldirir.

    Seviye farkini yok etmek icin sart: 6 dB daha sessiz bir kopya, z-skoru
    olmadan korelasyonu dusururdu.
    """
    if values.size == 0:
        return values
    centred = values - values.mean()
    scale = float(np.sqrt(np.mean(np.square(centred))))
    if scale <= 0.0:
        return np.zeros_like(centred)
    return centred / scale


def from_samples(
    x: np.ndarray,
    sample_rate: int,
    *,
    hop_ms: float = DEFAULT_HOP_MS,
    window_ms: float = DEFAULT_WINDOW_MS,
    keep_samples: bool = True,
) -> Envelope:
    """Mono ornek dizisinden zarf uretir. Saf fonksiyon, ffmpeg gerektirmez."""
    ensure_signal(x, "samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate pozitif olmali")
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    window = max(hop, round(sample_rate * window_ms / 1000.0))

    values = _zscore(_log_energy(x.astype(np.float64, copy=False), window, hop))
    samples = None
    if keep_samples and x.size:
        samples = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    return Envelope(
        values=values.astype(np.float32),
        hop_hz=sample_rate / hop,
        sample_rate=sample_rate,
        duration_s=x.size / sample_rate,
        samples=samples,
    )


def build(
    ffmpeg: Path,
    path: Path,
    *,
    stream_index: int = 0,
    rate: int = DEFAULT_RATE,
    hop_ms: float = DEFAULT_HOP_MS,
    window_ms: float = DEFAULT_WINDOW_MS,
    keep_samples: bool = True,
    cancel: CancelToken | None = None,
) -> Envelope:
    """Bir dosyayi 8 kHz mono cozup zarfini cikarir.

    Video izi HIC cozulmez (`pcm_args` zorunlu `-vn -sn -dn` uygular), yani
    20 GB'lik bir konteynerde maliyet ses izininkiyle aynidir.
    """
    blocks: list[np.ndarray] = []
    with open_pcm(
        ffmpeg,
        path,
        sample_rate=rate,
        channels=1,
        stream_index=stream_index,
        rate=rate,
        cancel=cancel,
    ) as stream:
        for block in stream.blocks(rate):
            blocks.append(block.reshape(-1).copy())
    if not blocks:
        return Envelope(
            values=np.zeros(0, dtype=np.float32),
            hop_hz=rate / max(1, int(rate * hop_ms / 1000)),
            sample_rate=rate,
            duration_s=0.0,
            samples=None,
        )
    return from_samples(
        np.concatenate(blocks),
        rate,
        hop_ms=hop_ms,
        window_ms=window_ms,
        keep_samples=keep_samples,
    )


def _prefix(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kumulatif toplam ve kare toplami (basa bir sifir eklenmis)."""
    return (
        np.concatenate(([0.0], np.cumsum(x))),
        np.concatenate(([0.0], np.cumsum(np.square(x)))),
    )


def _normalised_correlation(
    a: np.ndarray, b: np.ndarray, lags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Her gecikmede, ORTUSEN bolge uzerinden gercek Pearson korelasyonu.

    Neden bu sekilde: capraz korelasyonu sadece ortusme SAYISINA bolmek yanlis.
    Az ortusen gecikmelerde bolen kucuk oldugu icin gurultu boyutlandirilarak
    buyur ve gercek tepeyi gecebilir. Olculdu: 30 s'lik kayitta 4 s'lik kesit
    arandiginda sayiya bolme 40 denemenin 20'sinde YANLIS gecikme buldu
    (hatalar 3 s ile 32 s arasinda).

    Pearson'da pay ve payda birlikte olceklenir, sonuc +-1 ile sinirlidir ve
    dogrudan "bu ayni kayit mi" gostergesidir. Ortalama da ortusen bolgeden
    hesaplanir; zarfin kuresel z-skoru kismi ortusmede dogru merkezi vermez.
    """
    n, m = a.size, b.size
    size = 1 << int(np.ceil(np.log2(n + m)))
    cross = np.fft.irfft(np.conj(np.fft.rfft(a, size)) * np.fft.rfft(b, size), size)

    # Gecikme k icin a uzerindeki ortusme araligi [lo, hi)
    lo = np.maximum(0, -lags)
    hi = np.minimum(n, m - lags)
    counts = np.maximum(hi - lo, 0)
    safe = counts > 0
    lo_s = np.where(safe, lo, 0)
    hi_s = np.where(safe, hi, 0)
    shift = np.where(safe, lags, 0)

    sum_a, sq_a = _prefix(a)
    sum_b, sq_b = _prefix(b)
    total_a = sum_a[hi_s] - sum_a[lo_s]
    total_aa = sq_a[hi_s] - sq_a[lo_s]
    total_b = sum_b[hi_s + shift] - sum_b[lo_s + shift]
    total_bb = sq_b[hi_s + shift] - sq_b[lo_s + shift]

    count = np.where(safe, counts, 1).astype(np.float64)
    var_a = total_aa - np.square(total_a) / count
    var_b = total_bb - np.square(total_b) / count
    covariance = cross - total_a * total_b / count
    denominator = np.sqrt(np.maximum(var_a, 0.0) * np.maximum(var_b, 0.0))

    rho = np.zeros(lags.size, dtype=np.float64)
    np.divide(covariance, denominator, out=rho, where=safe & (denominator > 0.0))
    return np.clip(rho, -1.0, 1.0), counts


def coarse_match(
    reference: Envelope,
    test: Envelope,
    *,
    max_shift_s: float | None = None,
) -> CoarseMatch:
    """Iki zarfi karsilastirip kaba gecikmeyi bulur.

    Her gecikmede ortusen bolgenin gercek Pearson korelasyonu hesaplanir ve en
    buyuk olani secilir. Mutlak deger ALINMAZ: dalga formunun aksine bir enerji
    zarfinin polaritesi yoktur, yani negatif korelasyon "ters cevrilmis" degil
    "eslesmiyor" demektir. Mutlak deger alinmasi olculdu ve yanlis gecikmelerin
    kazanmasina yol aciyordu.

    Isaret uzlasimi `gccphat` ile ayni: pozitif gecikme, test'in GERIDE oldugu
    anlamina gelir.
    """
    a, b = reference.values, test.values
    if a.size == 0 or b.size == 0:
        return CoarseMatch(lag_s=0.0, lag_frames=0, rho=0.0, overlap_frames=0)
    if reference.hop_hz != test.hop_hz:
        raise ValueError("zarflarin cerceve hizi ayni olmali")

    size = 1 << int(np.ceil(np.log2(a.size + b.size)))
    index = np.arange(size)
    lags = np.where(index > size // 2, index - size, index)

    rho, counts = _normalised_correlation(a.astype(np.float64), b.astype(np.float64), lags)

    minimum = max(_MIN_OVERLAP_FRAMES, int(_MIN_OVERLAP_FRACTION * min(a.size, b.size)))
    valid = counts >= minimum
    if not valid.any():
        # Ortusme her yerde yetersiz: dosyalardan biri digerinin yaninda cok kisa.
        valid = counts >= max(1, int(counts.max()))
    if max_shift_s is not None:
        valid &= np.abs(lags) <= round(max_shift_s * reference.hop_hz)
        if not valid.any():
            return CoarseMatch(lag_s=0.0, lag_frames=0, rho=0.0, overlap_frames=0)

    scored = np.where(valid, rho, -np.inf)
    peak_index = int(np.argmax(scored))
    lag_frames = int(lags[peak_index])

    return CoarseMatch(
        lag_s=lag_frames / reference.hop_hz,
        lag_frames=lag_frames,
        rho=float(rho[peak_index]),
        overlap_frames=int(counts[peak_index]),
    )
