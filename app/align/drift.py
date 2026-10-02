"""Hiz orani ve surukelenme tespiti -- hizalamanin ucuncu sorusu.

Iki kayit ayni anda baslasa bile ayni HIZDA gitmeyebilir. Uc ayri sebep:

1. **PAL hizlandirmasi (%4.17).** Sinema filmi 24 fps'tir, PAL televizyon
   25 fps. Aktarim sirasinda film cogu zaman yeniden zamanlanmaz, sadece daha
   hizli oynatilir; ses de onunla birlikte %4.17 hizlanir ve perde bir yarim
   tona yakin yukselir. Bir filmin PAL DVD'sinden alinan ses ile ayni filmin
   Blu-ray'inden alinan ses arasindaki fark budur, codec farki degil.
2. **Saat surukelenmesi.** Farkli donanimla yapilmis iki kayit, nominal olarak
   ayni orneklemede calissa bile kristal toleransi kadar (tipik olarak birkac
   10 ppm) kayar. 2 saatlik bir kayitta 50 ppm, 360 ms eder.
3. **Kasitli zamanlama.** 30/29.97 (NTSC pulldown) ve 1.0001 gibi oranlar.

Neden onemli: oran 1'den farkliysa fark ARTIK codec farki degildir. Olculen
S/N'nin buyuk kismi zaman kaymasindan gelir ve rapor bunu soylemeden sayi
gostermemelidir.

Yontem: kayit boyunca dagitilmis capalarda tam sayi gecikme olculur, sonra
gecikme-zaman dogrusuna **Theil-Sen** ile egim gecirilir. En kucuk kareler
degil: capalarin bir kismi sessizlige, alkisa ya da bir dropout'a denk gelir ve
tamamen yanlis gecikme dondurur. Theil-Sen ikili egimlerin medyanini aldigi
icin capalarin %29'u bozuk olsa bile dogru kalir; tek bir uc deger en kucuk
kareler egimini tamamen cevirir.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.align import envelope, gccphat
from app.align.thresholds import MIN_ENVELOPE_CORRELATION
from app.core.messages import Message
from app.dsp.transforms import ensure_signal

# Bilinen hiz oranlari ve bunlara yakalanma toleransi. Olculen oran bir tablo
# degerine bu kadar yakinsa, olculen degil TABLO degeri kullanilir: bunlar
# tam rasyonel sayilardir ve olcum gurultusu yuzunden 1.041663 raporlamak
# yaniltici olur.
_SNAP_TABLE: tuple[tuple[float, str], ...] = (
    (25.0 / 24.0, Message("drift.pal_up", "PAL speed-up (24->25 fps)")),
    (24.0 / 25.0, Message("drift.pal_down", "PAL slow-down (25->24 fps)")),
    (30.0 / 29.97, Message("drift.ntsc_up", "NTSC pulldown (29.97->30 fps)")),
    (29.97 / 30.0, Message("drift.ntsc_down", "NTSC pulldown (30->29.97 fps)")),
    (1.0001, Message("drift.1_0001", "1.0001 timing")),
)
# Tolerans MUTLAK degil, sapmanin oraniyla olcekli. Sabit 2e-4 olarak
# denendi ve yanlis cikti: 1.0001 girdisinin kendi sapmasi 1e-4 oldugu icin
# boyle bir pencere SURUKLENME OLMAYAN bir dosyayi (oran 1.0) "1.0001
# zamanlama" diye etiketliyordu. Oransal tolerans her girdiyi kendi
# olceginde degerlendirir.
_SNAP_RELATIVE = 0.02
_SNAP_FLOOR = 1e-5

# Bu esigin altindaki oranlar "yok" sayilir. 5 ppm, 10 dakikalik bir parcada
# 3 ms eder -- L3 ince hizalamanin blok-yerel takibi bunu zaten yutar ve
# global bir yeniden orneklemeye deger bir sey degildir.
_NEGLIGIBLE_PPM = 5.0

# Theil-Sen kalintilarinin MAD'i bunu asarsa fit'e guvenilmez. 20 ms, capalarin
# arasinda tutarli bir dogru olmadigi anlamina gelir.
_MAX_RESIDUAL_MS = 20.0

# Anlamli bir fit icin gereken en az capa. Plan 12 diyor; ikili egim medyani
# bunun altinda tek bir bozuk capaya karsi kirilgan hale geliyor.
MIN_ANCHORS = 12

# Capalarin medyan belirsizligi (`gccphat.ambiguity`) bunu asarsa sinyal
# periyodik sayilir. Olculen: saf sinus, iki ton ~1.0; gercek muzik (5 parca)
# ve sentetik muzik <= 0.21; agir EQ'lu gurultu 0.24. Periyodik sinyalde her
# capa baska bir periyot katini secer ve Theil-Sen bunlardan +3004 ppm ya da
# "NTSC" uyduruyordu (denetim D6).
MAX_MEDIAN_AMBIGUITY = 0.5
# Capalarin medyan gecikmesi zarfinkine bu kadar yakinsa zarf onu dogrular
# (zarf cozunurlugu 10 ms).
_ENVELOPE_AGREEMENT_S = 0.02
# Periyodiklik karari icin gereken en az capa. Fit icin gerekenden (12) az:
# periyodik sinyalde capalar korelasyon esigine de takilabiliyor (Opus'lu iki
# ton: 8 capa) ve karar yine acik.
_MIN_PERIODIC_ANCHORS = 5

# Capa penceresinin varsayilan uzunlugu. Kisa olmasi ZORUNLU: pencere icinde
# biriken kayma `window * (ratio - 1)` ornektir ve icerigin periyodunun yarisini
# astiginda korelasyon coker. OLCULEN (8 kHz, 300 s, 30 capa):
#
#   sapma    0.25 s pencere        1.0 s          2.0 s
#      10 ppm  0.1 ppm hata     0.1 ppm        0.0 ppm
#     100 ppm  0.0 ppm hata     0.0 ppm        0.0 ppm
#    1000 ppm  1.0 ppm hata     capa YOK       capa YOK
#    5000 ppm  capa YOK         capa YOK       capa YOK
#
# Yani 0.25 s ile yakalama araligi ~1000 ppm. NTSC pulldown (1001 ppm) tam
# sinirda calisir; PAL (41667 ppm) hicbir pencerede calismaz ve on telafi
# gerektirir -- `estimate_from_audio` bunu hipotez sinamasiyla cozer.
DEFAULT_WINDOW_S = 0.25
MAX_ANCHOR_PPM = 1000.0

# Sinanan hipotezler: 1.0 (dogrudan olcum) ve tablodaki, capalarin OLCEMEYECEGI
# kadar buyuk oranlar. 1.0001 gibi aralik icindeki bir oran 1.0 hipotezinin
# artigindan zaten olculur; ayri bir hipotez olarak sinanirsa gurultude 1.0 ile
# neredeyse berabere skor alir ve rastgele kazanir (olculdu: 0 dB S/N'de
# 24 denemenin birinde 0.06 ppm "surukelenme" raporlandi).
_HYPOTHESES: tuple[float, ...] = (
    1.0,
    *(value for value, _ in _SNAP_TABLE if abs(value - 1.0) * 1e6 > MAX_ANCHOR_PPM),
)

DriftStatus = Literal["none", "drift", "unreliable"]


@dataclass(frozen=True)
class Anchor:
    """Kaydin tek bir noktasinda olculen gecikme."""

    # Capanin test icindeki merkez konumu, saniye.
    position_s: float
    # O noktada olculen gecikme, saniye. Pozitif = test geride.
    lag_s: float
    # Hizalanmis pencerelerin Pearson korelasyonu (ISARETLI).
    correlation: float
    psr: float
    # bkz. `gccphat.ambiguity`. Periyodik sinyalde ~1.
    ambiguity: float = 0.0


@dataclass(frozen=True)
class DriftEstimate:
    """Kayit boyunca olculen hiz orani."""

    # test'in reference'e gore hiz orani. 1'den buyuk = test daha HIZLI.
    ratio: float
    # Kaydin basindaki gecikme, saniye (Theil-Sen kesisimi).
    offset_s: float
    status: DriftStatus
    # Tabloya yakalandiysa insan-okunur adi, yoksa None.
    label: str | None
    anchors: int
    # Theil-Sen kalintilarinin medyan mutlak sapmasi, milisaniye. Fit'in ne
    # kadar tutarli oldugunun olcusu.
    residual_ms: float
    # Capalarin cogu periyodik bir sinyale dustu: gecikme belirsiz, hiz orani
    # olculemez. Durum bu durumda her zaman "unreliable".
    periodic: bool = False

    @property
    def ppm(self) -> float:
        """Orani milyonda bir cinsinden sapma olarak verir."""
        return (self.ratio - 1.0) * 1e6

    @property
    def is_resampling(self) -> bool:
        """Iki dosya FARKLI bir zamanlama aktarimindan mi geliyor?

        Dogruysa rapor ham S/N GOSTERMEMELIDIR. PAL ya da NTSC donusumu sesi
        yeniden orneklenmis (ve PAL'de perdesi kaymis) baska bir master yapar;
        aradaki fark codec hakkinda bir sey soylemez.

        Kucuk, etiketsiz saat kaymasi (birkac 10 ppm) bu sinifa GIRMEZ: icerik
        ayni, yalnizca iki saat farkli hizda. Plan geregi global olarak yeniden
        orneklenmez, blok-yerel gecikme takibiyle telafi edilir -- bkz.
        `needs_tracking`.
        """
        if self.status != "drift":
            return False
        return self.label is not None or abs(self.ppm) > MAX_ANCHOR_PPM

    @property
    def needs_tracking(self) -> bool:
        """Karsilastirma mumkun ama gecikme kayit boyunca izlenmeli mi?"""
        return self.status == "drift" and not self.is_resampling and abs(self.ppm) > _NEGLIGIBLE_PPM


class _Lazy:
    """Bir ornek dizisini pencere pencere float64 olarak veren gorunum.

    Tam boy kopya hic yapilmaz: int16 zarf ornekleri yalnizca okunan pencerede
    float64'e cevrilir, `ratio` verilirse hiz telafisi (dogrusal interpolasyon)
    da yalnizca o pencerede hesaplanir. Once ornekler tumden float64'e
    cevriliyor ve her PAL hipotezi icin test yeniden orneklenmis tam bir kopya
    oluyordu: 60 dakikalik ciftte 1.9 GB (denetim D5).

    Dogrusal interpolasyon yeterli: amac hangi HIPOTEZIN daha iyi hizalandigini
    secmek; interpolasyon hatasi tum adaylara ayni bulasir ve sadelesir. Nihai
    telafi boru hattinda soxr ile yapilir.
    """

    def __init__(self, data: np.ndarray, ratio: float = 1.0) -> None:
        if isinstance(data, _Lazy):
            raise TypeError("_Lazy ic ice kullanilmaz")
        ensure_signal(data, "samples")
        self._data = data
        self._ratio = ratio
        self.size = data.size if ratio == 1.0 else max(0, int(data.size / ratio))

    def __getitem__(self, window: slice) -> np.ndarray:
        start, stop, _ = window.indices(self.size)
        if stop <= start:
            return np.zeros(0, dtype=np.float64)
        if self._ratio == 1.0:
            return self._data[start:stop].astype(np.float64)
        positions = np.arange(start, stop) * self._ratio
        lo = int(positions[0])
        hi = min(self._data.size, int(np.ceil(positions[-1])) + 2)
        source = self._data[lo:hi].astype(np.float64)
        result: np.ndarray = np.interp(positions, np.arange(lo, hi), source)
        return result

    def chunks(self, size: int) -> Iterator[np.ndarray]:
        for start in range(0, self.size, size):
            yield self[start : start + size]


def _lazy(x: np.ndarray | _Lazy) -> _Lazy:
    return x if isinstance(x, _Lazy) else _Lazy(x)


def collect_anchors(
    reference: np.ndarray | _Lazy,
    test: np.ndarray | _Lazy,
    sample_rate: int,
    *,
    coarse_lag_s: float = 0.0,
    count: int = 24,
    window_s: float = DEFAULT_WINDOW_S,
    max_drift: float = 2.0 * MAX_ANCHOR_PPM * 1e-6,
    min_correlation: float = 0.5,
) -> list[Anchor]:
    """Kayit boyunca dagitilmis noktalarda gecikme olcer.

    Arama yaricapi konumla birlikte BUYUR: `max_drift` orani kadar, cunku
    biriken kayma konumla dogru orantilidir. Varsayilan, capalarin yakalama
    araliginin iki kati (2000 ppm). Daha genis aramak ise yaramaz -- o
    araligin disinda pencere ici kayma tepeyi zaten dagitiyor -- ama maliyeti
    buyutur: %5 ile 2 saatlik bir kaydin sonunda yaricap 360 s olurdu, yani
    her capa icin milyonlarca ornekli bir FFT. PAL gibi buyuk oranlar olculmez,
    `estimate_from_audio` icinde sinanir.

    Korelasyonu `min_correlation`in altinda kalan capalar ATILIR. Sessizlige
    veya alkisa denk gelen bir pencere tamamen anlamsiz bir gecikme dondurur;
    Theil-Sen bunlara dayanikli olsa da, once elenmeleri fit'i belirgin
    iyilestirir.
    """
    reference, test = _lazy(reference), _lazy(test)
    if sample_rate <= 0:
        raise ValueError("sample_rate pozitif olmali")
    if count < 2:
        raise ValueError("en az iki capa gerekir")

    window = int(window_s * sample_rate)
    if window <= 0 or test.size < window or reference.size < window:
        return []

    anchors: list[Anchor] = []
    # Capalar test'in kullanilabilir araligina esit araliklarla yayilir; bas ve
    # son kirpilir cunku oralarda referans tarafi pencereyi tasiyamayabilir.
    starts = np.linspace(0, test.size - window, count).astype(int)
    for start in starts:
        position_s = (start + window / 2) / sample_rate
        radius_s = window_s + max_drift * position_s
        # Referans tarafindan, kaba gecikmeye gore beklenen konumun etrafinda
        # yaricap kadar genis bir parca al.
        centre = start - int(coarse_lag_s * sample_rate)
        radius = int(radius_s * sample_rate)
        lo = max(0, centre - radius)
        hi = min(reference.size, centre + window + radius)
        if hi - lo < window:
            continue

        window_test = test[start : start + window]
        estimate = gccphat.estimate(reference[lo:hi], window_test)
        # gccphat gecikmeyi kirpilmis parcalar icinde verir; mutlak zamana cevir.
        # test[start+t] = reference[lo+t-L] ve test[T] = reference[T-d] oldugundan
        # start - d = lo - L, yani d = L + start - lo.
        lag_samples = estimate.lag + start - lo

        # DIKKAT: `estimate.correlation` burada KULLANILAMAZ. `gccphat` esit
        # uzunlukta girdi varsayar; referans dilimi test penceresinden uzun
        # oldugunda `aligned_slices` kisa olanin boyuna gore kirpar ve gercek
        # gecikme o boyu astiginda sessizce BOS dilim dondurur -- korelasyon
        # 0.0 olur, tepe dogru bulunmus olsa bile. Yerine dogru konumlanmis
        # referans parcasi dogrudan eslestiriliyor.
        offset = -estimate.lag
        if offset < 0 or offset + window > hi - lo:
            continue
        correlation = gccphat.correlation_at(
            reference[lo + offset : lo + offset + window], window_test, 0
        )
        if abs(correlation) < min_correlation:
            continue
        anchors.append(
            Anchor(
                position_s=position_s,
                lag_s=lag_samples / sample_rate,
                correlation=correlation,
                psr=estimate.psr,
                ambiguity=gccphat.ambiguity(reference[lo:hi], window_test),
            )
        )
    return anchors


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ikili egimlerin medyani ile dogru gecirir.

    Kirilma noktasi %29: capalarin bu kadari tamamen yanlis olsa bile egim
    dogru kalir. En kucuk karelerde tek bir uc deger yeterlidir.

    Kesisim, `median(y - slope * x)` ile bulunur (Siegel'in onerdigi bicim).
    """
    if x.size != y.size:
        raise ValueError("x ve y ayni uzunlukta olmali")
    if x.size < 2:
        return 0.0, float(y[0]) if y.size else 0.0

    rows, cols = np.triu_indices(x.size, k=1)
    run = x[cols] - x[rows]
    usable = run != 0.0
    if not usable.any():
        return 0.0, float(np.median(y))
    slope = float(np.median((y[cols] - y[rows])[usable] / run[usable]))
    return slope, float(np.median(y - slope * x))


def snap(ratio: float) -> tuple[float, str] | None:
    """Oran bilinen bir zamanlama oranina yakinsa onu dondurur."""
    for known, label in _SNAP_TABLE:
        tolerance = max(_SNAP_FLOOR, abs(known - 1.0) * _SNAP_RELATIVE)
        if abs(ratio - known) <= tolerance:
            return known, label
    return None


def estimate(
    anchors: list[Anchor], *, min_anchors: int = MIN_ANCHORS, check_periodic: bool = True
) -> DriftEstimate:
    """Capalardan hiz oranini cikarir.

    Gecikme, test icindeki konumun dogrusal bir fonksiyonu kabul edilir:

        lag(t) = offset + slope * t

    test, reference'a gore `r` kati hizliysa test'in `t` anindaki icerigi
    reference'in `r*t` anindadir, yani `t - lag = r*t` ve `slope = 1 - r`.
    Dolayisiyla `ratio = 1 - slope`. (Bu turetme sentetik bir testte bilinen
    bir oran enjekte edilerek dogrulaniyor -- isaret hatasi yapmak kolay.)
    """
    if len(anchors) < min_anchors:
        return DriftEstimate(
            ratio=1.0,
            offset_s=float(np.median([a.lag_s for a in anchors])) if anchors else 0.0,
            status="unreliable",
            label=None,
            anchors=len(anchors),
            residual_ms=float("inf"),
        )

    if check_periodic and _is_periodic(anchors):
        return _periodic(anchors)
    x = np.array([a.position_s for a in anchors], dtype=np.float64)
    y = np.array([a.lag_s for a in anchors], dtype=np.float64)
    slope, offset = theil_sen(x, y)
    residual = y - (offset + slope * x)
    residual_ms = float(np.median(np.abs(residual - np.median(residual)))) * 1000.0

    ratio = 1.0 - slope
    if residual_ms > _MAX_RESIDUAL_MS:
        return DriftEstimate(
            ratio=ratio,
            offset_s=offset,
            status="unreliable",
            label=None,
            anchors=len(anchors),
            residual_ms=residual_ms,
        )

    return _classify(ratio, offset, len(anchors), residual_ms)


def _envelope_confirms(match: envelope.CoarseMatch | None, anchors: list[Anchor]) -> bool:
    """Bilgili bir zarf capalarin gecikmesini dogruluyor mu?

    Periyodik sinyalde zarf duzdur (sabit ton) ya da bilgisizdir; EQ ve
    sikistirma uygulanmis bir kopyada (farkli master) ise faz bozulmasi
    belirsizlik oranini yukseltir ama zarf guclu eslesir ve capalar ayni
    gecikmede toplanir. Kalibrasyonda iki boyle cift "sabit ton" diye
    isaretleniyordu.
    """
    if match is None or match.rho < MIN_ENVELOPE_CORRELATION or not anchors:
        return False
    spread = float(np.median([abs(a.lag_s - match.lag_s) for a in anchors]))
    return spread <= _ENVELOPE_AGREEMENT_S


def _is_periodic(anchors: list[Anchor]) -> bool:
    return len(anchors) >= _MIN_PERIODIC_ANCHORS and (
        float(np.median([a.ambiguity for a in anchors])) > MAX_MEDIAN_AMBIGUITY
    )


def _periodic(anchors: list[Anchor]) -> DriftEstimate:
    return DriftEstimate(
        ratio=1.0,
        offset_s=float(np.median([a.lag_s for a in anchors])),
        status="unreliable",
        label=None,
        anchors=len(anchors),
        residual_ms=float("inf"),
        periodic=True,
    )


def _classify(ratio: float, offset_s: float, anchors: int, residual_ms: float) -> DriftEstimate:
    """Guvenilir bir orani tabloya yakalar ya da ihmal edilebilir sayar.

    TEK yer: hem dogrudan olcum hem hipotez yolu buradan gecer. Hipotez yolu
    once kendi siniflandirmasini yapiyordu ve "ihmal edilebilir" kuralini
    atlayip 0.06 ppm'i "surukelenme" diye raporladi.
    """
    snapped = snap(ratio)
    if snapped is not None:
        known, label = snapped
        return DriftEstimate(known, offset_s, "drift", label, anchors, residual_ms)
    if abs(ratio - 1.0) * 1e6 <= _NEGLIGIBLE_PPM:
        return DriftEstimate(1.0, offset_s, "none", None, anchors, residual_ms)
    return DriftEstimate(ratio, offset_s, "drift", None, anchors, residual_ms)


def estimate_from_audio(
    reference: np.ndarray,
    test: np.ndarray,
    sample_rate: int,
    *,
    coarse_lag_s: float | None = None,
    count: int = 30,
    window_s: float = DEFAULT_WINDOW_S,
    min_correlation: float = 0.3,
) -> DriftEstimate:
    """Ham sesten hiz oranini cikarir; PAL'i hipotez sinamasiyla cozer.

    Capalar tek baslarina yaklasik 1000 ppm'e kadar dayanir (bkz.
    `DEFAULT_WINDOW_S`), PAL ise 41667 ppm'dir -- dogrudan olculemez. Ama PAL
    SUREKLI bir bilinmeyen degildir: bilinen, ayrik ve kisa bir tablodan gelir.
    O yuzden olculmez, SINANIR. Her aday oran icin test on telafi edilir ve
    capalar yeniden toplanir; dogru hipotezde kayma ortadan kalkacagi icin hem
    capa sayisi hem korelasyon belirgin yukselir.

    Kazanan hipotezin uzerinde kalan artik oran normal yoldan olculur ve nihai
    oran ikisinin carpimidir.

    `coarse_lag_s` verilmezse her hipotez KENDI kaba gecikmesini telafi
    edilmis zarftan bulur. Tek bir disaridan verilen gecikme PAL'de tanimsizdir:
    gecikme kayit boyunca %4 degisir, zarf eslestirmesi ortalama bir yer secer
    ve 60 s'lik bir dosyada bile kaydin basinda 1 s'den fazla sapar.
    """
    reference_samples, test_samples = _Lazy(reference), test
    reference_envelope = (
        envelope.from_chunks(reference_samples.chunks(sample_rate), sample_rate, keep_samples=False)
        if coarse_lag_s is None
        else None
    )
    best: tuple[float, float, DriftEstimate] | None = None
    match: envelope.CoarseMatch | None = None
    for candidate in _HYPOTHESES:
        compensated = _Lazy(test_samples, 1.0 / candidate)
        lag_s = coarse_lag_s
        if reference_envelope is not None:
            match = envelope.coarse_match(
                reference_envelope,
                envelope.from_chunks(
                    compensated.chunks(sample_rate), sample_rate, keep_samples=False
                ),
            )
            # Zarf bilgisizse (dinamigi duz icerik) gecikmesine guvenilmez ve
            # sifira dusulur -- ama hipotez ELENMEZ. Ilk surum burada eliyordu
            # ve duragan icerikte DOGRU hipotezi de atip "bilmiyorum" diyordu
            # (50 ppm ve NTSC, capalar tek basina kusursuz calisirken). Karari
            # capalar verir; zarf yalnizca nereye bakilacagini soyler.
            lag_s = match.lag_s if match.rho >= MIN_ENVELOPE_CORRELATION else 0.0
        anchors = collect_anchors(
            reference_samples,
            compensated,
            sample_rate,
            coarse_lag_s=lag_s if lag_s is not None else 0.0,
            count=count,
            window_s=window_s,
            min_correlation=min_correlation,
        )
        if candidate == 1.0 and len(anchors) < MIN_ANCHORS and lag_s:
            # Zarf yaniltabilir: zarfi periyodik bir kayitta sonu degistirilmis
            # dosyada eslestirme degisen kuyrugu disarida birakan bir periyot
            # katina (-17.27 s) kilitleniyordu (denetim D41). Ayni kaydin iki
            # kodlamasi cogunlukla ayni zaman cizgisindedir; capalar bir kez de
            # sifir gecikme etrafinda aranir, daha cok capa veren kazanir.
            at_zero = collect_anchors(
                reference_samples,
                compensated,
                sample_rate,
                coarse_lag_s=0.0,
                count=count,
                window_s=window_s,
                min_correlation=min_correlation,
            )
            if len(at_zero) > len(anchors):
                anchors = at_zero
        if candidate == 1.0 and _is_periodic(anchors) and not _envelope_confirms(match, anchors):
            # Hipotez yarisina sokulmaz: olceklenmis bir periyodik sinyal de
            # periyodiktir ama kaydirilmis frekans tepeleri esitsizlestirip
            # sahte bir dogru uretebiliyor (1 kHz sinus: +3004 ppm).
            return _periodic(anchors)
        if len(anchors) < MIN_ANCHORS:
            continue
        score = float(np.median([abs(a.correlation) for a in anchors]))
        # Periyodiklik yalnizca dogal hipotezde ve zarf kosuluyla karara baglanir
        # (yukarida); burada yeniden sorulursa PAL hipotezinin bayragi sonuca
        # sizip EQ'lu bir kopyayi "sabit ton" yapiyordu.
        residual = estimate(anchors, check_periodic=False)
        if best is None or score > best[1]:
            best = (candidate, score, residual)

    if best is None:
        return DriftEstimate(
            ratio=1.0,
            offset_s=0.0,
            status="unreliable",
            label=None,
            anchors=0,
            residual_ms=float("inf"),
        )

    candidate, _, residual = best
    if candidate == 1.0 or residual.status == "unreliable":
        # Hipotezin artigi tutarsizsa hipotezin kendisi de kanitlanmis degil;
        # "PAL" demek icin capalarin telafi sonrasi bir dogru olusturmasi sart.
        return residual if candidate == 1.0 else _unreliable(residual)
    return _classify(
        candidate * residual.ratio, residual.offset_s, residual.anchors, residual.residual_ms
    )


def _unreliable(residual: DriftEstimate) -> DriftEstimate:
    return DriftEstimate(
        1.0,
        residual.offset_s,
        "unreliable",
        None,
        residual.anchors,
        residual.residual_ms,
        periodic=residual.periodic,
    )
