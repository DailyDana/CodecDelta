"""Maskeleme esigi -- referansin her cercevede ne kadar gurultuyu gizledigi.

Model, MPEG-1 psikoakustik model 1 / Johnston (1988) bicimindedir:

1. Referans cercevesinin guc spektrumu isitsel bantlara toplanir.
2. Her bandin enerjisi komsu bantlara Schroeder (1979) yayilma fonksiyonuyla
   yayilir: yukari dogru ~10 dB/Bark, asagi dogru ~25 dB/Bark. Bir bas
   davulu ustundeki bantlari, ustteki bir zil alttakileri az maskeler.
3. Yayilmis enerji, tonaliteye bagli bir "maskeleme payi" kadar dusurulur.
   Gurultu gurultuyu iyi maskeler (~5.5 dB), tonal bir sinyal gurultuyu kotu
   maskeler (14.5 dB + Bark). Tonalite, cercevenin spektral duzlugunden (SFM)
   kestirilir.
4. Sonuc, mutlak isitme esiginin (Terhardt 1979) altina inemez.

Sadelestirme ve bilinen sinirlar (planin "acik riskler" listesinde):

- Tonalite BANT BASINA, MPEG-1'in tonal bilesen kuraliyla (`band_tonality`).
  Uc yontem ayni gercek FLAC/Opus ciftinde (~141 kbps, dinlemede seffafa
  yakin) olculdu; MPEG toplam NMR medyani: kuresel SFM +5.8 dB, bant basina
  SFM -9.2 dB, MPEG tepe kurali +0.8 dB. Ayni dosya, ayni gurultu, 15 dB
  yayilim -- mutlak NMR'a esik baglanmamasinin olculmus gerekcesi. Kuresel
  SFM reddedildi (Johnston'in 14.5 + z payi 8 kHz'de 35 dB ve tum spektruma
  yayiliyor), bant SFM reddedildi (bant genisligine bagli; saf ton 20 binde
  "yari tonal"). Tepe kurali literaturde tanimli ve genislikten bagimsiz.
- Yayilma katsayilari ve ofsetler literaturden ama bu dogrulama kullanimi icin
  KALIBRE EDILMEDI. Hata yonu bilinmiyor. Capa merdiveni ayni modeli iki
  tarafa da uyguladigi icin goreli kullanimda buyuk olcude sadelesir.
- Mutlak seviye varsayimi: tam olcekli sinus = `FULL_SCALE_SPL_DB` dB SPL.
  Yalnizca ATH'yi etkiler. Dinleme seviyesi bilinmedigi icin bir varsayim
  kacinilmaz; MPEG'in geleneksel degeri kullaniliyor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.psycho.erb import BandLayout

# Tam olcekli (0 dBFS) bir sinusun varsayilan ses basinc duzeyi.
FULL_SCALE_SPL_DB = 96.0

# Johnston'in maskeleme payi: tonal icin 14.5 + z, gurultu icin 5.5 dB.
_TONAL_OFFSET_DB = 14.5
_NOISE_OFFSET_DB = 5.5
# SFM'nin tamamen tonal sayildigi deger (dB). Johnston: -60.
_SFM_TONAL_DB = -60.0

# Esik icin taban: tamamen sessiz bir cercevede bile ATH gecerli.
_TINY = 1e-30


def spreading_db(delta_bark: np.ndarray) -> np.ndarray:
    """Schroeder yayilma fonksiyonu, dB. `delta_bark` = maskelenen - maskeleyen."""
    x = delta_bark + 0.474
    result: np.ndarray = 15.81 + 7.5 * x - 17.5 * np.sqrt(1.0 + x * x)
    return result


def absolute_threshold_db_spl(frequency_hz: np.ndarray) -> np.ndarray:
    """Terhardt mutlak isitme esigi, dB SPL."""
    f = np.maximum(np.asarray(frequency_hz, dtype=np.float64), 20.0) / 1000.0
    result: np.ndarray = 3.64 * f**-0.8 - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f**4
    return result


@dataclass(frozen=True)
class MaskingModel:
    """Bir bant duzeni icin onceden hesaplanmis yayilma matrisi ve ATH."""

    layout: BandLayout
    # spread[i, j]: j bandindaki enerjinin i bandina yayilan orani (dogrusal),
    # satirlar toplam kazanca gore normalize (duz spektrumda esik = enerji - pay).
    spread: np.ndarray
    # Bant basina ATH, tam olcekli sinus = 1.0 olan guc olceginde.
    ath: np.ndarray
    # Johnston'in tonal payindaki Bark terimi, bant basina.
    tonal_offset_db: np.ndarray


def build(layout: BandLayout, *, full_scale_spl_db: float = FULL_SCALE_SPL_DB) -> MaskingModel:
    z = layout.centre_bark
    matrix = 10.0 ** (spreading_db(z[:, None] - z[None, :]) / 10.0)
    matrix /= matrix.sum(axis=1, keepdims=True)

    # Bandin en hassas noktasi: bant icindeki en dusuk ATH.
    bin_hz = layout.sample_rate / layout.fft_size
    ath = np.empty(layout.count)
    for i, (lo, hi) in enumerate(layout.edges):
        freqs = np.arange(lo, hi) * bin_hz
        ath[i] = float(absolute_threshold_db_spl(freqs).min())
    ath_power = 10.0 ** ((ath - full_scale_spl_db) / 10.0)

    return MaskingModel(
        layout=layout,
        spread=matrix,
        ath=ath_power,
        tonal_offset_db=_TONAL_OFFSET_DB + z,
    )


def _sfm_tonality(log_power: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Son eksen uzerinden SFM -> tonalite 0..1."""
    geometric = np.exp(np.mean(log_power, axis=-1))
    arithmetic = np.mean(power, axis=-1)
    sfm_db = 10.0 * np.log10(geometric / arithmetic)
    result: np.ndarray = np.clip(sfm_db / _SFM_TONAL_DB, 0.0, 1.0)
    return result


def tonality(power: np.ndarray) -> np.ndarray:
    """Cerceve basina KURESEL tonalite 0..1, spektral duzlukten (SFM).

    SFM = geometrik ortalama / aritmetik ortalama (dB). 0 dB duz gurultu,
    -60 dB ve alti tamamen tonal sayilir (Johnston). Ustel dagilimli gurultu
    binlerinde SFM ~ -2.5 dB, yani alpha ~ 0.04.
    """
    safe = np.maximum(power, _TINY)
    return _sfm_tonality(np.log(safe), safe)


# MPEG-1 model 1 tonal bilesen kurali: yerel tepe, komsularini bu kadar asmali.
_TONAL_PEAK_MARGIN = 10.0 ** (7.0 / 10.0)
# Hann ana lobu +-2 bin; karsilastirma onun DISINDAKI komsularla yapilir ve
# tepenin enerjisi ana lob boyunca toplanir.
_MAINLOBE = 2


def band_tonality(layout: BandLayout, power: np.ndarray) -> np.ndarray:
    """(cerceve, bant) tonalite: bandin enerjisinin ne kadari tonal tepelerde.

    Bant basina SFM denendi ve REDDEDILDI: 20 binlik bir bantta saf bir tonun
    SFM'i Hann sizintisi yuzunden ~-34 dB'de takiliyor (Johnston'in -60 dB
    hedefi binlerce binlik tam spektrum icin), yani saf ton "yari tonal"
    sayiliyordu (alpha 0.57). Olcut bant genisligine bagliydi.

    Bunun yerine MPEG-1 model 1'in tonal bilesen kurali: bir bin, ana lob
    disindaki komsularini (+-3, +-4) 7 dB asan bir yerel tepeyse tonaldir ve
    ana lobundaki enerji tonal sayilir. Bant tonalitesi = tonal enerji / bant
    enerjisi. Saf ton ~1, ustel dagilimli gurultu binlerinde tesadufi tepeler
    kucuk bir oran verir. Bant genisliginden bagimsiz.
    """
    bins = power.shape[1]
    safe = np.maximum(power, _TINY)
    peak = np.zeros_like(safe, dtype=bool)
    inner = slice(_MAINLOBE + 2, bins - _MAINLOBE - 2)
    centre = safe[:, inner]
    condition = (centre > safe[:, inner.start - 1 : inner.stop - 1]) & (
        centre >= safe[:, inner.start + 1 : inner.stop + 1]
    )
    for offset in (_MAINLOBE + 1, _MAINLOBE + 2):
        condition &= (
            centre > _TONAL_PEAK_MARGIN * safe[:, inner.start - offset : inner.stop - offset]
        )
        condition &= (
            centre > _TONAL_PEAK_MARGIN * safe[:, inner.start + offset : inner.stop + offset]
        )
    peak[:, inner] = condition
    # Tepenin ana lobu (+-MAINLOBE bin) tonal enerjiye girer.
    tonal_mask = peak.copy()
    for offset in range(1, _MAINLOBE + 1):
        tonal_mask[:, offset:] |= peak[:, :-offset]
        tonal_mask[:, :-offset] |= peak[:, offset:]
    tonal_energy = layout.energies(np.where(tonal_mask, safe, 0.0))
    total = layout.energies(safe)
    result: np.ndarray = np.clip(tonal_energy / np.maximum(total, _TINY), 0.0, 1.0)
    return result


def threshold(model: MaskingModel, power: np.ndarray) -> np.ndarray:
    """(cerceve, bin) referans guc spektrumundan (cerceve, bant) maskeleme esigi.

    `power`, tam olcekli sinusun bant enerjisi 1.0 olacak sekilde olceklenmis
    olmali (bkz. `nmr.power_scale`).
    """
    energies = model.layout.energies(power)
    spread = energies @ model.spread.T
    alpha = band_tonality(model.layout, power)
    offset_db = alpha * model.tonal_offset_db[None, :] + (1.0 - alpha) * _NOISE_OFFSET_DB
    masked = spread * 10.0 ** (-offset_db / 10.0)
    result: np.ndarray = np.maximum(masked, model.ath[None, :])
    return result
