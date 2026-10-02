"""Hizalama plani -- uc katmani tek karara baglar.

Tek soru: bu iki kayit, farklari codec farki olarak olculebilecek sekilde
hizalanabiliyor mu; evetse nasil?

    L1  zarf (100 Hz, tum kayit)      -> kaba gecikme, "ayni kayit mi" ilk ipucu
    --  surukelenme (8 kHz capalar)    -> hiz orani, PAL/NTSC, saat kaymasi
    L2  GCC-PHAT (tam hiz, pencere)    -> tam sayi ornek gecikme
    L3  artik aramasi (tam hiz)        -> kesirli gecikme + gecerlilik
    --  kanal esleme, polarite, kazanc

Bellek ilkesi: tam hizda dosyanin TAMAMI hic okunmaz. L1 ve surukelenme,
zarfin zaten tuttugu 8 kHz ornekleri kullanir; L2/L3 icin yalnizca birkac
saniyelik pencereler `WindowReader` uzerinden istenir. 20 GB'lik bir konteyner
ile 4 dakikalik bir parca ayni bellekle planlanir.

Karar ilkesi: tek bir sayiya guvenilmez. Zarf korelasyonu dusukse bu tek basina
"farkli kayit" demek DEGILDIR -- dinamigi duz bir icerikte zarf bilgi tasimaz
(olculdu, bkz. `drift.estimate_from_audio`). "Farkli kayit" hukmu ancak zarf
VE capa tabanli kanit birlikte basarisiz oldugunda verilir.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.align import drift as drift_mod
from app.align import gccphat, refine
from app.align.drift import DriftEstimate
from app.align.envelope import CoarseMatch, Envelope, coarse_match
from app.align.gccphat import LagEstimate
from app.align.refine import SubSampleEstimate
from app.align.thresholds import MIN_ENVELOPE_CORRELATION
from app.core.messages import Message
from app.dsp.transforms import db, fractional_shift, optimal_gain, to_mono

# (baslangic cercevesi, cerceve sayisi) -> (cerceve, kanal) float dizisi.
# Dosya sonunu asan istekte daha KISA dizi donebilir; negatif baslangic
# istenmez (plan bunu garanti eder).
WindowReader = Callable[[int, int], np.ndarray]

Verdict = Literal[
    "aligned",
    "different_master",
    "different_recording",
    "speed_mismatch",
    "channel_mismatch",
    "unaligned",
]

# L3 penceresi. Artik aramasinin ve kazancin istatistigi icin bol; maliyet
# pencere basina birkac FFT.
DEFAULT_WINDOW_S = 10.0

# L2'nin kaba gecikme etrafinda arayacagi yaricap. Zarf 10 ms cozunurlukte;
# gercek materyalde tepe bir-iki cerceve kayabilir. 0.25 s, bunun 25 kati.
DEFAULT_SEARCH_S = 0.25

# Surukelenme varken pencere icinde birikmesine izin verilen kayma, ornek.
# L3'un sozlesmesi kalan gecikmenin +-1 ornek icinde olmasi; 0.1 bunun onda biri.
_MAX_SMEAR_SAMPLES = 0.1
_MIN_WINDOW_FRAMES = 4096

# Ilk pencere sessizlige ya da alkisa denk gelirse denenecek diger konumlar
# (ortusmenin kesri olarak).
_POSITIONS = (0.5, 0.25, 0.75)
# Bunlardan hicbiri kullanilabilir bir pencere vermezse denenecek ek konumlar.
# Ortasi uzun sessiz bir dosyada uc pencere de sessizlige dusuyordu (D8).
_FALLBACK_POSITIONS = (0.1, 0.9, 0.4, 0.6, 0.0, 1.0)
# Zarfin bu kadarindan fazlasi sessizlik tabanindaysa pencere okunmaz bile.
_MAX_SILENT_FRACTION = 0.5
# Zarf eslesmesi bu gevsek ortusme esigiyle guclu cikiyorsa dosyalar ayni kayit
# ama az ortusuyor demektir; gerekce buna gore yazilir (hizalanmaz).
_PARTIAL_OVERLAP_FRACTION = 0.15

# Saat kaymasi izlenecekse ince gecikme bu noktalarin HEPSINDE olculur ve
# pipeline noktalardan bir dogru gecirir. Surukelenme tahmininin egimi
# capalardan gelir ve ~0.1 ppm hassasiyettedir; bu, kaydin merkezinden 300 s
# uzakta 48 kHz'de 1.4 ornek hata demek -- yuksek frekanslarda korelasyonu
# dagitmaya yeter. Tam hizda olculen noktalarin dogrusu ornek-alti hassas.
_TRACK_POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)

_CORE_GUARD = 64


@dataclass(frozen=True)
class _Point:
    """Tek bir tam hiz penceresinde olculen hizalama."""

    position_s: float
    delay: float
    lag: LagEstimate
    fine: SubSampleEstimate
    ref: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class AlignmentPlan:
    """Iki kaydin nasil hizalanacagi ve karsilastirilabilir olup olmadigi."""

    verdict: Verdict
    # test'in reference'e gore toplam gecikmesi. Pozitif = test GERIDE.
    # Surukelenme varsa bu, `position_s` anindaki gecikmedir. Bilinmiyorsa NaN.
    delay_s: float
    delay_samples: float
    # Gecikmenin olculdugu nokta (test icinde, saniye).
    position_s: float
    sample_rate: int
    envelope: CoarseMatch
    drift: DriftEstimate | None
    lag: LagEstimate | None
    fine: SubSampleEstimate | None
    # +1 ya da -1. test'in reference'e gore isareti.
    polarity: int
    # test'i reference seviyesine getirmek icin GEREKEN degil, test'in
    # reference'e gore OLCULEN seviyesi (dB). -3 = test 3 dB daha sessiz.
    gain_db: float
    # reference'in i. kanalina karsilik gelen test kanali.
    channel_map: tuple[int, ...]
    reasons: tuple[str, ...]
    # Tam hizda olculen (test konumu s, gecikme ornek) noktalari. Surukelenme
    # yoksa tek nokta; izleme gerekiyorsa `_TRACK_POSITIONS` kadar.
    track: tuple[tuple[float, float], ...] = ()

    @property
    def comparable(self) -> bool:
        """Uzerine codec farki olcumu kurulabilir mi?"""
        return self.verdict == "aligned"


def _mostly_silent(env: Envelope, start_s: float, length_s: float) -> bool:
    """Referans zarfinin bu araliginin cogu sessizlik tabaninda mi?

    Zarf degerleri z-skorlu log-enerji; dijital sessizlik ve -80 dB alti
    `envelope` tabanina kelepcelenir, yani hepsi tam olarak en kucuk degerdedir.
    Taban medyanin belirgin altinda degilse (dinamigi duz icerik) sessizlik
    yoktur.
    """
    values = env.values
    if values.size == 0:
        return False
    floor = float(values.min())
    if floor > float(np.median(values)) - 1.0:
        return False
    lo = max(0, int(start_s * env.hop_hz))
    hi = max(lo, int((start_s + length_s) * env.hop_hz))
    window = values[lo:hi]
    if window.size == 0:
        return False
    return float(np.mean(window <= floor + 1e-3)) > _MAX_SILENT_FRACTION


def _is_silent(env: Envelope) -> bool:
    """Zarf tamamen duz mu (dijital sessizlik ya da bos dosya)?"""
    if env.samples is not None:
        return not bool(np.any(env.samples))
    return env.values.size == 0 or float(np.ptp(env.values)) == 0.0


def _window_frames(sample_rate: int, window_s: float, drift: DriftEstimate | None) -> int:
    """Pencere uzunlugu; surukelenme varsa kayma sinirina gore kisaltilir."""
    frames = int(window_s * sample_rate)
    if drift is not None and drift.needs_tracking:
        limit = int(_MAX_SMEAR_SAMPLES / abs(drift.ratio - 1.0))
        frames = min(frames, max(_MIN_WINDOW_FRAMES, limit))
    return frames


def _channel_map(reference: np.ndarray, test: np.ndarray, lag: int) -> tuple[int, ...]:
    """Stereo kanallarin yer degistirip degistirmedigini bulur.

    2x2 korelasyon matrisi: capraz terimler duz terimlerden buyukse test'in
    kanallari takasli. Mono'ya yakin icerikte ikisi esit olur ve takas zaten
    fark yaratmaz; o yuzden esitlikte kimlik eslemesi korunur.
    """
    channels = reference.shape[1]
    identity = tuple(range(channels))
    if channels < 2:
        return identity
    c = [
        [abs(gccphat.correlation_at(reference[:, i], test[:, j], lag)) for j in (0, 1)]
        for i in (0, 1)
    ]
    if c[0][1] + c[1][0] > c[0][0] + c[1][1]:
        return (1, 0, *identity[2:])
    return identity


def build(
    reference_env: Envelope,
    test_env: Envelope,
    read_reference: WindowReader,
    read_test: WindowReader,
    sample_rate: int,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    search_s: float = DEFAULT_SEARCH_S,
) -> AlignmentPlan:
    """Hizalama planini cikarir.

    `reference_env`/`test_env`: `envelope.build` ciktilari. Surukelenme olcumu
    icin ornekleri (`keep_samples=True`) gerekir; yoksa atlanir ve bir gerekce
    olarak raporlanir.

    `read_*`: tam hizda (`sample_rate`) pencere okuyucu. Iki dosya ayni hiza
    getirilmis olmali (bkz. `ResampleCfg`).
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate pozitif olmali")
    reasons: list[str] = []

    envelope = coarse_match(reference_env, test_env)

    drift: DriftEstimate | None = None

    def result(
        verdict: Verdict,
        *,
        delay_samples: float = math.nan,
        position_s: float = math.nan,
        lag: LagEstimate | None = None,
        fine: SubSampleEstimate | None = None,
        polarity: int = 1,
        gain_db: float = math.nan,
        channel_map: tuple[int, ...] = (),
        track: tuple[tuple[float, float], ...] = (),
    ) -> AlignmentPlan:
        return AlignmentPlan(
            verdict=verdict,
            delay_s=delay_samples / sample_rate,
            delay_samples=delay_samples,
            position_s=position_s,
            sample_rate=sample_rate,
            envelope=envelope,
            drift=drift,
            lag=lag,
            fine=fine,
            polarity=polarity,
            gain_db=gain_db,
            channel_map=channel_map,
            reasons=tuple(reasons),
            track=track,
        )

    # -- surukelenme ---------------------------------------------------------
    # int16 ornekler OLDUGU GIBI verilir: hiz tahmini olcekten bagimsizdir ve
    # pencereleri kendisi cevirir. Tam boy float64 kopya 60 dakikalik ciftte
    # 550 MB tutuyordu (denetim D5).
    ref_samples, test_samples = reference_env.samples, test_env.samples
    if ref_samples is None or test_samples is None:
        reasons.append(
            Message("plan.no_samples", "speed ratio not measured: envelope kept no samples")
        )
    else:
        drift = drift_mod.estimate_from_audio(ref_samples, test_samples, reference_env.sample_rate)
        if drift.is_resampling:
            label = drift.label or f"{drift.ppm:+.0f} ppm"
            reasons.append(
                Message(
                    "plan.speed",
                    "different speed ({label}, ratio {ratio:.6f}); "
                    "the difference is not a codec difference",
                    label=label,
                    ratio=drift.ratio,
                )
            )
            return result("speed_mismatch")
        if drift.needs_tracking:
            reasons.append(
                Message(
                    "plan.drift", "clock drift {ppm:+.1f} ppm; delay must be tracked", ppm=drift.ppm
                )
            )

    if drift is not None and drift.periodic:
        # Sabit ton ya da dongu: her periyotta esit tepe var, hangi gecikmenin
        # dogru oldugu bilinemez. Devam edilirse L2 rastgele bir periyot katina
        # hizalar ve olcum anlamsiz olur; zarf da duz oldugu icin "farkli kayit"
        # denirdi, bu da yanlis (denetim D6).
        reasons.append(
            Message(
                "plan.periodic",
                "the signal repeats itself (a steady tone or a loop): the delay between the "
                "files is ambiguous and cannot be measured",
            )
        )
        return result("unaligned")

    anchored = drift is not None and drift.status != "unreliable"

    # -- "ayni kayit mi" ilk kapi -------------------------------------------
    if envelope.rho < MIN_ENVELOPE_CORRELATION:
        if not anchored:
            # "Farkli kayit" demeden once iki baska aciklama: dosyalardan biri
            # sessiz, ya da ayni kayit ama yarisindan azi ortusuyor. Ikisi de
            # "farkli kayit" diye etiketleniyordu (denetim D34).
            for env, which in ((reference_env, "reference"), (test_env, "test")):
                if _is_silent(env):
                    reasons.append(
                        Message(
                            "plan.silent",
                            "the {which} file is silent: there is nothing to align",
                            which=which,
                        )
                    )
                    return result("unaligned")
            partial = coarse_match(
                reference_env, test_env, min_overlap_fraction=_PARTIAL_OVERLAP_FRACTION
            )
            shorter = min(reference_env.frames, test_env.frames)
            if partial.rho >= MIN_ENVELOPE_CORRELATION and shorter:
                reasons.append(
                    Message(
                        "plan.partial_overlap",
                        "the files overlap for only {seconds:.0f} s ({percent:.0f}% of the "
                        "shorter one): too little to align reliably",
                        seconds=reference_env.seconds(partial.overlap_frames),
                        percent=100.0 * partial.overlap_frames / shorter,
                    )
                )
                return result("unaligned")
            reasons.append(
                Message(
                    "plan.different_recording",
                    "envelope correlation {rho:.2f} and no consistent anchors: "
                    "the files do not contain the same recording",
                    rho=envelope.rho,
                )
            )
            return result("different_recording")
        reasons.append(
            Message(
                "plan.flat_envelope",
                "envelope correlation {rho:.2f} is low but anchors agree; "
                "envelope is uninformative for this material",
                rho=envelope.rho,
            )
        )

    # Kaba gecikme: capalar varsa onlardan (ornek hassasiyetinde ve kayda bagli),
    # yoksa zarftan.
    def coarse_delay_at(position_s: float) -> float:
        if anchored and drift is not None:
            return drift.offset_s + (1.0 - drift.ratio) * position_s
        return envelope.lag_s

    # -- L2 + L3 ---------------------------------------------------------------
    frames = _window_frames(sample_rate, window_s, drift)
    search = int(search_s * sample_rate)
    delay_guess = coarse_delay_at(0.0)
    # test zamaninda ortusme: test[T] = reference[T - d]
    lo_s = max(0.0, delay_guess) + search_s
    hi_s = min(test_env.duration_s, reference_env.duration_s + delay_guess) - search_s
    span = int((hi_s - lo_s) * sample_rate)
    if span < _MIN_WINDOW_FRAMES:
        reasons.append(Message("plan.short_overlap", "overlap too short to align"))
        return result("unaligned")
    frames = min(frames, span)

    tracking = drift is not None and drift.needs_tracking
    points: list[_Point] = []
    positions = list(_TRACK_POSITIONS if tracking else _POSITIONS)
    extended = tracking
    index = 0
    while True:
        if index == len(positions):
            if points or extended:
                break
            positions.extend(_FALLBACK_POSITIONS)
            extended = True
        fraction = positions[index]
        index += 1
        test_start = int(lo_s * sample_rate) + int((span - frames) * fraction)
        position_s = (test_start + frames / 2) / sample_rate
        coarse = round(coarse_delay_at(position_s) * sample_rate)
        ref_start = test_start - coarse
        if ref_start < 0:
            continue
        if _mostly_silent(reference_env, ref_start / sample_rate, frames / sample_rate):
            continue

        ref_block = np.asarray(read_reference(ref_start, frames), dtype=np.float64)
        test_block = np.asarray(read_test(test_start, frames), dtype=np.float64)
        if ref_block.ndim != 2 or test_block.ndim != 2:
            raise ValueError("WindowReader (frames, channels) seklinde dizi dondurmeli")
        if ref_block.shape[1] != test_block.shape[1]:
            reasons.append(
                Message(
                    "plan.channels",
                    "channel count differs ({ref} vs {test}); choose a downmix explicitly",
                    ref=ref_block.shape[1],
                    test=test_block.shape[1],
                )
            )
            return result("channel_mismatch")
        n = min(ref_block.shape[0], test_block.shape[0])
        if n < _MIN_WINDOW_FRAMES:
            continue
        ref_block, test_block = ref_block[:n], test_block[:n]
        ref_mono, test_mono = to_mono(ref_block), to_mono(test_block)

        lag = gccphat.estimate(ref_mono, test_mono, max_lag=search)
        fine = refine.refine(ref_mono, test_mono, lag.lag)
        if not fine.usable:
            continue
        points.append(
            _Point(position_s, coarse + lag.lag + fine.delay, lag, fine, ref_block, test_block)
        )
        # Sabit gecikmede "tamam" bir pencerede dur; "zayif" bir pencereyle
        # yetinme. Duzenlenmis bir dosyada pencerelerden biri kesime denk gelip
        # zayif cikabilir; ilk kullanilabilir pencerede durmak tum dosyayi
        # "farkli master" yapiyordu (denetim D1).
        if not tracking and fine.status == "ok":
            break

    if not points:
        reasons.append(Message("plan.no_window", "no analysis window produced a valid alignment"))
        return result("unaligned")

    # Hukum ve kazanc ilk (izlemede: en iyi) noktadan; izleme noktalarindan
    # yalnizca "ok" olanlar dogruya girer.
    best = max(points, key=lambda pt: abs(pt.fine.correlation))
    track = tuple((pt.position_s, pt.delay) for pt in points if pt.fine.status == "ok")
    if tracking and len(track) < 2:
        reasons.append(
            Message(
                "plan.track_points",
                "clock drift needs at least two aligned points to track, found {count}",
                count=len(track),
            )
        )
    point = best if tracking else next((p for p in points if p.fine.status == "ok"), points[0])
    lag, fine = point.lag, point.fine
    a, b = gccphat.aligned_slices(to_mono(point.ref), to_mono(point.test), lag.lag)
    shifted = fractional_shift(a, fine.delay)
    core = slice(_CORE_GUARD, a.size - _CORE_GUARD)
    gain = optimal_gain(shifted[core], b[core])

    if fine.status == "ok":
        verdict: Verdict = "aligned"
    elif drift is not None and drift.status == "unreliable":
        # Zarf eslesti ama dalga formu kaydin HICBIR yerinde eslesmiyor: capa
        # yok ve hizali korelasyon dusuk. Tek kanit zarf, ve zarf yalnizca
        # "ne zaman yuksek sesliydi"yi olcer -- ayni ses yuksekligi egrisine
        # sahip iki farkli kayit (ayni duzenlemenin iki icrasi, ayni tremolo
        # ile uretilmis iki gurultu) onu gecer. Ilk surum bunu "farkli master"
        # diye etiketleyip OLCUYORDU: zarf 0.917, capa 0, r 0.024.
        reasons.append(
            Message(
                "plan.shared_contour",
                "envelopes match but waveforms do not (no consistent anchors, aligned "
                "correlation {r:.2f}): not the same recording",
                r=fine.correlation,
            )
        )
        return result("different_recording", position_s=point.position_s, lag=lag, fine=fine)
    else:
        verdict = "different_master"
        reasons.append(
            Message(
                "plan.different_master",
                "aligned correlation {r:.2f} is too low for a pure time shift: "
                "different master, EQ or partial overlap",
                r=fine.correlation,
            )
        )
    return result(
        verdict,
        delay_samples=point.delay,
        position_s=point.position_s,
        lag=lag,
        fine=fine,
        polarity=-1 if gain < 0 else 1,
        gain_db=db(abs(gain)),
        channel_map=_channel_map(point.ref, point.test, lag.lag),
        track=track,
    )
