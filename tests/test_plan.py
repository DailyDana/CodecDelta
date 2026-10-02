"""Hizalama plani testleri.

Her senaryo bilinen bir donusum ENJEKTE eder (gecikme, kesir, kazanc,
polarite, kanal takasi, hiz orani) ve planin onu geri bulmasini ya da dogru
hukmu vermesini bekler.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.test_drift import music_like, resample
from tests.test_envelope import dynamic_noise

from app.align import plan
from app.align.envelope import from_samples
from app.dsp.transforms import fractional_shift, to_mono

RATE = 8000


def stereo(seconds: float, seed: int = 0) -> np.ndarray:
    """Iki bagimsiz, dinamigi olan kanal. Bagimsizlik kanal takasini olculebilir yapar."""
    return np.stack([dynamic_noise(seconds, seed), dynamic_noise(seconds, seed + 500)], axis=1)


def delayed(x: np.ndarray, samples: float) -> np.ndarray:
    """Kanal kanal `samples` kadar geciktirir (kesirli olabilir). Pozitif = geride."""
    whole = int(np.floor(samples))
    frac = samples - whole
    padded = np.concatenate([np.zeros((whole, x.shape[1])), x]) if whole > 0 else x[-whole:]
    if frac:
        padded = np.stack(
            [fractional_shift(padded[:, c], frac) for c in range(padded.shape[1])], axis=1
        )
    return padded


def lowpass(x: np.ndarray, keep: float) -> np.ndarray:
    """Bandin alttaki `keep` kesrini birakan alcak geciren, kanal kanal."""
    spectrum = np.fft.rfft(x, axis=0)
    spectrum[int(spectrum.shape[0] * keep) :] = 0.0
    return np.fft.irfft(spectrum, x.shape[0], axis=0)


def run(
    reference: np.ndarray, test: np.ndarray, *, keep_samples: bool = True
) -> plan.AlignmentPlan:
    ref_env = from_samples(to_mono(reference), RATE, keep_samples=keep_samples)
    test_env = from_samples(to_mono(test), RATE, keep_samples=keep_samples)
    return plan.build(
        ref_env,
        test_env,
        lambda start, count: reference[start : start + count],
        lambda start, count: test[start : start + count],
        RATE,
    )


# -- hizalanan durumlar -----------------------------------------------------


def test_recovers_offset_fraction_and_gain() -> None:
    """Ana senaryo: 1.5 s + 0.3 ornek gecikme, 3 dB sessiz."""
    reference = stereo(40.0, seed=1)
    true_delay = 1.5 * RATE + 0.3
    test = delayed(reference, true_delay) * 10 ** (-3 / 20)

    result = run(reference, test)
    assert result.verdict == "aligned", result.reasons
    assert result.comparable
    assert result.delay_samples == pytest.approx(true_delay, abs=0.02)
    assert result.delay_s == pytest.approx(true_delay / RATE, abs=1e-5)
    assert result.gain_db == pytest.approx(-3.0, abs=0.05)
    assert result.polarity == 1
    assert result.channel_map == (0, 1)


def test_excerpt_gives_a_negative_delay() -> None:
    """test, reference'in 3.2 s'den baslayan bir kesiti: test ONDE, gecikme negatif."""
    reference = stereo(60.0, seed=2)
    test = reference[int(3.2 * RATE) : int(3.2 * RATE) + 20 * RATE]
    result = run(reference, test)
    assert result.verdict == "aligned", result.reasons
    assert result.delay_samples == pytest.approx(-3.2 * RATE, abs=0.02)


def test_inverted_polarity_is_reported_not_treated_as_damage() -> None:
    reference = stereo(30.0, seed=3)
    result = run(reference, -delayed(reference, 400))
    assert result.verdict == "aligned", result.reasons
    assert result.polarity == -1
    assert result.gain_db == pytest.approx(0.0, abs=0.05)


def test_swapped_channels_are_detected() -> None:
    reference = stereo(30.0, seed=4)
    result = run(reference, delayed(reference, 250)[:, ::-1])
    assert result.verdict == "aligned", result.reasons
    assert result.channel_map == (1, 0)


def test_works_without_envelope_samples_but_says_so() -> None:
    """Ornek tutulmadiysa hiz orani olculemez; bu sessizce gecilmemeli."""
    reference = stereo(30.0, seed=5)
    result = run(reference, delayed(reference, 800), keep_samples=False)
    assert result.verdict == "aligned"
    assert result.drift is None
    assert any("speed ratio not measured" in r for r in result.reasons)


def noisy_copy(source: np.ndarray, delay: int, snr_db: float, seed: int) -> np.ndarray:
    test = delayed(source, delay)
    scale = np.sqrt(np.mean(source**2)) * 10 ** (-snr_db / 20)
    return test + np.random.default_rng(seed).standard_normal(test.shape) * scale


def test_flat_envelope_does_not_mean_different_recording() -> None:
    """Dinamigi duz icerik + gurultu: zarf bilgisiz, ama ses AYNI.

    Duragan icerigin zarfi yalnizca cerceve enerjisinin rastgele
    dalgalanmasidir; 3 dB S/N'de eklenen bagimsiz gurultu bu dalgalanmayi
    degistirir ve zarf korelasyonu 0.475'e duser -- esigin (0.70) cok altina.
    Dalga formu ise hala 0.82 korelasyonlu. Zarfa tek basina guvenen bir plan
    burada "farkli kayit" derdi.
    """
    source = np.stack([music_like(60.0, seed=6), music_like(60.0, seed=7)], axis=1)
    result = run(source, noisy_copy(source, 800, snr_db=3.0, seed=1))

    assert result.envelope.rho < 0.70, "senaryo zarfi bilgisiz kilmiyor"
    assert result.verdict == "aligned", result.reasons
    assert result.delay_samples == pytest.approx(800, abs=0.05)
    assert any("envelope is uninformative" in r for r in result.reasons)


@pytest.mark.parametrize("seed", range(12))
def test_noise_does_not_invent_drift(seed: int) -> None:
    """0 dB S/N'de sahte surukelenme raporlanmamali.

    Ilk surumde 24 denemenin birinde 0.06 ppm "drift" cikti: aralik icindeki
    1.0001 hipotezi 1.0 ile berabere skor alip kazaniyor, hipotez yolu da
    "ihmal edilebilir" kuralini atliyordu.
    """
    source = np.stack([music_like(60.0, seed=seed), music_like(60.0, seed=seed + 100)], axis=1)
    result = run(source, noisy_copy(source, 800, snr_db=0.0, seed=seed))
    assert result.drift is not None
    assert result.drift.status == "none"


# -- reddedilen durumlar ----------------------------------------------------


def test_unrelated_recordings_are_rejected() -> None:
    result = run(stereo(40.0, seed=8), stereo(40.0, seed=9))
    assert result.verdict == "different_recording"
    assert not result.comparable
    assert np.isnan(result.delay_s)


def test_pal_is_a_speed_mismatch_not_a_codec_difference() -> None:
    reference = stereo(120.0, seed=10)
    test = np.stack([resample(reference[:, c], 25.0 / 24.0) for c in (0, 1)], axis=1)

    result = run(reference, test)
    assert result.verdict == "speed_mismatch"
    assert not result.comparable
    assert result.drift is not None and result.drift.label is not None
    assert "PAL" in result.drift.label
    assert any("not a codec difference" in r for r in result.reasons)


def test_heavy_eq_is_a_different_master() -> None:
    """Ayni kayit, ayni zamanlama, ama spektrum cok farkli.

    Zarf ve capalar esler (ayni kayit), ama hizali korelasyon saf bir zaman
    kaymasi icin fazla dusuk: fark codec'ten degil mastering'den geliyor.
    """
    reference = stereo(40.0, seed=11)
    result = run(reference, lowpass(delayed(reference, 300), 0.15))
    assert result.verdict == "different_master", result.reasons
    assert not result.comparable
    assert result.fine is not None and result.fine.status == "weak"


def test_channel_count_mismatch_asks_instead_of_downmixing() -> None:
    reference = stereo(30.0, seed=12)
    result = run(reference, to_mono(reference)[:, None])
    assert result.verdict == "channel_mismatch"
    assert any("downmix" in r for r in result.reasons)


def test_reader_must_return_two_dimensional_blocks() -> None:
    reference = stereo(20.0, seed=13)
    env = from_samples(to_mono(reference), RATE)
    with pytest.raises(ValueError, match="frames, channels"):
        plan.build(
            env,
            env,
            lambda s, n: to_mono(reference)[s : s + n],
            lambda s, n: to_mono(reference)[s : s + n],
            RATE,
        )


def test_rejects_bad_sample_rate() -> None:
    env = from_samples(dynamic_noise(5.0), RATE)
    with pytest.raises(ValueError, match="sample_rate"):
        plan.build(env, env, lambda s, n: np.zeros((n, 1)), lambda s, n: np.zeros((n, 1)), 0)


def test_shared_loudness_contour_is_not_the_same_recording() -> None:
    """Ayni ses yuksekligi egrisi, farkli dalga formu.

    Zarf yalnizca "ne zaman yuksek sesliydi"yi olcer; ayni duzenlemenin iki
    icrasi onu gecer. Karari dalga formu vermeli. Ilk surum bu durumu "farkli
    master" deyip olcuyordu (zarf 0.917, capa 0, hizali r 0.024).
    """
    rng = np.random.default_rng(40)
    n = 40 * RATE
    t = np.arange(n) / RATE
    contour = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t) * np.sin(2 * np.pi * 0.13 * t))
    a = np.stack([rng.standard_normal(n) * contour for _ in range(2)], axis=1)
    b = np.stack([rng.standard_normal(n) * contour for _ in range(2)], axis=1)

    result = run(a, b)
    assert result.envelope.rho > 0.70, "senaryo zarfi eslestirmiyor"
    assert result.verdict == "different_recording", result.reasons
    assert not result.comparable


# -- periyodik sinyaller (denetim D6) ------------------------------------------


def _tone(seconds: float, freqs: tuple[float, ...]) -> np.ndarray:
    """Sabit ton(lar), stereo."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)
    return np.repeat(np.asarray(x)[:, None], 2, axis=1)


@pytest.mark.parametrize("freqs", [(1000.0,), (440.0,), (1000.0, 1500.0)])
def test_a_steady_tone_is_ambiguous_not_a_speed_change(freqs: tuple[float, ...]) -> None:
    """Sabit tonda her periyotta esit tepe var: hiz orani ya da gecikme uydurulmamali.

    Once capalar rastgele periyot katlarini seciyor ve +3004 ppm ya da
    "NTSC pulldown" raporlaniyordu.
    """
    reference = _tone(30.0, freqs)
    test = delayed(reference, 120)
    # Codec gurultusu yerine: iki dosyada ORTAK olmayan -50 dB gurultu. Ortak
    # gurultu gecikmeyi gercekten belirler ve sinyali periyodik olmaktan cikarir.
    test = test + np.random.default_rng(2).standard_normal(test.shape) * 0.3 * 10 ** (-50 / 20)
    result = run(reference, test)
    assert result.verdict == "unaligned", result.reasons
    assert result.drift is not None and result.drift.periodic
    assert any(getattr(r, "key", "") == "plan.periodic" for r in result.reasons)


def test_heavy_eq_is_not_mistaken_for_a_periodic_signal() -> None:
    """Bandi daraltilmis gurultude PHAT beyazlatmasi yuksek yan tepeler uretir."""
    reference = stereo(40.0, seed=11)
    result = run(reference, lowpass(delayed(reference, 300), 0.15))
    assert result.drift is not None and not result.drift.periodic


def test_a_long_silent_middle_still_aligns() -> None:
    """Uc analiz penceresi de sessizlige dusunce "hizalanamadi" deniyordu (D8)."""
    reference = np.concatenate(
        [stereo(20.0, seed=31), np.zeros((100 * RATE, 2)), stereo(20.0, seed=32)]
    )
    test = delayed(reference, 250)
    test = test + np.random.default_rng(5).standard_normal(test.shape) * 1e-3
    result = run(reference, test)
    assert result.verdict == "aligned", result.reasons
    assert result.delay_samples == pytest.approx(250, abs=0.05)


def test_a_silent_file_is_not_called_a_different_recording() -> None:
    """Tamamen sessiz dosya "farkli kayit" diye etiketleniyordu (D34)."""
    reference = stereo(30.0, seed=41)
    result = run(reference, np.zeros_like(reference))
    assert result.verdict == "unaligned"
    assert any(getattr(r, "key", "") == "plan.silent" for r in result.reasons)


def test_a_partial_overlap_is_not_called_a_different_recording() -> None:
    """Yarisindan azi ortusen ayni kayit "farkli kayit" diye etiketleniyordu (D34)."""
    reference = stereo(40.0, seed=43)
    test = np.concatenate([reference[25 * RATE :], stereo(25.0, seed=44)])
    result = run(reference, test)
    assert result.verdict == "unaligned", result.reasons
    assert any(getattr(r, "key", "") == "plan.partial_overlap" for r in result.reasons)
