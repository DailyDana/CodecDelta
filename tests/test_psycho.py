"""Psikoakustik model testleri.

Her test modelin FIZIKSEL bir iddiasini sinar: ayni banttaki gurultu uzak
banttakinden daha iyi maskelenir, tonal maskeleyici gurultuden kotu maskeler,
ATH altindaki gurultu duyulmaz, ozdes sinyal gurultusuzdur. Katsayilarin
mutlak dogrulugu iddia edilmiyor (bkz. mask.py docstring); sinanan sey yon ve
buyukluk sirasi.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.dsp.stft import StreamingStft, hann
from app.psycho import mask, nmr
from app.psycho.erb import erb_layout, erb_to_hz, hz_to_bark, hz_to_erb

RATE = 48000
N = 4096
LAYOUT = erb_layout(RATE, N)
MODEL = mask.build(LAYOUT)
SCALE = nmr.power_scale(N)
T = np.arange(N) / RATE


def power(x: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(x * hann(N))
    return (spectrum.real**2 + spectrum.imag**2)[None, :] * SCALE


def band_of(hz: float) -> int:
    return int(np.argmin(np.abs(LAYOUT.centre_hz - hz)))


def db(x: float) -> float:
    return 10.0 * math.log10(x)


# -- olcekler ---------------------------------------------------------------


def test_erb_scale_round_trips() -> None:
    f = np.array([50.0, 1000.0, 12000.0])
    assert np.allclose(erb_to_hz(hz_to_erb(f)), f)


def test_bark_matches_reference_points() -> None:
    """Zwicker tablosu: 1 kHz ~ 8.5 Bark, 8 kHz ~ 21 Bark."""
    assert float(hz_to_bark(1000.0)) == pytest.approx(8.5, abs=0.1)
    assert float(hz_to_bark(8000.0)) == pytest.approx(21.0, abs=0.5)


def test_layout_covers_the_band_without_gaps() -> None:
    edges = LAYOUT.edges
    assert edges[0, 0] <= 2
    assert np.all(edges[1:, 0] == edges[:-1, 1])
    assert 35 <= LAYOUT.count <= 45
    assert np.all(np.diff(edges, axis=1) >= 1)


def test_layout_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="pozitif"):
        erb_layout(0, N)


# -- olcek kalibrasyonu ------------------------------------------------------


def test_full_scale_sine_has_unit_band_energy() -> None:
    """ATH kalibrasyonunun dayanagi: 0 dBFS sinus -> bant enerjisi 1.0 (+-0.3 dB)."""
    for hz in (440.0, 1000.0, 3011.7, 10000.0):
        energy = LAYOUT.energies(power(np.sin(2 * np.pi * hz * T)))[0]
        assert db(float(energy.sum())) == pytest.approx(0.0, abs=0.3)


# -- maskeleme --------------------------------------------------------------


def test_masker_hides_more_in_its_own_band_than_far_away() -> None:
    """Yayilma: esik maskeleyicinin bandinda en yuksek, uzaklastikca duser.

    Yukari yayilma asagi yayilmadan daha yavas (Schroeder: ~-10 dB/Bark
    yukari, ~-25 dB/Bark asagi).
    """
    threshold = mask.threshold(MODEL, power(np.sin(2 * np.pi * 1000 * T)))[0]
    own = db(threshold[band_of(1000)])
    below = db(threshold[band_of(500)])
    above = db(threshold[band_of(2000)])
    far = db(threshold[band_of(6000)])
    assert own > above > far
    assert own > below
    # Asagi yayilma daha dik: 500 Hz (~-3 Bark) 2 kHz'den (~+5 Bark) daha dusuk
    # OLMAMALI mesafe basina; per-Bark egimleri karsilastir.
    dz_below = abs(LAYOUT.centre_bark[band_of(500)] - LAYOUT.centre_bark[band_of(1000)])
    dz_above = abs(LAYOUT.centre_bark[band_of(2000)] - LAYOUT.centre_bark[band_of(1000)])
    assert (own - below) / dz_below > (own - above) / dz_above


def test_noise_masks_better_than_a_tone() -> None:
    """Johnston: gurultu payi ~5.5 dB, tonal pay 14.5 + z dB."""
    rng = np.random.default_rng(0)
    noise_power = power(rng.standard_normal(N) * 0.05)
    tone_power = power(np.sin(2 * np.pi * 4000 * T) * 0.05)
    b = band_of(4000)
    noise_margin = db(LAYOUT.energies(noise_power)[0][b] / mask.threshold(MODEL, noise_power)[0][b])
    tone_margin = db(LAYOUT.energies(tone_power)[0][b] / mask.threshold(MODEL, tone_power)[0][b])
    assert 5.0 < noise_margin < 8.0
    assert tone_margin > noise_margin + 10.0
    assert mask.tonality(noise_power)[0] < 0.1
    assert mask.tonality(tone_power)[0] > 0.9


def test_tonality_is_judged_per_band() -> None:
    """Tonal bas + gurultulu tiz: tiz bandi gurultu payi almali, bas bandi tonal.

    Kuresel tonalite ikisine de ortalama bir pay uygular; gercek bir Opus
    ciftinde bu medyan NMR'i 17 dB yukari itiyordu (mask.py docstring).
    """
    rng = np.random.default_rng(7)
    # 2 kHz: 1 ERB ~ 240 Hz = 20 bin, yerel SFM anlamli. 200 Hz'de bant 4 bin
    # olurdu ve kuresel degere duserdi (tasarim geregi).
    x = 0.3 * np.sin(2 * np.pi * 2000 * T) + 0.01 * rng.standard_normal(N)
    alpha = mask.band_tonality(LAYOUT, power(x))[0]
    assert alpha[band_of(2000)] > 0.8
    assert alpha[band_of(8000)] < 0.15
    global_alpha = float(mask.tonality(power(x))[0])
    assert alpha[band_of(8000)] < global_alpha < alpha[band_of(2000)]


def test_band_tonality_does_not_depend_on_band_width() -> None:
    """Saf ton: dar (200 Hz, ~4 bin) ve genis (8 kHz, ~90 bin) bantta ayni hukum.

    Bant basina SFM burada dusuyordu (20 binde -34 dB tabani -> alpha 0.57).
    """
    for hz in (200.0, 2000.0, 8000.0):
        alpha = mask.band_tonality(LAYOUT, power(np.sin(2 * np.pi * hz * T)))[0]
        assert alpha[band_of(hz)] > 0.9, hz
    rng = np.random.default_rng(8)
    noise_alpha = mask.band_tonality(LAYOUT, power(rng.standard_normal(N)))[0]
    assert np.median(noise_alpha) < 0.2


def test_silence_threshold_is_the_absolute_threshold() -> None:
    """Terhardt: en hassas nokta ~3-4 kHz'de ~-5 dB SPL, 100 Hz'de ~23, 15 kHz'de ~45."""
    threshold = mask.threshold(MODEL, np.zeros((1, N // 2 + 1)))[0]
    spl = 10.0 * np.log10(threshold) + mask.FULL_SCALE_SPL_DB
    assert spl[band_of(3300)] == pytest.approx(-5.0, abs=2.0)
    assert spl[band_of(100)] == pytest.approx(23.0, abs=3.0)
    assert 40.0 < spl[band_of(15000)] < 55.0
    assert (
        int(np.argmin(spl)) == band_of(3300)
        or abs(LAYOUT.centre_hz[int(np.argmin(spl))] - 3300) < 800
    )


# -- NMR ----------------------------------------------------------------------


def _accumulate(reference: np.ndarray, test: np.ndarray, gain: float = 1.0) -> nmr.NmrSummary:
    acc = nmr.NmrAccumulator(RATE, N, gain=gain)
    acc.add(StreamingStft(N).push(reference), StreamingStft(N).push(test))
    return acc.summary()


def test_identical_signals_are_inaudible() -> None:
    x = np.random.default_rng(1).standard_normal(RATE * 2) * 0.1
    summary = _accumulate(x, x.copy())
    assert summary.frames > 0
    assert summary.max_db < -100.0
    assert summary.peak_max_db < -100.0
    assert summary.fraction_above_0 == 0.0


def test_noise_in_the_masker_band_is_hidden_but_far_from_it_is_not() -> None:
    """Ayni seviyedeki gurultu: maskeleyicinin bandinda NMR < 0, 3 oktav ustunde > 0."""
    rng = np.random.default_rng(2)
    reference = np.sin(2 * np.pi * 1000 * np.arange(RATE * 2) / RATE) * 0.5
    noise = rng.standard_normal(RATE * 2) * 0.5 * 10 ** (-40 / 20)

    def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, 1 / RATE)
        spectrum[(freqs < lo) | (freqs > hi)] = 0.0
        return np.fft.irfft(spectrum, x.size)

    near = _accumulate(reference, reference + bandpass(noise, 950, 1050))
    far = _accumulate(reference, reference + bandpass(noise, 7000, 9000))
    for statistic in ("p95_db", "peak_p95_db"):
        assert getattr(near, statistic) < 0.0
        assert getattr(far, statistic) > 0.0
        assert getattr(far, statistic) > getattr(near, statistic) + 20.0


def test_noise_below_the_absolute_threshold_is_inaudible() -> None:
    """Sessiz referans + -110 dBFS gurultu: ATH'nin altinda, NMR negatif."""
    rng = np.random.default_rng(3)
    reference = np.zeros(RATE * 2)
    summary = _accumulate(reference, rng.standard_normal(RATE * 2) * 10 ** (-110 / 20))
    assert summary.frames > 0
    assert summary.peak_max_db < 0.0


def test_louder_noise_gives_higher_nmr_monotonically() -> None:
    rng = np.random.default_rng(4)
    reference = rng.standard_normal(RATE * 2) * 0.1
    noise = rng.standard_normal(RATE * 2)
    values = [
        _accumulate(reference, reference + noise * 0.1 * 10 ** (-snr / 20)).p95_db
        for snr in (40.0, 30.0, 20.0, 10.0)
    ]
    assert values == sorted(values)
    assert values[-1] - values[0] == pytest.approx(30.0, abs=1.5)


def test_gain_is_removed_before_judging() -> None:
    """-6 dB kopya, plan kazanci verildiginde gurultusuz olmali."""
    x = np.random.default_rng(5).standard_normal(RATE) * 0.2
    assert _accumulate(x, x * 0.5, gain=0.5).max_db < -100.0
    assert _accumulate(x, x * 0.5, gain=1.0).max_db > 0.0


@pytest.mark.parametrize(("expected", "columns"), [(69, 69), (5000, 14)])
def test_summary_grid_has_bands_by_time(expected: int, columns: int) -> None:
    """Izgara: beklenen cerceve sayisi 1000'i asarsa sutunlar birlestirilir."""
    x = np.random.default_rng(6).standard_normal(RATE * 3) * 0.1
    acc = nmr.NmrAccumulator(RATE, N, gain=1.0, expected_frames=expected)
    acc.add(StreamingStft(N).push(x), StreamingStft(N).push(x * 0.9))
    summary = acc.summary()
    assert summary.grid_db.shape == (LAYOUT.count, columns)
    assert summary.seconds_per_column == pytest.approx(acc._frames_per_column * 2048 / RATE)
    assert summary.centre_hz.size == LAYOUT.count


def test_all_silent_input_is_not_judged() -> None:
    zeros = np.zeros(RATE)
    summary = _accumulate(zeros, zeros)
    assert summary.frames == 0
    assert math.isnan(summary.p95_db)
    assert math.isnan(summary.fraction_above_0)


def test_rejects_mismatched_spectra() -> None:
    acc = nmr.NmrAccumulator(RATE, N, gain=1.0)
    with pytest.raises(ValueError, match="farkli"):
        acc.add(np.zeros((2, N // 2 + 1)), np.zeros((3, N // 2 + 1)))
