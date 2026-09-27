"""Enerji zarfi ve kaba eslestirme testleri.

Zarf katmaninin varlik sebebi iki sey: uzun kayitta kisa parcayi bulmak ve
spektral farklara dayanikli olmak. Testler ikisini de dogrudan olcuyor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.align import envelope
from app.align.envelope import Envelope, coarse_match, from_samples
from app.core.ffmpeg_locate import FFmpegTools

RATE = 8000


def dynamic_noise(seconds: float, seed: int = 0, rate: int = RATE) -> np.ndarray:
    """Genligi yavasca degisen gurultu: muzigin dinamigini taklit eder.

    Duz gurultunun zarfi sabittir, yani korelasyonun tutunacagi bir yapi
    olmaz. Gercek muzikte zarfi tasiyan sey bu yavas degisimdir.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    carrier = rng.standard_normal(n)
    # 0.5-4 Hz arasi birkac bilesenle yavas zarf
    t = np.arange(n) / rate
    env = np.zeros(n)
    for _ in range(6):
        freq = rng.uniform(0.3, 4.0)
        env += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    env = 0.2 + 0.8 * (env - env.min()) / max(float(np.ptp(env)), 1e-12)
    out = carrier * env
    return out / np.max(np.abs(out))


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    """Normalize kesim frekansinda alcak geciren (codec benzeri)."""
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(x.size) > cutoff] = 0.0
    return np.fft.irfft(spec, x.size)


# -- zarf uretimi -----------------------------------------------------------


def test_envelope_shape_and_rate() -> None:
    x = dynamic_noise(2.0)
    env = from_samples(x, RATE)
    assert env.hop_hz == pytest.approx(100.0)
    assert env.sample_rate == RATE
    assert env.duration_s == pytest.approx(2.0, abs=0.01)
    # 2 s @ 100 Hz, son pencere tasmayacagi icin birkac cerceve eksik olabilir
    assert 190 <= env.frames <= 200


def test_envelope_is_level_invariant() -> None:
    """6 dB daha sessiz bir kopyanin zarfi AYNI olmali.

    z-skoru bunu garanti eder; olmasa korelasyon seviye farkindan dusardi.
    """
    x = dynamic_noise(3.0, seed=1)
    loud = from_samples(x, RATE, keep_samples=False)
    quiet = from_samples(x * 0.5, RATE, keep_samples=False)
    assert np.allclose(loud.values, quiet.values, atol=1e-5)


def test_silence_does_not_produce_infinities() -> None:
    """Sessiz cerceveler -inf degil, sonlu bir tabana oturmali."""
    x = np.concatenate([dynamic_noise(1.0, seed=2), np.zeros(RATE)])
    env = from_samples(x, RATE, keep_samples=False)
    assert np.all(np.isfinite(env.values))


def test_all_zero_input_is_handled() -> None:
    env = from_samples(np.zeros(RATE), RATE, keep_samples=False)
    assert np.all(env.values == 0.0)


def test_samples_are_kept_as_int16() -> None:
    """2 saatlik kayitta float32 230 MB, int16 115 MB tutar."""
    x = dynamic_noise(1.0, seed=3)
    env = from_samples(x, RATE, keep_samples=True)
    assert env.samples is not None
    assert env.samples.dtype == np.int16
    dropped = from_samples(x, RATE, keep_samples=False)
    assert dropped.samples is None


def test_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="tek boyutlu"):
        from_samples(np.zeros((100, 2)), RATE)


# -- kaba eslestirme --------------------------------------------------------


@pytest.mark.parametrize("shift_s", [0.0, 0.5, 2.0, -0.5, -3.0])
def test_recovers_known_shift(shift_s: float) -> None:
    source = dynamic_noise(20.0, seed=4)
    pad = int(5 * RATE)
    start = pad
    reference = source[start : start + 10 * RATE]
    test_start = start - int(shift_s * RATE)
    test = source[test_start : test_start + 10 * RATE]

    ref_env = from_samples(reference, RATE, keep_samples=False)
    test_env = from_samples(test, RATE, keep_samples=False)
    match = coarse_match(ref_env, test_env)

    assert match.lag_s == pytest.approx(shift_s, abs=0.02)
    assert match.rho > 0.9


def test_finds_short_excerpt_inside_long_recording() -> None:
    """Amiral gemisi vaka: uzun kayitta kisa parcayi bulmak.

    Burada 60 s icinde 5 s'lik bir kesit araniyor; gercekte 2 saatlik konserde
    4 dakikalik parca. Zarf uzerinde calismanin sebebi bu: tam hizda ayni arama
    yuzlerce kat pahali olurdu.
    """
    long_take = dynamic_noise(60.0, seed=5)
    offset_s = 37.5
    start = int(offset_s * RATE)
    excerpt = long_take[start : start + 5 * RATE]

    reference = from_samples(long_take, RATE, keep_samples=False)
    test = from_samples(excerpt, RATE, keep_samples=False)
    match = coarse_match(reference, test)

    # test[t] = reference[t + offset] oldugu icin gecikme NEGATIF offset
    assert match.lag_s == pytest.approx(-offset_s, abs=0.05)
    assert match.rho > 0.99
    # Ortusme kisa zarfin tamami kadar olmali
    assert match.overlap_frames >= 0.9 * test.frames


@pytest.mark.parametrize("seed", range(12))
def test_short_excerpt_is_found_reliably_not_by_luck(seed: int) -> None:
    """Ayni senaryo bircok tohumda: kaba eslestirmenin ana regresyon testi.

    Tek tohumla gecen bir test burada yaniltici oluyordu. Korelasyonun sadece
    ortusme SAYISINA bolundugu ilk surumde ayni sinav 40 tohumun 20'sinde
    basarisizdi (3 s ile 32 s arasi hatalar); Pearson normalizasyonuyla 40/40
    dogru. Bu yuzden parametrik.
    """
    long_take = dynamic_noise(30.0, seed=seed)
    offset_s = 18.3
    start = int(offset_s * RATE)
    excerpt = long_take[start : start + 4 * RATE]

    match = coarse_match(
        from_samples(long_take, RATE, keep_samples=False),
        from_samples(excerpt, RATE, keep_samples=False),
    )
    assert match.lag_s == pytest.approx(-offset_s, abs=0.02)
    assert match.rho > 0.99


def test_excerpt_from_the_very_start_is_not_pulled_inwards() -> None:
    """Bastaki kesit "ortada" bulunmamali.

    Normalizasyon yanlis oldugunda az ortusen uc gecikmeler cezalandirilir ve
    tepe dosyanin ortasina dogru kayar; bu test o hatayi yakalar.
    """
    long_take = dynamic_noise(40.0, seed=6)
    reference = from_samples(long_take, RATE, keep_samples=False)
    test = from_samples(long_take[: 4 * RATE], RATE, keep_samples=False)
    assert abs(coarse_match(reference, test).lag_s) < 0.05


def test_survives_lowpass_like_a_codec() -> None:
    """Alcak geciren bir kopya hala bulunmali.

    Zarfin asil degeri bu: codec yuksek frekanslari silse de "ne zaman yuksek
    sesliydi" degismez.
    """
    source = dynamic_noise(20.0, seed=7)
    reference = source[: 10 * RATE]
    test = lowpass(source[: 10 * RATE], 0.15)

    match = coarse_match(
        from_samples(reference, RATE, keep_samples=False),
        from_samples(test, RATE, keep_samples=False),
    )
    assert abs(match.lag_s) < 0.02
    assert match.rho > 0.85


def test_survives_gain_difference() -> None:
    source = dynamic_noise(15.0, seed=8)
    match = coarse_match(
        from_samples(source, RATE, keep_samples=False),
        from_samples(source * 0.05, RATE, keep_samples=False),
    )
    assert abs(match.lag_s) < 0.02
    assert match.rho > 0.95


def test_unrelated_recordings_score_low() -> None:
    """Iliskisiz ciftte korelasyon dusuk kalmali.

    Kaba eslestirme her zaman bir gecikme dondurur; "bunlar ayni kayit degil"
    hukmunu verecek olan tek sey rho'dur. 40 tohumda olculen ayrim:
    ilgili 0.949..0.977, kesit 1.000, ilgisiz 0.058..0.458.
    """
    a = from_samples(dynamic_noise(20.0, seed=9), RATE, keep_samples=False)
    b = from_samples(dynamic_noise(20.0, seed=99), RATE, keep_samples=False)
    assert coarse_match(a, b).rho < 0.5


def test_max_shift_bounds_the_search() -> None:
    source = dynamic_noise(30.0, seed=10)
    reference = source[: 20 * RATE]
    test = source[int(6 * RATE) : int(6 * RATE) + 20 * RATE]

    wide = coarse_match(
        from_samples(reference, RATE, keep_samples=False),
        from_samples(test, RATE, keep_samples=False),
    )
    assert wide.lag_s == pytest.approx(-6.0, abs=0.05)

    narrow = coarse_match(
        from_samples(reference, RATE, keep_samples=False),
        from_samples(test, RATE, keep_samples=False),
        max_shift_s=2.0,
    )
    assert abs(narrow.lag_s) <= 2.0


def test_empty_envelope_is_handled() -> None:
    empty = Envelope(np.zeros(0, dtype=np.float32), 100.0, RATE, 0.0)
    full = from_samples(dynamic_noise(2.0, seed=11), RATE, keep_samples=False)
    match = coarse_match(full, empty)
    assert match.lag_s == 0.0
    assert match.rho == 0.0


def test_mismatched_frame_rates_are_rejected() -> None:
    a = from_samples(dynamic_noise(2.0, seed=12), RATE, keep_samples=False)
    b = from_samples(dynamic_noise(2.0, seed=12), RATE, hop_ms=20.0, keep_samples=False)
    with pytest.raises(ValueError, match="cerceve hizi"):
        coarse_match(a, b)


# -- gercek ffmpeg ----------------------------------------------------------


@pytest.mark.needs_ffmpeg
def test_build_from_real_file(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Gercek bir dosyadan zarf cikarma ve icindeki kesiti bulma."""
    full = tmp_path / "full.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=30",
            "-af",
            "tremolo=f=0.7:d=0.9",
            "-ac",
            "2",
            str(full),
        ],
        check=True,
        capture_output=True,
    )
    excerpt = tmp_path / "excerpt.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            "12",
            "-t",
            "6",
            "-i",
            str(full),
            "-ac",
            "2",
            str(excerpt),
        ],
        check=True,
        capture_output=True,
    )

    reference = envelope.build(ffmpeg_tools.ffmpeg, full)
    test = envelope.build(ffmpeg_tools.ffmpeg, excerpt, keep_samples=False)

    assert reference.duration_s == pytest.approx(30.0, abs=0.1)
    assert reference.samples is not None
    assert reference.samples.dtype == np.int16

    match = coarse_match(reference, test)
    assert match.lag_s == pytest.approx(-12.0, abs=0.05)
    assert match.rho > 0.85
