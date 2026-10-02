"""Capraz gecis karistiricisi: hicbir gecis ani bir sicrama (tik) uretmemeli.

En kotu durum A = sinus, B = ters sinus: ham kesmede fark 2x genlik kadar
sicrar. Dogru gecis boyunca ornekten ornege degisim, sinusun kendi en buyuk
egimini (2*pi*f/fs) cok az asar.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.abx.mixer import CrossfadeMixer

RATE = 48000
FREQ = 997.0  # periyodu ornek sayisina tam bolunmez: dongu kenari da sinanir
SLOPE = 2 * np.pi * FREQ / RATE  # sinusun en buyuk ornek-arasi degisimi


def _pair(seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(seconds * RATE)) / RATE
    a = np.sin(2 * np.pi * FREQ * t)[:, None].repeat(2, axis=1).astype(np.float32)
    return a, -a


def _max_step(x: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(x[:, 0]))))


def _running(mixer: CrossfadeMixer) -> CrossfadeMixer:
    mixer.start()
    mixer.render(2 * mixer.fade)  # giris rampasi bitsin
    return mixer


def test_switching_to_the_inverted_source_has_no_jump() -> None:
    a, b = _pair()
    mixer = _running(CrossfadeMixer(a, b, RATE))
    before = mixer.render(500)
    mixer.select("B")
    during = mixer.render(1000)
    out = np.concatenate([before, during])
    assert _max_step(out) < SLOPE * 1.5
    # Gecisten sonra tamamen B: bastan B'de calan bir karistiriciyla ayni konumda ayni
    after = mixer.render(200)
    only_b = CrossfadeMixer(a, b, RATE)
    only_b.select("B")
    expected = _running(only_b).render(500 + 1000 + 200)[-200:]
    assert np.allclose(after, expected, atol=1e-6)


def test_fade_is_linear_and_lasts_8_ms() -> None:
    a = np.ones((RATE, 1), dtype=np.float32)
    b = np.zeros((RATE, 1), dtype=np.float32)
    mixer = _running(CrossfadeMixer(a, b, RATE))
    mixer.select("B")
    ramp = mixer.render(mixer.fade + 10)[:, 0]
    assert mixer.fade == round(0.008 * RATE)
    assert np.allclose(np.diff(ramp[: mixer.fade]), -1.0 / mixer.fade, atol=1e-6)
    assert ramp[-1] == 0.0


def test_selecting_the_same_source_changes_nothing() -> None:
    """A -> A ayni makineden gecer ama cikti A olarak kalir."""
    a, b = _pair()
    reference = _running(CrossfadeMixer(a, b, RATE)).render(3000)
    mixer = _running(CrossfadeMixer(a, b, RATE))
    first = mixer.render(1000)
    mixer.select("A")
    rest = mixer.render(2000)
    assert np.allclose(np.concatenate([first, rest]), reference, atol=1e-6)


def test_an_interrupted_fade_stays_smooth() -> None:
    a, b = _pair()
    mixer = _running(CrossfadeMixer(a, b, RATE))
    out = [mixer.render(100)]
    mixer.select("B")
    out.append(mixer.render(mixer.fade // 3))
    mixer.select("A")  # gecis yarida tersine doner
    out.append(mixer.render(mixer.fade * 2))
    assert _max_step(np.concatenate(out)) < SLOPE * 1.5


def test_the_loop_point_has_no_jump() -> None:
    a, b = _pair(0.25)
    mixer = _running(CrossfadeMixer(a, b, RATE))
    out = mixer.render(3 * a.shape[0])  # dongu kenarini birkac kez gecer
    assert _max_step(out) < SLOPE * 1.5


def test_start_and_stop_ramp_from_and_to_silence() -> None:
    a, b = _pair()
    mixer = CrossfadeMixer(a, b, RATE)
    assert np.all(mixer.render(100) == 0.0)  # baslatilmadan sessiz
    mixer.start()
    onset = mixer.render(mixer.fade * 2)
    assert abs(onset[0, 0]) < 0.01 and _max_step(onset) < SLOPE * 1.5
    mixer.stop()
    tail = mixer.render(mixer.fade * 2)
    assert np.all(tail[mixer.fade :] == 0.0) and mixer.silent


def test_switching_keeps_the_playback_position() -> None:
    a, b = _pair()
    mixer = _running(CrossfadeMixer(a, b, RATE))
    mixer.render(4000)
    position = mixer.position_s
    mixer.select("B")
    assert mixer.position_s == position


def test_mismatched_or_tiny_excerpts_are_rejected() -> None:
    a, b = _pair()
    with pytest.raises(ValueError):
        CrossfadeMixer(a, b[:-1], RATE)
    with pytest.raises(ValueError):
        CrossfadeMixer(a[:100], b[:100], RATE)
