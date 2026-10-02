"""ABX oynatici: aygit karistiricinin ciktisini dogru bicimde vermeli."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from app.abx.excerpts import ExcerptAudio
from app.abx.mixer import CrossfadeMixer

_APP: list[object] = []


def _qt_app() -> object:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    _APP.append(app)  # referans tutulmazsa toplanir ve surec sessizce kapanir
    return app


def _audio(seconds: float = 1.0, rate: int = 48000) -> ExcerptAudio:
    t = np.arange(int(seconds * rate)) / rate
    a = (0.2 * np.sin(2 * np.pi * 440 * t))[:, None].repeat(2, axis=1).astype(np.float32)
    return ExcerptAudio(a=a, b=a * 0.5, rate=rate, start_s=0.0, b_gain_db=0.0, scale=1.0)


def test_device_streams_the_mixer_as_interleaved_float32() -> None:
    _qt_app()
    from PyQt6.QtCore import QIODevice

    from app.ui.abx_player import MixerDevice

    audio = _audio()
    mixer = CrossfadeMixer(audio.a, audio.b, audio.rate)
    twin = CrossfadeMixer(audio.a, audio.b, audio.rate)
    for m in (mixer, twin):
        m.start()
    device = MixerDevice(mixer)
    device.open(QIODevice.OpenModeFlag.ReadOnly)
    chunks = [bytes(device.read(4096 * 8)) for _ in range(5)]
    data = np.frombuffer(b"".join(chunks), dtype="<f4").reshape(-1, 2)
    assert data.shape[0] == 5 * 4096
    assert np.allclose(data, twin.render(5 * 4096), atol=1e-7)
    assert device.isSequential() and device.bytesAvailable() > 0


def test_player_plays_and_stops_on_a_real_output() -> None:
    """Gercek cihazda kisa bir sessiz calma; cihaz yoksa (CI) atlanir."""
    app = _qt_app()
    from PyQt6.QtMultimedia import QMediaDevices

    from app.ui.abx_player import AbxPlayer, playback_rate

    if QMediaDevices.defaultAudioOutput().isNull():
        pytest.skip("ses cikis cihazi yok")
    rate = playback_rate()
    silent = np.zeros((rate, 2), dtype=np.float32)
    audio = ExcerptAudio(a=silent, b=silent, rate=rate, start_s=0.0, b_gain_db=0.0, scale=1.0)
    player = AbxPlayer()
    try:
        player.load(audio)
    except RuntimeError as exc:
        pytest.skip(f"cihaz bu bicimi kabul etmiyor: {exc}")
    player.play("A")
    deadline = time.time() + 2
    while not player.playing and time.time() < deadline:
        app.processEvents()  # type: ignore[attr-defined]
        time.sleep(0.01)
    assert player.playing
    player.play("B")
    player.stop()
    deadline = time.time() + 2
    while player.playing and time.time() < deadline:
        app.processEvents()  # type: ignore[attr-defined]
        time.sleep(0.01)
    assert not player.playing
    player.release()
