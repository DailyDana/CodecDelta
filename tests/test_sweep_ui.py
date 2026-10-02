"""Bit hizi taramasi arayuzu: egri, tablo, duyulabilirlik notu ve ABX koprusu."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from app.core.ffmpeg_locate import FFmpegTools

_APP: list[object] = []


def _qt_app():  # type: ignore[no-untyped-def]
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    _APP.append(app)
    return app


@pytest.mark.needs_ffmpeg
def test_sweep_shows_the_curve_and_hands_a_rung_to_abx(
    ffmpeg_tools: FFmpegTools, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _qt_app()
    from app.core.settings import Settings
    from app.ui import tab_encode
    from app.ui.main_window import MainWindow

    source = tmp_path / "src.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=48000:duration=12:seed=3,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(source),
        ],
        check=True,
    )
    monkeypatch.setattr(tab_encode.sweep, "bitrates_for", lambda spec: (96, 192))
    window = MainWindow(ffmpeg_tools, Settings(language="en", temp_root=str(tmp_path)))
    encode = window.encode
    encode.source.set_path(source)
    encode.codec.setCurrentIndex(encode.codec.findData("opus"))
    assert encode.sweep_button.isEnabled()
    encode.start_sweep()
    assert not encode.start_button.isEnabled()  # tarama surerken kodlama baslatilamaz
    deadline = time.time() + 120
    while (encode.sweep_runner.busy or encode.sweep is None) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
    assert encode.sweep is not None and [p.bitrate_kbps for p in encode.sweep.points] == [96, 192]
    assert encode.sweep_table.rowCount() == 2 and not encode.sweep_card.isHidden()
    assert encode.sweep_table.item(1, 0).text() == "192 kbps"

    encode.sweep_table.selectRow(1)
    encode._sweep_to_abx()
    assert window.tabs.currentWidget() is window.abx
    pair = window.abx.pair
    assert pair is not None and pair.test.path.name.endswith("192k.opus")
    assert pair.reference.path == source
    window.close()
