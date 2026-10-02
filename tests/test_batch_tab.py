"""Toplu tarama sekmesi: tarama, filtre, CSV, onbellek, Analiz'e gecis."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from app.core.ffmpeg_locate import FFmpegTools

_APP: list[object] = []
NOISE = "anoisesrc=color=pink:sample_rate=44100:duration=30:seed=5,tremolo=f=1.1:d=0.85"


def _qt_app():  # type: ignore[no-untyped-def]
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    _APP.append(app)
    return app


def _wait(app, condition, seconds: float = 120.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.time() + seconds
    while not condition() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture(scope="module")
def library(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("lib")
    ff = str(ffmpeg_tools.ffmpeg)
    clean = root / "clean.flac"
    subprocess.run(
        [ff, "-v", "error", "-f", "lavfi", "-i", NOISE, "-ac", "2", "-c:a", "flac", str(clean)],
        check=True,
    )
    mp3 = root / "x.mp3"
    subprocess.run(
        [ff, "-v", "error", "-i", str(clean), "-c:a", "libmp3lame", "-b:a", "128k", str(mp3)],
        check=True,
    )
    subprocess.run(
        [ff, "-v", "error", "-i", str(mp3), "-c:a", "flac", str(root / "fake.flac")], check=True
    )
    (root / "broken.flac").write_bytes(bytes([0x66, 0x4C, 0x61, 0x43, 0, 0]))
    return root


@pytest.mark.needs_ffmpeg
def test_scan_fills_the_table_filters_exports_and_remembers(
    ffmpeg_tools: FFmpegTools, library: Path, tmp_path: Path
) -> None:
    app = _qt_app()
    from app.ui.tab_batch import BatchTab

    cache = tmp_path / "cache.json"
    tab = BatchTab(ffmpeg_tools, cache_path=lambda: cache)
    tab.folder.setText(str(library))
    tab.start()
    _wait(app, lambda: not tab.runner.busy and tab.entries)
    buckets = sorted(e.bucket for e in tab.entries)
    assert buckets == ["consistent_lossless", "consistent_lossy", "error"]
    assert tab.table.rowCount() == 3 and "3 / 3" in tab.status.text()

    tab.filter.setCurrentIndex(tab.filter.findData("suspicious"))
    assert tab.table.rowCount() == 1
    assert tab.table.item(0, 0).text() == "fake.flac"

    target = tmp_path / "scan.csv"
    tab.write_csv(target)
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and raw.count(b"\n") == 4

    seen: list[Path] = []
    tab.verify_requested.connect(seen.append)
    tab._open(0, 0)
    assert seen == [library / "fake.flac"]

    # Ikinci tarama onbellekten: dosyalar yeniden cozulmez
    started = time.perf_counter()
    tab.start()
    _wait(app, lambda: not tab.runner.busy and len(tab.entries) == 3)
    assert time.perf_counter() - started < 5.0
    assert all(e.seconds > 0 for e in tab.entries)  # kayitlar ilk taramanin sureleriyle


@pytest.mark.needs_ffmpeg
def test_a_scanned_file_opens_in_analyze_and_survives_a_language_switch(
    ffmpeg_tools: FFmpegTools, library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _qt_app()
    from app.core.settings import Settings
    from app.ui import tab_batch
    from app.ui.main_window import MainWindow

    monkeypatch.setattr(tab_batch, "cache_file", lambda: tmp_path / "cache.json")
    window = MainWindow(ffmpeg_tools, Settings(language="en"))
    window.batch._cache_path = lambda: tmp_path / "cache.json"
    window.batch.folder.setText(str(library))
    window.batch.start()
    _wait(app, lambda: not window.batch.runner.busy and window.batch.entries)

    window._on_verify_requested(library / "fake.flac")
    assert window.tabs.currentWidget() is window.analyze
    _wait(app, lambda: window.analyze.last_verdict is not None and not window.analyze.runner.busy)
    assert window.analyze.last_verdict is not None
    assert window.analyze.last_verdict[0].bucket == "consistent_lossy"

    window.set_language("tr")
    assert len(window.batch.entries) == 3 and window.batch.table.rowCount() == 3
    assert window.batch.folder.text() == str(library)
    window.close()
