"""ABX sekmesi: akis, geri bildirim kurali, sonuc ve gunluk (ses cihazi gerekmez)."""

from __future__ import annotations

import json
import os
import random
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


class FakePlayer:
    """Gercek cihaz yerine cagrilari kaydeder."""

    def __init__(self) -> None:
        self.audio = None
        self.calls: list[str] = []

    @property
    def loaded(self) -> bool:
        return self.audio is not None

    def load(self, audio) -> None:  # type: ignore[no-untyped-def]
        self.audio = audio

    def play(self, source: str) -> None:
        self.calls.append(source)

    def stop(self) -> None:
        self.calls.append("stop")

    def rewind(self) -> None:
        pass

    def release(self) -> None:
        self.audio = None


@pytest.fixture(scope="module")
def pair(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    from app.abx.excerpts import valid_range
    from app.compare.pipeline import compare, open_track
    from app.ui.tab_abx import AbxPair

    root = tmp_path_factory.mktemp("abxtab")
    ff = str(ffmpeg_tools.ffmpeg)
    ref = root / "ref.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=25:seed=5,tremolo=f=0.8:d=0.6",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(ref),
        ],
        check=True,
    )
    test = root / "test.opus"
    subprocess.run(
        [ff, "-v", "error", "-i", str(ref), "-c:a", "libopus", "-b:a", "96k", str(test)],
        check=True,
    )
    a, b = open_track(ffmpeg_tools.ffprobe, ref), open_track(ffmpeg_tools.ffprobe, test)
    result = compare(ffmpeg_tools.ffmpeg, a, b)
    assert result.status == "measured" and valid_range(result, 6.0) is not None
    return AbxPair(result, a, b)


def _wait(app, condition, seconds: float = 120.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.time() + seconds
    while not condition() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(5):
        app.processEvents()


def _tab(ffmpeg_tools: FFmpegTools, mode: str, trials: int = 8):  # type: ignore[no-untyped-def]
    from app.ui.tab_abx import AbxTab

    player = FakePlayer()
    tab = AbxTab(ffmpeg_tools, player=player, rate=lambda: 48000, rng=random.Random(3))  # type: ignore[arg-type]
    tab.mode.setCurrentIndex(tab.mode.findData(mode))
    tab.kind.setCurrentIndex(tab.kind.findData("random"))
    tab.trials.setValue(trials)
    tab.length.setValue(6.0)
    return tab, player


@pytest.mark.needs_ffmpeg
def test_a_fixed_test_runs_without_feedback_and_logs_names_only(
    ffmpeg_tools: FFmpegTools, pair, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    app = _qt_app()
    tab, player = _tab(ffmpeg_tools, "fixed")
    assert not tab.prepare_button.isEnabled()
    tab.set_pair(pair)
    assert "8" in tab.requirement.text()  # kac dogru gerektigi testten once soylenir
    tab.prepare()
    _wait(app, lambda: tab.session is not None)
    session = tab.session
    assert session is not None and player.audio is not None and player.audio.rate == 48000
    assert tab.answer_a.isEnabled() and not tab.prepare_button.isEnabled()

    for _ in range(8):
        tab.play("A")
        tab.play("X")
        assert player.calls[-1] == session.x_source  # X gizli kaynagi calar
        x = session.x_source
        tab.answer(x)
        assert tab.feedback.text() == ""  # deneme sirasinda geri bildirim yok
        assert player.calls[-1] == "stop"  # yanittan sonra calma durur
    assert session.finished and not tab.result_card.isHidden()
    assert session.verdict() == "shown"
    assert "8/8" in tab.details.text()

    target = tmp_path / "log.json"
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "JSON"))
    tab.save()
    log = json.loads(target.read_text(encoding="utf-8"))
    assert log["reference"] == "ref.flac" and log["test"] == "test.opus"
    assert str(pair.reference.path.parent) not in target.read_text(encoding="utf-8")
    assert log["n"] == 8 and log["p_value"] < 0.01


@pytest.mark.needs_ffmpeg
def test_practice_shows_answers_and_a_second_excerpt_tightens_alpha(
    ffmpeg_tools: FFmpegTools, pair
) -> None:  # type: ignore[no-untyped-def]
    app = _qt_app()
    tab, _ = _tab(ffmpeg_tools, "practice")
    tab.set_pair(pair)
    tab.prepare()
    _wait(app, lambda: tab.session is not None)
    assert tab.session is not None
    tab.answer(tab.session.x_source)
    assert tab.feedback.text() in ("Correct", "Doğru")
    tab.stop()
    assert not tab.save_log.isVisible()

    tab.mode.setCurrentIndex(tab.mode.findData("fixed"))
    tab.prepare()
    _wait(app, lambda: tab.session is not None and tab.session.mode == "fixed")
    assert tab.session is not None and tab.session.excerpts_tried == 2


@pytest.mark.needs_ffmpeg
def test_the_analyze_tab_hands_its_comparison_to_abx(ffmpeg_tools: FFmpegTools, pair) -> None:  # type: ignore[no-untyped-def]
    app = _qt_app()
    from app.core.settings import Settings
    from app.ui.main_window import MainWindow

    window = MainWindow(ffmpeg_tools, Settings(language="en"))
    analyze = window.analyze
    analyze.load(pair.reference.path, pair.test.path)
    analyze.start_compare()
    _wait(app, lambda: analyze.last_result is not None and not analyze.runner.busy)
    assert analyze.results.abx_button.isEnabled()
    analyze.results.abx_button.click()
    assert window.tabs.currentWidget() is window.abx
    assert window.abx.pair is not None and window.abx.pair.test.path == pair.test.path
    window.close()
