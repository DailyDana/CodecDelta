"""ABX kor dinleme sekmesi.

Analiz sekmesindeki son karsilastirmadan bir kesit cifti hazirlar ve
`app.abx.session` kurallariyla bir test yurutur. Kurallarin kendisi oturum
modulunde; burada yalnizca akis ve gosterim var. Ozellikle:

- Deneme sirasinda dogru/yanlis GOSTERILMEZ (pratik modu haric).
- Her yanittan sonra calma durur: yeni X, eski X'in calmaya devam etmesiyle
  ele verilmesin.
- Kritik kesit secildiginde "genellenemez" rozeti surekli gorunur.
- Sonuc metni oturumun `describe()`'indan gelir; "fark yok" yazilmaz.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QElapsedTimer, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.abx import excerpts
from app.abx.session import AbxSession, Excerpt, ExcerptKind, Mode, Source
from app.compare.pipeline import Track
from app.compare.result import ComparisonResult
from app.core import stats
from app.core.errors import UnsupportedInputError
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.ui.abx_player import AbxPlayer, playback_rate
from app.ui.i18n import localize, tr
from app.ui.theme import TONE_COLORS
from app.ui.worker import Runner

_VERDICT_TONE = {"shown": "good", "not_shown": "warn", "inconclusive": "warn", "none": "neutral"}


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(8)
    return frame, layout


def _section(title: str) -> QLabel:
    label = QLabel(title.upper())
    label.setObjectName("SectionTitle")
    return label


@dataclass(frozen=True)
class AbxPair:
    """Teste konu olan karsilastirma."""

    result: ComparisonResult
    reference: Track
    test: Track

    @property
    def key(self) -> tuple[str, str]:
        return (str(self.reference.path), str(self.test.path))


class AbxTab(QWidget):
    finished = pyqtSignal(object)

    def __init__(
        self,
        tools: FFmpegTools,
        parent: QWidget | None = None,
        *,
        player: AbxPlayer | None = None,
        rate: Callable[[], int] = playback_rate,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(parent)
        self.tools = tools
        self.player = player if player is not None else AbxPlayer(self)
        self._rate = rate
        self._rng = rng or random.SystemRandom()
        self.runner = Runner(self)
        self.pair: AbxPair | None = None
        self.session: AbxSession | None = None
        # Ayni cift icin denenen kesit sayisi (Sidak).
        self._tried: dict[tuple[str, str], int] = {}
        self._timer = QElapsedTimer()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # -- kurulum ------------------------------------------------------------
        setup_card, setup = _card()
        setup.addWidget(_section(tr("abx.setup")))
        self.pair_label = QLabel(tr("abx.no_pair"))
        self.pair_label.setObjectName("Muted")
        self.pair_label.setWordWrap(True)
        setup.addWidget(self.pair_label)
        form = QFormLayout()
        self.mode = QComboBox()
        for key in ("fixed", "sprt", "practice"):
            self.mode.addItem(tr(f"abx.mode.{key}"), key)
        self.trials = QSpinBox()
        self.trials.setRange(8, 40)
        self.trials.setValue(16)
        self.kind = QComboBox()
        for key in ("critical", "random", "manual"):
            self.kind.addItem(tr(f"abx.kind.{key}"), key)
        self.start = QDoubleSpinBox()
        self.start.setRange(0.0, 36000.0)
        self.start.setSuffix(" s")
        self.length = QDoubleSpinBox()
        self.length.setRange(5.0, 30.0)
        self.length.setValue(excerpts.DEFAULT_LENGTH_S)
        self.length.setSuffix(" s")
        form.addRow(tr("abx.mode"), self.mode)
        form.addRow(tr("abx.trials"), self.trials)
        form.addRow(tr("abx.kind"), self.kind)
        form.addRow(tr("abx.start"), self.start)
        form.addRow(tr("abx.length"), self.length)
        setup.addLayout(form)
        self.requirement = QLabel()
        self.requirement.setObjectName("Muted")
        self.requirement.setWordWrap(True)
        setup.addWidget(self.requirement)
        self.badge = QLabel(tr("abx.not_general"))
        self.badge.setStyleSheet(f"color: {TONE_COLORS['warn']};")
        setup.addWidget(self.badge)
        row = QHBoxLayout()
        self.prepare_button = QPushButton(tr("abx.prepare"))
        self.prepare_button.setObjectName("Primary")
        self.stage = QLabel()
        self.stage.setObjectName("Stage")
        row.addWidget(self.prepare_button)
        row.addWidget(self.stage, 1)
        setup.addLayout(row)
        layout.addWidget(setup_card)

        # -- dinleme ------------------------------------------------------------
        listen_card, listen = _card()
        listen.addWidget(_section(tr("abx.listen")))
        self.progress = QLabel()
        listen.addWidget(self.progress)
        sources = QHBoxLayout()
        self.play_a = QPushButton("A")
        self.play_b = QPushButton("B")
        self.play_x = QPushButton("X")
        self.stop_play = QPushButton(tr("abx.stop_play"))
        for button in (self.play_a, self.play_b, self.play_x, self.stop_play):
            sources.addWidget(button)
        # Calan kaynak isaretli gorunur: ayni seste gecis duyulmaz, "gecti mi"
        # ancak boyle anlasilir. X calarken yalnizca X isaretlenir.
        for button in (self.play_a, self.play_b, self.play_x):
            button.setCheckable(True)
        listen.addLayout(sources)
        answers = QHBoxLayout()
        self.answer_a = QPushButton(tr("abx.x_is_a"))
        self.answer_b = QPushButton(tr("abx.x_is_b"))
        self.stop_test = QPushButton(tr("abx.stop_test"))
        for button in (self.answer_a, self.answer_b, self.stop_test):
            answers.addWidget(button)
        listen.addLayout(answers)
        self.feedback = QLabel()
        listen.addWidget(self.feedback)
        hint = QLabel(tr("abx.keys"))
        hint.setObjectName("Muted")
        listen.addWidget(hint)
        layout.addWidget(listen_card)

        # -- sonuc --------------------------------------------------------------
        self.result_card, result = _card()
        result.addWidget(_section(tr("abx.result")))
        self.verdict = QLabel()
        self.verdict.setObjectName("HeadlineTitle")
        self.verdict.setWordWrap(True)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.save_log = QPushButton(tr("abx.save_log"))
        result.addWidget(self.verdict)
        result.addWidget(self.details)
        result.addWidget(self.save_log, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.result_card)
        layout.addStretch(1)
        self.result_card.hide()

        # -- baglantilar --------------------------------------------------------
        self.mode.currentIndexChanged.connect(lambda _: self._refresh())
        self.trials.valueChanged.connect(lambda _: self._refresh())
        self.kind.currentIndexChanged.connect(lambda _: self._refresh())
        self.prepare_button.clicked.connect(self.prepare)
        self.play_a.clicked.connect(lambda: self.play("A"))
        self.play_b.clicked.connect(lambda: self.play("B"))
        self.play_x.clicked.connect(lambda: self.play("X"))
        self.stop_play.clicked.connect(self.stop_playback)
        self.answer_a.clicked.connect(lambda: self.answer("A"))
        self.answer_b.clicked.connect(lambda: self.answer("B"))
        self.stop_test.clicked.connect(self.stop)
        self.save_log.clicked.connect(self.save)
        self.runner.succeeded.connect(self._on_prepared)
        self.runner.failed.connect(self._on_failed)
        self.runner.stage.connect(lambda key: self.stage.setText(tr(f"stage.{key}")))
        for key, action in (
            ("A", lambda: self.play("A")),
            ("B", lambda: self.play("B")),
            ("X", lambda: self.play("X")),
            ("Space", self.stop_playback),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
        self._refresh()

    # -- durum ----------------------------------------------------------------

    def set_pair(self, pair: AbxPair | None) -> None:
        """Analiz sekmesindeki karsilastirmayi teste baglar."""
        if self.session is not None and not self.session.finished:
            self.stop()
        self.player.release()
        self._mark_playing(None)
        self.pair = pair
        self.session = None
        self.result_card.hide()
        if pair is None:
            self.pair_label.setText(tr("abx.no_pair"))
        else:
            self.pair_label.setText(
                tr("abx.pair", reference=pair.reference.path.name, test=pair.test.path.name)
            )
        self._refresh()

    def _mode(self) -> Mode:
        mode: Mode = self.mode.currentData()
        return mode

    def _kind(self) -> ExcerptKind:
        kind: ExcerptKind = self.kind.currentData()
        return kind

    def _refresh(self) -> None:
        mode, kind = self._mode(), self._kind()
        active = self.session is not None and not self.session.finished
        self.badge.setVisible(kind != "random")
        if mode == "fixed":
            n = self.trials.value()
            needed = stats.min_correct_for_significance(n)
            self.requirement.setText(
                tr("abx.need", needed=needed or n, n=n, power=100 * stats.power_at(n, 0.75))
            )
        elif mode == "sprt":
            self.requirement.setText(tr("abx.need_sprt"))
        else:
            self.requirement.setText(tr("abx.need_practice"))
        busy = self.runner.busy
        self.prepare_button.setEnabled(self.pair is not None and not busy and not active)
        for widget in (self.mode, self.kind, self.length):
            widget.setEnabled(not active)
        self.trials.setEnabled(mode == "fixed" and not active)
        self.start.setEnabled(kind == "manual" and not active)
        for button in (
            self.play_a,
            self.play_b,
            self.play_x,
            self.stop_play,
            self.answer_a,
            self.answer_b,
            self.stop_test,
        ):
            button.setEnabled(active)
        if active and self.session is not None:
            done = len(self.session.trials) + 1
            if self.session.mode == "fixed":
                self.progress.setText(tr("abx.trial_of", k=done, n=self.session.trials_planned))
            else:
                self.progress.setText(tr("abx.trial", k=done))
        else:
            self.progress.setText("")

    # -- hazirlik -------------------------------------------------------------

    def prepare(self) -> None:
        pair = self.pair
        if pair is None or self.runner.busy:
            return
        kind, length = self._kind(), self.length.value()
        manual = self.start.value()
        rng = self._rng
        ffmpeg = self.tools.ffmpeg
        rate = self._rate()

        def job(token: CancelToken, stage: Callable[[str], None]) -> excerpts.ExcerptAudio:
            stage("excerpt")
            if kind == "critical":
                start = excerpts.critical_start(pair.result, length)
            elif kind == "random":
                start = excerpts.random_start(pair.result, length, rng)
            else:
                span = excerpts.valid_range(pair.result, length)
                start = None if span is None else min(max(manual, span[0]), span[1])
            if start is None:
                raise UnsupportedInputError(tr("abx.no_room"))
            return excerpts.prepare(
                ffmpeg,
                pair.reference,
                pair.test,
                pair.result,
                start,
                length,
                rate=rate,
                cancel=token,
            )

        self.result_card.hide()
        self.runner.start(job)
        self._refresh()

    def _on_prepared(self, audio: object, seconds: float) -> None:
        assert isinstance(audio, excerpts.ExcerptAudio)
        pair = self.pair
        if pair is None:
            return
        self._mark_playing(None)
        try:
            self.player.load(audio)
        except RuntimeError as exc:
            QMessageBox.critical(self, tr("abx.error"), str(exc))
            self.stage.setText("")
            self._refresh()
            return
        tried = self._tried.get(pair.key, 0) + 1
        self._tried[pair.key] = tried
        self.start.setValue(audio.start_s)
        self.session = AbxSession(
            self._mode(),
            Excerpt(self._kind(), audio.start_s, audio.length_s),
            trials_planned=self.trials.value(),
            excerpts_tried=tried,
            rng=self._rng,
        )
        self.stage.setText(tr("abx.ready", start=audio.start_s, length=audio.length_s))
        self.feedback.setText("")
        self._timer.start()
        self._refresh()

    def _on_failed(self, message: str) -> None:
        self.stage.setText("")
        QMessageBox.critical(self, tr("abx.error"), message)
        self._refresh()

    # -- deneme ---------------------------------------------------------------

    def play(self, which: str) -> None:
        session = self.session
        if session is None or session.finished:
            return
        source: Source = session.x_source if which == "X" else ("A" if which == "A" else "B")
        session.note_switch()
        self.player.play(source)
        self._mark_playing(which)

    def stop_playback(self) -> None:
        """Calmayi durdurur (test surer)."""
        self.player.stop()
        self._mark_playing(None)

    def _mark_playing(self, which: str | None) -> None:
        for key, button in (("A", self.play_a), ("B", self.play_b), ("X", self.play_x)):
            button.setChecked(key == which)

    def answer(self, choice: Source) -> None:
        session = self.session
        if session is None or session.finished:
            return
        self.stop_playback()
        feedback = session.answer(choice, float(self._timer.elapsed()))
        if feedback is not None:
            self.feedback.setText(tr("abx.correct") if feedback else tr("abx.wrong"))
        self._timer.restart()
        if session.finished:
            self._show_result()
        self._refresh()

    def stop(self) -> None:
        session = self.session
        if session is None:
            return
        self.stop_playback()
        session.stop()
        self._show_result()
        self._refresh()

    def _show_result(self) -> None:
        session = self.session
        if session is None:
            return
        verdict = session.verdict()
        tone = TONE_COLORS[_VERDICT_TONE[verdict]]
        self.result_card.setStyleSheet(f"QFrame#Card {{ border-left: 4px solid {tone}; }}")
        self.verdict.setText(tr(f"abx.verdict.{verdict}"))
        self.verdict.setStyleSheet(f"color: {tone};")
        self.details.setText("\n".join(localize(m) for m in session.describe()))
        self.save_log.setVisible(session.mode != "practice" and bool(session.trials))
        self.result_card.show()
        self.feedback.setText("")
        self.finished.emit(session)

    def log(self) -> dict[str, object]:
        """Kaydedilecek gunluk: oturum + dosya ADLARI (yol yok, paylasilabilir)."""
        assert self.session is not None and self.pair is not None
        data: dict[str, object] = dict(self.session.to_log())
        data["reference"] = self.pair.reference.path.name
        data["test"] = self.pair.test.path.name
        return data

    def save(self) -> None:
        if self.session is None or self.pair is None or self.session.mode == "practice":
            return
        suggested = self.pair.test.path.with_name(f"{self.pair.test.path.stem}_abx.json")
        path, _ = QFileDialog.getSaveFileName(
            self, tr("abx.save_log"), str(suggested), "JSON (*.json)"
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.log(), indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, tr("abx.error"), str(exc))
            return
        self.stage.setText(tr("abx.saved", name=Path(path).name))
