"""Analiz sekmesi: iki dosya -> karsilastirma; tek dosya -> referanssiz dogrulama.

Arayuz yalnizca motoru cagirir ve `present` satirlarini cizer. Tum uzun isler
`Runner` uzerinden arka planda; sekme is suresince giris alanlarini kilitler.
"""

from __future__ import annotations

import contextlib
import html
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.compare import ladder
from app.compare.pipeline import Track, compare
from app.compare.result import ComparisonResult
from app.core.errors import CodecDeltaError
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.core.settings import Settings
from app.report import html as report_html
from app.single import verdict as single_verdict
from app.single.verdict import Verdict
from app.ui import present
from app.ui.i18n import localize, tr
from app.ui.present import Headline
from app.ui.theme import COLORS, TONE_COLORS
from app.ui.widgets.charts import BandChart, NmrView
from app.ui.widgets.file_slot import FileSlot
from app.ui.worker import Runner


def _section(title: str) -> QLabel:
    label = QLabel(title.upper())
    label.setObjectName("SectionTitle")
    return label


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(8)
    return frame, layout


def _bullets(items: list[tuple[str, str]]) -> str:
    """(isaret, metin) -> zengin metin liste. Isaret rengi anlam tasir.

    Metin HTML'den KACIRILIR: notlar dosyadan gelen dizeler tasir (FLAC vendor,
    etiketler) ve bir `<` zengin metni bozar ya da bicim enjekte eder.
    """
    colors = {"+": COLORS["red"], "-": COLORS["green"], "·": COLORS["overlay1"]}
    lines = []
    for mark, text in items:
        dot = f'<span style="color:{colors[mark]};font-weight:700">{mark}</span>'
        lines.append(f'<div style="margin:2px 0">{dot}&nbsp; {html.escape(localize(text))}</div>')
    return "".join(lines)


class ResultsPanel(QWidget):
    """Son sonucu gosterir. Karsilastirma ve dogrulama ayni paneli kullanir."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.headline_card, head = _card()
        self.title = QLabel()
        self.title.setObjectName("HeadlineTitle")
        self.detail = QLabel()
        self.detail.setObjectName("HeadlineDetail")
        self.detail.setWordWrap(True)
        head.addWidget(self.title)
        head.addWidget(self.detail)
        layout.addWidget(self.headline_card)

        self.summary_card, summary = _card()
        summary.addWidget(_section(tr("section.summary")))
        self.summary_grid = QGridLayout()
        self.summary_grid.setHorizontalSpacing(24)
        self.summary_grid.setVerticalSpacing(4)
        summary.addLayout(self.summary_grid)
        layout.addWidget(self.summary_card)

        self.ladder_card, ladder_box = _card()
        self.ladder_button = QPushButton(tr("action.ladder"))
        self.ladder_label = QLabel()
        self.ladder_label.setWordWrap(True)
        self.ladder_label.setObjectName("Muted")
        row = QHBoxLayout()
        row.addWidget(self.ladder_button)
        row.addWidget(self.ladder_label, 1)
        ladder_box.addLayout(row)
        disclaimer = QLabel(tr("disclaimer.audibility"))
        disclaimer.setObjectName("Muted")
        ladder_box.addWidget(disclaimer)
        layout.addWidget(self.ladder_card)

        self.bands_card, bands = _card()
        bands.addWidget(_section(tr("section.bands")))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [
                tr("bands.band"),
                tr("bands.mid"),
                tr("bands.side"),
                tr("bands.linear"),
                tr("bands.floor"),
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bands.addWidget(self.table)
        self.band_chart = BandChart()
        bands.addWidget(self.band_chart)
        layout.addWidget(self.bands_card)

        self.nmr_card, nmr = _card()
        nmr.addWidget(_section(tr("section.nmr")))
        self.nmr_view = NmrView()
        nmr.addWidget(self.nmr_view)
        layout.addWidget(self.nmr_card)

        self.notes_card, notes = _card()
        self.notes_title = _section(tr("section.notes"))
        notes.addWidget(self.notes_title)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.TextFormat.RichText)
        notes.addWidget(self.notes)
        layout.addWidget(self.notes_card)
        layout.addStretch()
        self.clear()

    # -- yardimcilar ---------------------------------------------------------------

    def clear(self) -> None:
        for card in (
            self.headline_card,
            self.summary_card,
            self.ladder_card,
            self.bands_card,
            self.nmr_card,
            self.notes_card,
        ):
            card.hide()

    def _headline(self, headline: Headline) -> None:
        color = TONE_COLORS[headline.tone]
        self.headline_card.setStyleSheet(f"QFrame#Card {{ border-left: 4px solid {color}; }}")
        self.title.setText(headline.title)
        self.title.setStyleSheet(f"color: {color};")
        self.detail.setText(headline.detail)
        self.detail.setVisible(bool(headline.detail))
        self.headline_card.show()

    def _summary(self, rows: list[tuple[str, str]]) -> None:
        while self.summary_grid.count():
            item = self.summary_grid.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().deleteLater()
        for i, (key, value) in enumerate(rows):
            k = QLabel(key)
            k.setObjectName("Muted")
            v = QLabel(value)
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.summary_grid.addWidget(k, i, 0)
            self.summary_grid.addWidget(v, i, 1)
        self.summary_grid.setColumnStretch(1, 1)
        self.summary_card.setVisible(bool(rows))

    def _notes(self, title: str, items: list[tuple[str, str]]) -> None:
        self.notes_title.setText(title.upper())
        self.notes.setText(_bullets(items))
        self.notes_card.setVisible(bool(items))

    # -- gosterim --------------------------------------------------------------------

    def show_comparison(self, result: ComparisonResult) -> None:
        self.clear()
        self._headline(present.comparison_headline(result))
        self._summary(present.summary_rows(result))
        rows = present.band_rows(result)
        self.table.setRowCount(len(rows))
        muted = COLORS["overlay1"]
        for r, row in enumerate(rows):
            for c, text in enumerate((row.label, row.mid, row.side, row.linear, row.floor)):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter
                    | (Qt.AlignmentFlag.AlignLeft if c == 0 else Qt.AlignmentFlag.AlignRight)
                )
                if not row.measurable:
                    item.setForeground(QColor(muted))
                self.table.setItem(r, c, item)
        self.table.setFixedHeight(self.table.horizontalHeader().height() + 30 * len(rows) + 4)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.band_chart.set_rows(rows)
        self.bands_card.setVisible(bool(rows))
        self.nmr_view.set_summary(result.nmr)
        self.nmr_card.setVisible(result.nmr is not None and result.nmr.frames > 0)
        self.ladder_card.setVisible(result.status == "measured")
        self.ladder_label.setText("")
        notes = [("·", n) for n in (*result.plan.reasons, *result.notes)]
        self._notes(tr("section.notes"), notes)

    def show_verdict(self, verdict: Verdict) -> None:
        self.clear()
        self._headline(present.verdict_headline(verdict))
        self._summary(present.verdict_rows(verdict))
        items = [("+", r) for r in verdict.reasons] + [("-", c) for c in verdict.counter_reasons]
        items += [("·", n) for n in verdict.notes]
        self._notes(tr("section.reasons"), items)


@dataclass(frozen=True)
class AnalyzeState:
    """Analiz sekmesinin dil degisiminde tasinan durumu."""

    reference: Path | None
    reference_stream: int
    test: Path | None
    test_stream: int
    result: ComparisonResult | None
    tracks: tuple[Track, Track] | None
    verdict: tuple[Verdict, Track] | None
    ladder_verdict: ladder.LadderVerdict | None


class AnalyzeTab(QWidget):
    def __init__(
        self, tools: FFmpegTools, settings: Settings, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.tools = tools
        self.settings = settings
        self.runner = Runner(self)
        self.last_result: ComparisonResult | None = None
        self._last_tracks: tuple[Track, Track] | None = None
        # Rapor icin: son dogrulama (ve hangi iz), son merdiven.
        self.last_verdict: tuple[Verdict, Track] | None = None
        self.last_ladder: ladder.LadderVerdict | None = None
        self._verify_track: Track | None = None

        self.reference = FileSlot(tr("slot.reference"), tools.ffprobe)
        self.test = FileSlot(tr("slot.test"), tools.ffprobe)
        for slot in (self.reference, self.test):
            slot.changed.connect(self._update_buttons)
        # Iki dosya birlikte birakilirsa ikincisi diger yuvaya gider; once
        # sessizce atiliyordu (D26).
        self.reference.dropped_more.connect(lambda paths: self.test.set_path(paths[0]))
        self.test.dropped_more.connect(lambda paths: self.reference.set_path(paths[0]))
        slots = QHBoxLayout()
        slots.setSpacing(12)
        slots.addWidget(self.reference)
        slots.addWidget(self.test)

        self.compare_button = QPushButton(tr("action.compare"))
        self.compare_button.setObjectName("Primary")
        self.compare_button.clicked.connect(self.start_compare)
        self.verify_button = QPushButton(tr("action.verify"))
        self.verify_button.clicked.connect(self.start_verify)
        self.cancel_button = QPushButton(tr("action.cancel"))
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.cancel_button.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.stage = QLabel()
        self.stage.setObjectName("Stage")
        actions = QHBoxLayout()
        actions.addWidget(self.compare_button)
        actions.addWidget(self.verify_button)
        actions.addWidget(self.cancel_button)
        actions.addSpacing(12)
        actions.addWidget(self.stage, 1)
        actions.addWidget(self.progress, 1)
        self.report_button = QPushButton(tr("action.save_report"))
        self.report_button.clicked.connect(self.save_report)
        actions.addWidget(self.report_button)

        self.results = ResultsPanel()
        self.results.ladder_button.clicked.connect(self.start_ladder)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.results)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(12)
        layout.addLayout(slots)
        layout.addLayout(actions)
        layout.addWidget(scroll, 1)

        self.runner.stage.connect(self._on_stage)
        self.runner.failed.connect(self._on_failed)
        self.runner.cancelled.connect(lambda: self.stage.setText(tr("stage.cancelled")))
        self.runner.busy_changed.connect(self._on_busy)
        self._update_buttons()

    # -- durum -------------------------------------------------------------------------

    def _tracks(self) -> tuple[Track | None, Track | None]:
        def track(slot: FileSlot) -> Track | None:
            if slot.info is None:
                return None
            return Track(path=slot.info.path, info=slot.info, stream_index=slot.stream_index)

        return track(self.reference), track(self.test)

    def _update_buttons(self) -> None:
        busy = self.runner.busy
        ref, test = self._tracks()
        self.compare_button.setEnabled(not busy and ref is not None and test is not None)
        self.verify_button.setEnabled(not busy and ref is not None)
        self.results.ladder_button.setEnabled(not busy and self.last_result is not None)
        has_result = self.last_result is not None or self.last_verdict is not None
        self.report_button.setEnabled(not busy and has_result)

    def _on_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.cancel_button.setVisible(busy)
        for slot in (self.reference, self.test):
            slot.setEnabled(not busy)
        self._update_buttons()

    def _on_stage(self, key: str) -> None:
        if key.startswith("rung:"):
            self.stage.setText(tr("stage.rung", kbps=key.split(":", 1)[1]))
        else:
            self.stage.setText(tr(f"stage.{key}"))

    def _on_failed(self, message: str) -> None:
        self.stage.setText("")
        QMessageBox.critical(self, tr("error.title"), message)

    def _done(self, seconds: float) -> None:
        self.stage.setText(tr("stage.done", seconds=seconds))

    # -- isler ----------------------------------------------------------------------------

    def start_compare(self) -> None:
        ref, test = self._tracks()
        if ref is None or test is None or self.runner.busy:
            return
        ffmpeg = self.tools.ffmpeg
        tracks = (ref, test)

        def done(result: object, seconds: float) -> None:
            # Izler ancak sonucla BIRLIKTE kaydedilir. Is basinda yazildiginda
            # iptal ya da hata sonrasi merdiven yeni dosyadan kurulup eski
            # sonucla yargilaniyordu (denetim D17).
            self._last_tracks = tracks
            self._on_compare(result, seconds)

        self._connect_once(done)
        self.runner.start(
            lambda token, stage: compare(ffmpeg, ref, test, cancel=token, stage=stage)
        )

    def start_verify(self) -> None:
        ref, _ = self._tracks()
        if ref is None or self.runner.busy:
            return
        ffmpeg = self.tools.ffmpeg

        def job(token: CancelToken, stage: Callable[[str], None]) -> Verdict:
            stage("verify")
            return single_verdict.verify(
                ffmpeg, ref.info, stream_index=ref.stream_index, cancel=token
            )

        self._verify_track = ref
        self._connect_once(self._on_verify)
        self.runner.start(job)

    def start_ladder(self) -> None:
        if self.last_result is None or self._last_tracks is None or self.runner.busy:
            return
        ref, _ = self._last_tracks
        result = self.last_result
        tools = self.tools
        workdir = self.settings.temp_dir() / "ladder"

        def job(token: CancelToken, stage: Callable[[str], None]) -> ladder.LadderVerdict:
            workdir.mkdir(parents=True, exist_ok=True)
            rungs = ladder.build(
                tools.ffmpeg, tools.ffprobe, ref, workdir, cancel=token, stage=stage
            )
            return ladder.judge(rungs, result)

        self._connect_once(self._on_ladder)
        self.runner.start(job)

    def _connect_once(self, handler: Callable[[object, float], None]) -> None:
        """`handler`'i yalnizca bu isin basarisina baglar.

        Uc baglantinin UCU de is bitince sokulur; once yalnizca `succeeded`
        sokuluyordu ve her iste failed/cancelled alicilari birikiyordu (D19).
        """

        def disconnect() -> None:
            for signal, slot in (
                (self.runner.succeeded, once),
                (self.runner.failed, failed),
                (self.runner.cancelled, cancelled),
            ):
                with contextlib.suppress(TypeError):
                    signal.disconnect(slot)

        def once(result: object, seconds: float) -> None:
            disconnect()
            handler(result, seconds)

        def failed(_message: str) -> None:
            disconnect()

        def cancelled() -> None:
            disconnect()

        self.runner.succeeded.connect(once)
        self.runner.failed.connect(failed)
        self.runner.cancelled.connect(cancelled)

    def _on_compare(self, result: object, seconds: float) -> None:
        assert isinstance(result, ComparisonResult)
        self.last_result = result
        self.last_verdict = None
        self.last_ladder = None
        self.results.show_comparison(result)
        self._done(seconds)
        self._update_buttons()

    def _on_verify(self, verdict: object, seconds: float) -> None:
        assert isinstance(verdict, Verdict)
        self.last_result = None
        self.last_ladder = None
        assert self._verify_track is not None
        self.last_verdict = (verdict, self._verify_track)
        self.results.show_verdict(verdict)
        self._done(seconds)
        self._update_buttons()

    def _on_ladder(self, verdict: object, seconds: float) -> None:
        assert isinstance(verdict, ladder.LadderVerdict)
        self._show_ladder(verdict)
        self._done(seconds)

    def _show_ladder(self, verdict: ladder.LadderVerdict) -> None:
        self.last_ladder = verdict
        steps = "  ·  ".join(
            f"{r.bitrate_kbps}k {present.db_text(r.snr_db)}" for r in verdict.rungs
        )
        caption = html.escape(f"{tr('ladder.caption')}: {steps}")
        muted = COLORS["subtext0"]
        headline = html.escape(present.ladder_text(verdict))
        text = f"<b>{headline}</b><br><span style='color:{muted}'>{caption}</span>"
        self.results.ladder_label.setTextFormat(Qt.TextFormat.RichText)
        self.results.ladder_label.setText(text)

    def report_html(self) -> tuple[str, Path] | None:
        """Son sonucun raporu ve onerilen dosya adi; sonuc yoksa None."""
        if self.last_result is not None:
            result = self.last_result
            html_text = report_html.render_comparison(result, ladder=self.last_ladder)
            stem = result.test.path.stem
            folder = result.test.path.parent
        elif self.last_verdict is not None:
            verdict, track = self.last_verdict
            html_text = report_html.render_verification(
                verdict, track.info, stream_index=track.stream_index
            )
            stem = track.path.stem
            folder = track.path.parent
        else:
            return None
        if self.settings.output_dir:
            folder = Path(self.settings.output_dir)
        return html_text, folder / f"{stem}_codecdelta.html"

    def save_report(self) -> None:
        prepared = self.report_html()
        if prepared is None:
            return
        html_text, suggested = prepared
        path, _ = QFileDialog.getSaveFileName(
            self, tr("action.save_report"), str(suggested), "HTML (*.html)"
        )
        if not path:
            return
        try:
            written = report_html.write(Path(path), html_text)
        except (OSError, CodecDeltaError) as exc:
            message = exc.user_message() if isinstance(exc, CodecDeltaError) else str(exc)
            QMessageBox.critical(self, tr("error.report"), localize(message))
            return
        self.stage.setText(tr("report.saved", name=written.name))

    def state(self) -> AnalyzeState:
        """Dil degisiminde arayuz yeniden kurulurken tasinacak durum."""
        return AnalyzeState(
            reference=self.reference.path,
            reference_stream=self.reference.stream_index,
            test=self.test.path,
            test_stream=self.test.stream_index,
            result=self.last_result,
            tracks=self._last_tracks,
            verdict=self.last_verdict,
            ladder_verdict=self.last_ladder,
        )

    def restore(self, state: AnalyzeState) -> None:
        """`state`i geri yukler: dosyalar, iz secimi, sonuclar (denetim D18).

        Once yalnizca dosya yollari tasiniyordu; sonuc kayboluyor ve cok izli
        dosyada iz secimi sessizce varsayilana donuyordu.
        """
        self.load(state.reference, state.test)
        if state.reference is not None:
            self.reference.select_stream(state.reference_stream)
        if state.test is not None:
            self.test.select_stream(state.test_stream)
        self._last_tracks = state.tracks
        if state.result is not None:
            self.last_result = state.result
            self.results.show_comparison(state.result)
        elif state.verdict is not None:
            self.last_verdict = state.verdict
            self._verify_track = state.verdict[1]
            self.results.show_verdict(state.verdict[0])
        if state.ladder_verdict is not None and state.result is not None:
            self._show_ladder(state.ladder_verdict)
        self._update_buttons()

    def load(self, reference: Path | None, test: Path | None) -> None:
        """Komut satirindan ya da testten dosya yuklemek icin."""
        if reference is not None:
            self.reference.set_path(reference)
        if test is not None:
            self.test.set_path(test)
