"""Kodlama sekmesi: bir kaynagi secilen codec ve ayarla kodlar.

Uc katman: codec + bitrate (cogu kullanici icin yeter), kalite modu (VBR
olcegi olan codec'lerde), gelismis (codec'e ozel secenekler, gizli baslar).
Secenek listesi `encode.matrix`ten uretilir; bu ffmpeg'de olmayan codec'ler
gri ve sebebi ipucunda.

"Bitince karsilastir" isaretliyse is bitince `encoded` sinyali kaynak ve
cikti ile yayilir; ana pencere Analiz sekmesine gecip karsilastirmayi baslatir.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.compare.pipeline import Track, open_track
from app.core import settings as settings_mod
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.core.settings import Settings
from app.core.tasks import ffmpeg_slot
from app.encode import jobs, matrix, naming, sweep
from app.encode.matrix import CodecSpec, EncodeSettings, Mode
from app.ui.i18n import localize, tr
from app.ui.present import db_text
from app.ui.theme import COLORS
from app.ui.widgets.charts import SweepChart
from app.ui.widgets.file_slot import FileSlot
from app.ui.worker import Runner

_PROGRESS_PREFIX = "progress:"


@dataclass(frozen=True)
class Encoded:
    source: Path
    stream_index: int
    output: Path
    compare: bool


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(8)
    label = QLabel(title.upper())
    label.setObjectName("SectionTitle")
    layout.addWidget(label)
    return frame, layout


class EncodeTab(QWidget):
    encoded = pyqtSignal(object)
    # Ayarlar (orn. cikti klasoru) kaydedildi; diger sekmeler guncellenmeli (D21).
    settings_changed = pyqtSignal(object)
    # Tarama basamagini ABX ile dogrulama istegi: (sonuc, referans, test).
    abx_requested = pyqtSignal(object, object, object)

    def __init__(
        self, tools: FFmpegTools, settings: Settings, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.tools = tools
        self.settings = settings
        self.runner = Runner(self)
        self._taken: set[Path] = set()
        self._option_boxes: dict[str, QComboBox] = {}
        self._pending: Encoded | None = None

        self.source = FileSlot(tr("encode.source"), tools.ffprobe)
        self.source.changed.connect(self._refresh)
        # Tek alan: dikeyde icerigi kadar; yoksa sayfadaki bos yeri yutuyordu.
        self.source.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # -- codec -------------------------------------------------------------------
        codec_card, codec_box = _card(tr("encode.codec"))
        self.codec = QComboBox()
        model = QStandardItemModel()
        self.codec.setModel(model)
        first_available = -1
        for i, spec in enumerate(matrix.SPECS):
            self.codec.addItem(spec.name, spec.key)
            reason = matrix.availability(spec, tools.caps.encoders)
            item = model.item(i)
            if reason is not None and item is not None:
                item.setEnabled(False)
                item.setToolTip(localize(reason))
                item.setText(f"{spec.name}  —  {tr('encode.unavailable')}")
            elif first_available < 0:
                first_available = i
        self.codec.setCurrentIndex(max(0, first_available))
        self.codec.currentIndexChanged.connect(lambda _: self._on_codec())
        self.note = QLabel()
        self.note.setObjectName("Muted")
        self.note.setWordWrap(True)

        self.mode_bitrate = QRadioButton(tr("encode.mode.bitrate"))
        self.mode_quality = QRadioButton(tr("encode.mode.quality"))
        self.mode_bitrate.toggled.connect(lambda _: self._on_mode())
        self.bitrate = QComboBox()
        self.bitrate.currentIndexChanged.connect(lambda _: self._refresh())
        self.quality = QSpinBox()
        self.quality.valueChanged.connect(lambda _: self._refresh())
        self.quality_hint = QLabel()
        self.quality_hint.setObjectName("Muted")
        self.lossless_label = QLabel(tr("encode.mode.lossless"))
        self.lossless_label.setObjectName("Muted")

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_bitrate)
        mode_row.addWidget(self.bitrate)
        mode_row.addSpacing(18)
        mode_row.addWidget(self.mode_quality)
        mode_row.addWidget(self.quality)
        mode_row.addWidget(self.quality_hint)
        mode_row.addWidget(self.lossless_label)
        mode_row.addStretch()

        codec_row = QHBoxLayout()
        codec_row.addWidget(self.codec, 1)
        codec_box.addLayout(codec_row)
        codec_box.addLayout(mode_row)
        codec_box.addWidget(self.note)

        self.advanced_toggle = QPushButton(f"{tr('encode.advanced')}  ▸")
        self.advanced_toggle.setObjectName("Link")
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        self.advanced = QWidget()
        self.advanced_form = QFormLayout(self.advanced)
        self.advanced_form.setContentsMargins(0, 0, 0, 0)
        self.advanced.hide()
        codec_box.addWidget(self.advanced_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        codec_box.addWidget(self.advanced)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {COLORS['yellow']};")
        codec_box.addWidget(self.warning)

        # -- cikti ---------------------------------------------------------------------
        out_card, out_box = _card(tr("encode.output"))
        self.folder = QLineEdit(settings.output_dir)
        self.folder.setPlaceholderText(tr("encode.output_placeholder"))
        self.folder.textChanged.connect(lambda _: self._refresh())
        self.browse = QPushButton(tr("encode.browse"))
        self.browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.browse)
        self.preview = QLabel()
        self.preview.setObjectName("Muted")
        self.compare_after = QCheckBox(tr("encode.compare_after"))
        self.compare_after.setChecked(True)
        out_box.addLayout(folder_row)
        out_box.addWidget(self.preview)
        out_box.addWidget(self.compare_after)

        # -- eylem -----------------------------------------------------------------------
        self.start_button = QPushButton(tr("encode.start"))
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QPushButton(tr("action.cancel"))
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        self.sweep_button = QPushButton(tr("sweep.start"))
        self.sweep_button.setToolTip(tr("sweep.tooltip"))
        self.sweep_button.clicked.connect(self.start_sweep)
        self.sweep_runner = Runner(self)
        self.sweep: sweep.Sweep | None = None
        self._sweep_reference: Track | None = None
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.stage = QLabel()
        self.stage.setObjectName("Stage")
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
        actions.addWidget(self.sweep_button)
        actions.addWidget(self.cancel_button)
        actions.addSpacing(12)
        actions.addWidget(self.stage, 1)
        actions.addWidget(self.progress, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(12)
        layout.addWidget(self.source)
        layout.addWidget(codec_card)
        layout.addWidget(out_card)
        layout.addLayout(actions)
        layout.addWidget(self._build_sweep_card())
        layout.addStretch()

        self.runner.stage.connect(self._on_stage)
        self.runner.succeeded.connect(self._on_done)
        self.runner.failed.connect(self._on_failed)
        self.runner.cancelled.connect(self._on_cancelled)
        self.runner.busy_changed.connect(self._on_busy)
        self.sweep_runner.stage.connect(self._on_sweep_stage)
        self.sweep_runner.succeeded.connect(self._on_sweep_done)
        self.sweep_runner.failed.connect(self._on_failed)
        self.sweep_runner.cancelled.connect(lambda: self.stage.setText(tr("stage.cancelled")))
        self.sweep_runner.busy_changed.connect(self._on_busy)
        self._on_codec()

    # -- secim durumu -------------------------------------------------------------------

    @property
    def spec(self) -> CodecSpec:
        return matrix.by_key(str(self.codec.currentData()))

    def _mode(self) -> Mode:
        spec = self.spec
        if spec.lossless:
            return "lossless"
        if "quality" in spec.modes and (
            self.mode_quality.isChecked() or "bitrate" not in spec.modes
        ):
            return "quality"
        return "bitrate"

    def choice(self) -> EncodeSettings:
        mode = self._mode()
        return EncodeSettings(
            codec=self.spec.key,
            mode=mode,
            bitrate_kbps=int(self.bitrate.currentData()) if mode == "bitrate" else None,
            quality=self.quality.value() if mode == "quality" else None,
            options={key: box.currentData() for key, box in self._option_boxes.items()},
        )

    def apply_choice(self, choice: EncodeSettings) -> None:
        """Bir secimi geri yukler (dil degisiminde arayuz yeniden kurulurken, D18)."""
        position = self.codec.findData(choice.codec)
        if position >= 0:
            self.codec.setCurrentIndex(position)
        if choice.mode == "quality":
            self.mode_quality.setChecked(True)
        else:
            self.mode_bitrate.setChecked(True)
        if choice.bitrate_kbps is not None:
            position = self.bitrate.findData(choice.bitrate_kbps)
            if position >= 0:
                self.bitrate.setCurrentIndex(position)
        if choice.quality is not None:
            self.quality.setValue(choice.quality)
        for key, value in choice.options.items():
            box = self._option_boxes.get(key)
            if box is not None and (position := box.findData(value)) >= 0:
                box.setCurrentIndex(position)
        self._refresh()

    def output_folder(self) -> Path | None:
        text = self.folder.text().strip()
        return Path(text) if text else None

    def planned_output(self) -> Path | None:
        """Onizleme: rezervasyon YAPMADAN hangi ada yazilacagi."""
        if self.source.path is None:
            return None
        return naming.output_path(
            self.source.path,
            matrix.describe(self.choice()),
            self.spec.extension,
            directory=self.output_folder(),
            suffix=self.settings.output_suffix,
            taken=set(self._taken),
        )

    # -- tepkiler ----------------------------------------------------------------------------

    def _on_codec(self) -> None:
        spec = self.spec
        self.bitrate.blockSignals(True)
        self.bitrate.clear()
        for kbps in spec.bitrates:
            self.bitrate.addItem(f"{kbps} kbps", kbps)
        if spec.default_bitrate is not None:
            self.bitrate.setCurrentIndex(spec.bitrates.index(spec.default_bitrate))
        self.bitrate.blockSignals(False)

        quality = spec.quality
        self.quality.blockSignals(True)
        if quality is not None:
            self.quality.setRange(quality.low, quality.high)
            self.quality.setValue(quality.default)
            self.quality_hint.setText(
                tr("encode.quality_higher")
                if quality.higher_is_better
                else tr("encode.quality_lower")
            )
        self.quality.blockSignals(False)

        has_bitrate, has_quality = "bitrate" in spec.modes, "quality" in spec.modes
        self.mode_bitrate.setVisible(has_bitrate and has_quality)
        self.mode_quality.setVisible(has_bitrate and has_quality)
        self.mode_bitrate.setChecked(True)
        self.lossless_label.setVisible(spec.lossless)

        while self.advanced_form.rowCount():
            self.advanced_form.removeRow(0)
        self._option_boxes = {}
        for option in spec.options:
            box = QComboBox()
            for value, label in option.choices:
                box.addItem(localize(label), value)
            box.setCurrentIndex([c[0] for c in option.choices].index(option.default))
            box.currentIndexChanged.connect(lambda _: self._refresh())
            self.advanced_form.addRow(localize(option.label), box)
            self._option_boxes[option.key] = box
        self.advanced_toggle.setVisible(bool(spec.options))
        if not spec.options:
            self.advanced.hide()
        self.note.setText(localize(spec.note) if spec.note else "")
        self.note.setVisible(bool(spec.note))
        self._on_mode()

    def _on_mode(self) -> None:
        mode = self._mode()
        self.bitrate.setVisible(mode == "bitrate")
        quality = mode == "quality"
        self.quality.setVisible(quality)
        self.quality_hint.setVisible(quality)
        self._refresh()

    def _toggle_advanced(self) -> None:
        shown = not self.advanced.isVisible()
        self.advanced.setVisible(shown)
        arrow = "▾" if shown else "▸"
        self.advanced_toggle.setText(f"{tr('encode.advanced')}  {arrow}")

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("encode.output"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _draft(self, output: Path) -> tuple[jobs.EncodeJob, tuple[str, ...]]:
        """Kodlama isi, kodlayicinin sinirlarina gore hazirlanmis.

        Kodlayici kaynagi kodlayamiyorsa `EncodeUnsupportedError` (D14).
        """
        info = self.source.info
        assert info is not None
        choice = self.choice()
        spec = choice.spec
        index = self.source.stream_index
        stream = info.stream(index)
        job = jobs.EncodeJob(
            source=info.path,
            output=output,
            codec=spec.encoder,
            bitrate_kbps=choice.bitrate_kbps,
            stream_index=index,
            source_rate=stream.sample_rate,
            extra=matrix.extra_args(choice),
        )
        return jobs.prepare(
            job,
            jobs.encoder_limits(self.tools.ffmpeg, spec.encoder),
            channels=stream.channels,
            channel_layout=stream.channel_layout,
        )

    def _refresh(self) -> None:
        messages = [localize(w) for w in matrix.warnings(self.choice())]
        blocked = False
        if self.source.info is not None:
            try:
                _, notes = self._draft(Path("draft"))
                messages += [localize(n) for n in notes]
            except jobs.EncodeUnsupportedError as exc:
                messages.append(localize(exc.args[0]))
                blocked = True
        self.warning.setText("\n".join(messages))
        self.warning.setVisible(bool(messages))
        planned = self.planned_output()
        self.preview.setText(f"→ {planned}" if planned is not None else "")
        busy = self.runner.busy or self.sweep_runner.busy
        self.start_button.setEnabled(not busy and planned is not None and not blocked)
        available = self.codec.model().item(self.codec.currentIndex()).isEnabled()
        self.sweep_button.setEnabled(
            not busy
            and self.source.info is not None
            and available
            and not blocked
            and bool(sweep.bitrates_for(self.spec))
        )

    def _build_sweep_card(self) -> QFrame:
        self.sweep_card, box = _card(tr("sweep.title"))
        self.sweep_chart = SweepChart()
        box.addWidget(self.sweep_chart)
        self.sweep_table = QTableWidget(0, 3)
        self.sweep_table.setHorizontalHeaderLabels(
            [tr("sweep.col.bitrate"), tr("summary.snr"), "NMR p95"]
        )
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.sweep_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.sweep_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sweep_table.setMaximumHeight(190)
        box.addWidget(self.sweep_table)
        note = QLabel(tr("disclaimer.audibility"))
        note.setObjectName("Muted")
        note.setWordWrap(True)
        box.addWidget(note)
        self.sweep_abx = QPushButton(tr("action.abx"))
        self.sweep_abx.clicked.connect(self._sweep_to_abx)
        box.addWidget(self.sweep_abx, 0, Qt.AlignmentFlag.AlignLeft)
        self.sweep_card.hide()
        return self.sweep_card

    # -- bit hizi taramasi -------------------------------------------------------------------

    def start_sweep(self) -> None:
        info = self.source.info
        spec = self.spec
        if info is None or self.runner.busy or self.sweep_runner.busy:
            return
        rates = sweep.bitrates_for(spec)
        if not rates:
            return
        reference = Track(path=info.path, info=info, stream_index=self.source.stream_index)
        tools = self.tools
        workdir = self.settings.temp_dir() / "sweep"

        def work(token: CancelToken, stage: Callable[[str], None]) -> sweep.Sweep:
            with ffmpeg_slot():
                return sweep.run(
                    tools.ffmpeg, tools.ffprobe, reference, spec, workdir, cancel=token, stage=stage
                )

        self._sweep_reference = reference
        self.progress.setValue(0)
        self.sweep_runner.start(work)

    def _on_sweep_stage(self, key: str) -> None:
        if key.startswith("rung:"):
            self.stage.setText(tr("sweep.rung", kbps=key[5:]))
        else:
            self.stage.setText(tr(f"stage.{key}"))

    def _on_sweep_done(self, result: object, seconds: float) -> None:
        assert isinstance(result, sweep.Sweep)
        self.sweep = result
        points = result.points
        self.sweep_chart.show_points(
            [p.bitrate_kbps for p in points], [p.result.headline_snr_db for p in points]
        )
        self.sweep_table.setRowCount(len(points))
        for row, point in enumerate(points):
            nmr = point.result.nmr.p95_db if point.result.nmr is not None else float("nan")
            for column, text in enumerate(
                (f"{point.bitrate_kbps} kbps", db_text(point.result.headline_snr_db), db_text(nmr))
            ):
                self.sweep_table.setItem(row, column, QTableWidgetItem(text))
        if points:
            self.sweep_table.selectRow(len(points) // 2)
        self.sweep_card.show()
        self.stage.setText(tr("sweep.done", seconds=seconds))
        self._refresh()

    def _sweep_to_abx(self) -> None:
        if self.sweep is None or self._sweep_reference is None:
            return
        rows = self.sweep_table.selectionModel().selectedRows()
        row = rows[0].row() if rows else 0
        point = self.sweep.points[row]
        if point.result.status != "measured" or not point.path.exists():
            QMessageBox.warning(self, tr("error.encode"), tr("sweep.no_abx"))
            return
        self.abx_requested.emit(
            point.result, self._sweep_reference, open_track(self.tools.ffprobe, point.path)
        )

    def _cancel(self) -> None:
        self.runner.cancel()
        self.sweep_runner.cancel()

    def _on_busy(self, busy: bool) -> None:
        # Iki is (kodlama, tarama) ayni denetimleri kilitler.
        busy = busy or self.runner.busy or self.sweep_runner.busy
        self.progress.setVisible(busy)
        self.cancel_button.setVisible(busy)
        # Kodlama surerken TUM ayarlar kilitli: once bit hizi/kalite/gelismis
        # secenekler acik kaliyor, degistirilince onizleme isle uyusmuyordu (D25).
        for widget in (
            self.source,
            self.codec,
            self.mode_bitrate,
            self.mode_quality,
            self.bitrate,
            self.quality,
            self.advanced_toggle,
            self.advanced,
            self.folder,
            self.browse,
            self.compare_after,
        ):
            widget.setEnabled(not busy)
        self._refresh()

    def _on_stage(self, key: str) -> None:
        if key.startswith(_PROGRESS_PREFIX):
            fraction = float(key[len(_PROGRESS_PREFIX) :])
            self.progress.setValue(round(fraction * 1000))
            self.stage.setText(tr("stage.encode_pct", pct=100 * fraction))

    # -- is --------------------------------------------------------------------------------------

    def start(self) -> None:
        info = self.source.info
        if info is None or self.runner.busy:
            return
        choice = self.choice()
        spec = choice.spec
        stream_index = self.source.stream_index
        output = naming.output_path(
            info.path,
            matrix.describe(choice),
            spec.extension,
            directory=self.output_folder(),
            suffix=self.settings.output_suffix,
            taken=self._taken,
        )
        try:
            job, _ = self._draft(output)
        except jobs.EncodeUnsupportedError as exc:
            QMessageBox.critical(self, tr("error.encode"), localize(exc.args[0]))
            self._taken.discard(output)
            return
        duration = info.stream(stream_index).duration or info.duration
        self._pending = Encoded(info.path, stream_index, output, self.compare_after.isChecked())
        ffmpeg = self.tools.ffmpeg

        def work(token: CancelToken, stage: Callable[[str], None]) -> Path:
            # Toplu taramayla ayni ffmpeg semaforu: ikisi birlikte calisirsa
            # surec sayisi sinirli kalir.
            with ffmpeg_slot():
                return jobs.run_with_progress(
                    ffmpeg,
                    job,
                    duration=duration,
                    on_progress=lambda f: stage(f"{_PROGRESS_PREFIX}{f:.4f}"),
                    cancel=token,
                )

        self.progress.setValue(0)
        self.stage.setText(tr("stage.encode_pct", pct=0.0))
        self.runner.start(work)

    def _remember_folder(self) -> None:
        folder = self.folder.text().strip()
        if folder != self.settings.output_dir:
            self.settings = Settings(**{**self.settings.__dict__, "output_dir": folder}).clamped()
            settings_mod.save(self.settings)
            self.settings_changed.emit(self.settings)

    def _on_done(self, output: object, seconds: float) -> None:
        assert isinstance(output, Path)
        self.stage.setText(tr("encode.done", name=output.name, seconds=seconds))
        pending, self._pending = self._pending, None
        # Klasor yalnizca basarili bir kodlamadan sonra hatirlanir (D22).
        self._remember_folder()
        self._refresh()
        if pending is not None:
            self.encoded.emit(pending)

    def _release(self) -> None:
        """Basarisiz/iptal edilen kodlamanin ad rezervasyonunu birakir (D22).

        Rezerve kalan ad bir sonraki denemeyi gereksiz yere `_2` yapiyordu.
        """
        if self._pending is not None:
            self._taken.discard(self._pending.output)
        self._pending = None
        self._refresh()

    def _on_failed(self, message: str) -> None:
        self._release()
        self.stage.setText("")
        QMessageBox.critical(self, tr("error.encode"), message)

    def _on_cancelled(self) -> None:
        self._release()
        self.stage.setText(tr("stage.cancelled"))
