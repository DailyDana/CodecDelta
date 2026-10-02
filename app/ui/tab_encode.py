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
    QVBoxLayout,
    QWidget,
)

from app.core import settings as settings_mod
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.core.settings import Settings
from app.encode import jobs, matrix, naming
from app.encode.matrix import CodecSpec, EncodeSettings, Mode
from app.ui.i18n import localize, tr
from app.ui.theme import COLORS
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
        browse = QPushButton(tr("encode.browse"))
        browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
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
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.cancel_button.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.stage = QLabel()
        self.stage.setObjectName("Stage")
        actions = QHBoxLayout()
        actions.addWidget(self.start_button)
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
        layout.addStretch()

        self.runner.stage.connect(self._on_stage)
        self.runner.succeeded.connect(self._on_done)
        self.runner.failed.connect(self._on_failed)
        self.runner.cancelled.connect(self._on_cancelled)
        self.runner.busy_changed.connect(self._on_busy)
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
        self.start_button.setEnabled(not self.runner.busy and planned is not None and not blocked)

    def _on_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.cancel_button.setVisible(busy)
        for widget in (self.source, self.codec, self.folder, self.compare_after):
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
            return
        self._remember_folder()
        duration = info.stream(stream_index).duration or info.duration
        self._pending = Encoded(info.path, stream_index, output, self.compare_after.isChecked())
        ffmpeg = self.tools.ffmpeg

        def work(token: CancelToken, stage: Callable[[str], None]) -> Path:
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

    def _on_done(self, output: object, seconds: float) -> None:
        assert isinstance(output, Path)
        self.stage.setText(tr("encode.done", name=output.name, seconds=seconds))
        pending, self._pending = self._pending, None
        self._refresh()
        if pending is not None:
            self.encoded.emit(pending)

    def _on_failed(self, message: str) -> None:
        self._pending = None
        self.stage.setText("")
        QMessageBox.critical(self, tr("error.encode"), message)

    def _on_cancelled(self) -> None:
        self._pending = None
        self.stage.setText(tr("stage.cancelled"))
