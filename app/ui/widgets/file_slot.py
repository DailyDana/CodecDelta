"""Surukle-birak dosya alani: dosya adi, bicim ozeti ve iz secici.

Dosya birakildiginda ffprobe calistirilir (konteyner basliklari; milisaniyeler).
Birden fazla ses izi varsa (film, konser MKV'si) iz secici gorunur; varsayilan
iz `default_stream` kuraliyla secilir.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.errors import CodecDeltaError
from app.core.probe import Probe, default_stream, probe
from app.ui.i18n import localize, tr

MEDIA_FILTER = (
    "Media (*.flac *.wav *.aif *.aiff *.alac *.m4a *.mp4 *.mp3 *.opus *.ogg *.oga *.webm "
    "*.mka *.mkv *.wv *.ape *.tak *.tta *.aac *.ac3 *.eac3 *.dts *.wma *.mpc *.mov *.ts);;All (*)"
)


def _format_label(info: Probe, index: int) -> str:
    stream = info.stream(index)
    parts = [stream.codec, f"{stream.sample_rate / 1000:g} kHz"]
    if stream.bits_per_raw_sample:
        parts.append(f"{stream.bits_per_raw_sample}-bit")
    parts.append(f"{stream.channels} ch")
    if info.duration:
        minutes, seconds = divmod(round(info.duration), 60)
        parts.append(f"{minutes}:{seconds:02d}")
    return "  ·  ".join(parts)


class FileSlot(QFrame):
    changed = pyqtSignal()
    # Birden fazla dosya birakildiginda ilkinden sonrakiler (D26): yuvayi
    # barindiran sekme bunlari baska bir yuvaya verebilir.
    dropped_more = pyqtSignal(list)

    def __init__(self, title: str, ffprobe: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DropSlot")
        self.setAcceptDrops(True)
        self.setMinimumHeight(118)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ffprobe = ffprobe
        self.info: Probe | None = None

        self._title = QLabel(title.upper())
        self._title.setObjectName("SlotTitle")
        self._clear = QPushButton(tr("slot.clear"))
        self._clear.setObjectName("Link")
        self._clear.clicked.connect(self.clear)
        self._clear.hide()
        top = QHBoxLayout()
        top.addWidget(self._title)
        top.addStretch()
        top.addWidget(self._clear)

        self._name = QLabel(tr("slot.hint"))
        self._name.setObjectName("SlotInfo")
        self._name.setWordWrap(True)
        self._info = QLabel()
        self._info.setObjectName("SlotInfo")
        self._tracks = QComboBox()
        self._tracks.currentIndexChanged.connect(lambda _: self._on_track())
        self._tracks.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.addLayout(top)
        layout.addWidget(self._name)
        layout.addWidget(self._info)
        layout.addWidget(self._tracks)
        layout.addStretch()

    # -- durum -----------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        return self.info.path if self.info is not None else None

    @property
    def stream_index(self) -> int:
        data = self._tracks.currentData()
        return int(data) if data is not None else 0

    def set_path(self, path: Path) -> None:
        try:
            info = probe(self._ffprobe, path)
        except CodecDeltaError as exc:
            self.info = None
            self._name.setText(path.name)
            self._name.setObjectName("SlotName")
            self._info.setText(localize(exc.args[0]) if exc.args else exc.user_message())
            # Onceki dosyanin iz listesi kalmasin (D23).
            self._tracks.clear()
            self._tracks.hide()
            self._refresh()
            self.changed.emit()
            return
        self.info = info
        self._name.setText(path.name)
        self._name.setToolTip(str(path))
        self._tracks.blockSignals(True)
        self._tracks.clear()
        for stream in info.audio:
            self._tracks.addItem(f"{tr('slot.track')} {stream.label()}", stream.audio_index)
        self._tracks.setCurrentIndex(default_stream(info))
        self._tracks.setVisible(len(info.audio) > 1)
        self._tracks.blockSignals(False)
        self._on_track(emit=False)
        self._refresh()
        self.changed.emit()

    def select_stream(self, audio_index: int) -> None:
        """Belirli bir ses izini secer (orn. kodlanan iz Analiz'e tasinirken)."""
        position = self._tracks.findData(audio_index)
        if position >= 0:
            self._tracks.setCurrentIndex(position)

    def clear(self) -> None:
        self.info = None
        self._name.setText(tr("slot.hint"))
        self._name.setToolTip("")
        self._info.setText("")
        self._tracks.clear()
        self._tracks.hide()
        self._refresh()
        self.changed.emit()

    def _on_track(self, *, emit: bool = True) -> None:
        if self.info is None:
            return
        text = _format_label(self.info, self.stream_index)
        if self.info.has_video:
            text += f"\n{tr('slot.video')}"
        self._info.setText(text)
        if emit:
            self.changed.emit()

    def _refresh(self) -> None:
        filled = self.info is not None
        self._name.setObjectName("SlotName" if filled else "SlotInfo")
        self.setProperty("filled", filled)
        self._clear.setVisible(filled)
        for widget in (self, self._name):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    # -- surukle-birak ------------------------------------------------------------

    def _set_hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData() is not None and event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, event: QDragLeaveEvent | None) -> None:
        self._set_hover(False)

    def dropEvent(self, event: QDropEvent | None) -> None:
        self._set_hover(False)
        if event is None or event.mimeData() is None:
            return
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.set_path(Path(urls[0].toLocalFile()))
            event.acceptProposedAction()
            if len(urls) > 1:
                self.dropped_more.emit([Path(u.toLocalFile()) for u in urls[1:]])

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, self._title.text().title(), "", MEDIA_FILTER
            )
            if path:
                self.set_path(Path(path))
