"""Toplu kutuphane taramasi sekmesi.

Tarama arka planda (`Runner`) calisir; sonuclar is parcacigindan bir kuyruga
yazilir ve arayuz onlari 200 ms'de bir TOPLU olarak tabloya ekler. Her sonuc
icin ayri bir sinyal 1000 dosyalik taramada arayuzu sundururdu.

Cift tiklanan dosya Analiz sekmesinde tek dosya dogrulamasiyla acilir.
"""

from __future__ import annotations

import csv
import queue
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.batch import scanner
from app.batch.cache import ScanCache
from app.batch.model import ScanEntry
from app.core import settings as settings_mod
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.ui.i18n import localize, tr
from app.ui.theme import TONE_COLORS
from app.ui.worker import Runner

_TONES = {
    "consistent_lossy": "bad",
    "undetermined": "warn",
    "consistent_lossless": "good",
    "not_applicable": "neutral",
    "error": "warn",
}
_FILTERS = ("all", "suspicious", "undetermined", "errors", "lossless")
_COLUMNS = ("file", "verdict", "cutoff", "reason")


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(8)
    return frame, layout


class _NumberItem(QTableWidgetItem):
    """Bir ondalikla gosterilen ama sayi olarak siralanan hucre (bos = en sonda)."""

    def __init__(self, value: float | None) -> None:
        super().__init__(f"{value:.1f}" if value is not None else "")
        self.value = value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        mine = self.value if self.value is not None else float("inf")
        theirs = getattr(other, "value", None)
        return mine < (theirs if theirs is not None else float("inf"))


def cache_file() -> Path:
    return settings_mod.config_dir() / "scan_cache.json"


def _visible(entry: ScanEntry, choice: str) -> bool:
    return {
        "all": True,
        "suspicious": entry.bucket == "consistent_lossy",
        "undetermined": entry.bucket == "undetermined",
        "errors": entry.bucket == "error",
        "lossless": entry.bucket == "consistent_lossless",
    }[choice]


class BatchTab(QWidget):
    # Kullanici bir dosyayi tek dosya dogrulamasinda acmak istiyor.
    verify_requested = pyqtSignal(object)

    def __init__(
        self,
        tools: FFmpegTools,
        parent: QWidget | None = None,
        *,
        cache_path: Callable[[], Path] = cache_file,
    ) -> None:
        super().__init__(parent)
        self.tools = tools
        self._cache_path = cache_path
        self.runner = Runner(self)
        self.entries: list[ScanEntry] = []
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._total = 0
        self._started = 0.0
        self._root: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        top, box = _card()
        hint = QLabel(tr("batch.intro"))
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        box.addWidget(hint)
        row = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setPlaceholderText(tr("batch.folder"))
        browse = QPushButton(tr("encode.browse"))
        browse.clicked.connect(self._browse)
        self.scan_button = QPushButton(tr("batch.scan"))
        self.scan_button.setObjectName("Primary")
        self.cancel_button = QPushButton(tr("action.cancel"))
        self.cancel_button.hide()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        row.addWidget(self.scan_button)
        row.addWidget(self.cancel_button)
        box.addLayout(row)
        self.rescan = QCheckBox(tr("batch.rescan"))
        box.addWidget(self.rescan)
        self.progress = QProgressBar()
        self.progress.hide()
        box.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName("Stage")
        box.addWidget(self.status)
        layout.addWidget(top)

        results, rbox = _card()
        bar = QHBoxLayout()
        self.filter = QComboBox()
        for key in _FILTERS:
            self.filter.addItem(tr(f"batch.filter.{key}"), key)
        self.export_button = QPushButton(tr("batch.export"))
        bar.addWidget(self.filter)
        bar.addStretch(1)
        bar.addWidget(self.export_button)
        rbox.addLayout(bar)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([tr(f"batch.col.{c}") for c in _COLUMNS])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 360)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 110)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        rbox.addWidget(self.table, 1)
        double = QLabel(tr("batch.double_click"))
        double.setObjectName("Muted")
        rbox.addWidget(double)
        layout.addWidget(results, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._drain)

        self.scan_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.filter.currentIndexChanged.connect(lambda _: self._rebuild())
        self.export_button.clicked.connect(self.export)
        self.table.cellDoubleClicked.connect(self._open)
        self.folder.textChanged.connect(lambda _: self._refresh())
        self.runner.succeeded.connect(lambda result, seconds: self._finished(False))
        self.runner.cancelled.connect(lambda: self._finished(True))
        self.runner.failed.connect(self._on_failed)
        self.runner.busy_changed.connect(lambda _: self._refresh())
        self._refresh()

    # -- durum ----------------------------------------------------------------

    def _refresh(self) -> None:
        busy = self.runner.busy
        self.scan_button.setVisible(not busy)
        self.cancel_button.setVisible(busy)
        self.scan_button.setEnabled(bool(self.folder.text().strip()))
        self.folder.setEnabled(not busy)
        self.rescan.setEnabled(not busy)
        self.export_button.setEnabled(bool(self.entries) and not busy)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("batch.folder"), self.folder.text())
        if folder:
            self.folder.setText(folder)

    # -- tarama ---------------------------------------------------------------

    def start(self) -> None:
        text = self.folder.text().strip()
        if not text or self.runner.busy:
            return
        root = Path(text)
        if not root.is_dir():
            QMessageBox.warning(self, tr("tab.batch"), tr("batch.no_folder"))
            return
        self._root = root
        self.entries = []
        self.table.setRowCount(0)
        self._total = 0
        self._started = time.perf_counter()
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status.setText(tr("batch.discovering"))
        tools, rescan, out = self.tools, self.rescan.isChecked(), self._queue
        cache_path = self._cache_path()

        def job(token: CancelToken, stage: Callable[[str], None]) -> int:
            files = scanner.discover(root)
            out.put(("total", len(files)))
            run = scanner.Scanner(tools.ffmpeg, tools.ffprobe, ScanCache(cache_path, scanner.RULES))
            return run.run(
                files,
                on_result=lambda entry, cached: out.put(("entry", entry)),
                cancel=token,
                rescan=rescan,
            )

        self.runner.start(job)
        self._timer.start()
        self._refresh()

    def _drain(self) -> None:
        added: list[ScanEntry] = []
        while True:
            try:
                kind, value = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "total":
                assert isinstance(value, int)
                self._total = value
                self.progress.setRange(0, max(1, value))
            else:
                assert isinstance(value, ScanEntry)
                added.append(value)
        if added:
            self.entries.extend(added)
            self.table.setSortingEnabled(False)
            choice = self.filter.currentData()
            for entry in added:
                if _visible(entry, choice):
                    self._append(entry)
            self.table.setSortingEnabled(True)
        if self._total:
            self.progress.setValue(len(self.entries))
            self.status.setText(self._summary())

    def _summary(self) -> str:
        counts = Counter(e.bucket for e in self.entries)
        done = len(self.entries)
        elapsed = time.perf_counter() - self._started
        text = tr(
            "batch.status",
            done=done,
            total=self._total,
            lossy=counts["consistent_lossy"],
            undetermined=counts["undetermined"],
            errors=counts["error"],
        )
        if 0 < done < self._total and elapsed > 2:
            remaining = elapsed / done * (self._total - done)
            text += "  ·  " + tr("batch.eta", seconds=remaining)
        return text

    def _finished(self, cancelled: bool) -> None:
        self._drain()
        self._timer.stop()
        self.progress.hide()
        key = "batch.cancelled" if cancelled else "batch.done"
        self.status.setText(
            self._summary() + "  ·  " + tr(key, seconds=time.perf_counter() - self._started)
        )
        self._refresh()

    def _on_failed(self, message: str) -> None:
        self._timer.stop()
        self.progress.hide()
        self.status.setText("")
        QMessageBox.critical(self, tr("tab.batch"), message)
        self._refresh()

    # -- tablo ----------------------------------------------------------------

    def restore(self, folder: str, root: Path | None, entries: list[ScanEntry]) -> None:
        """Dil degisiminde arayuz yeniden kurulurken sonuclari geri yukler."""
        self.folder.setText(folder)
        self._root = root
        self.entries = list(entries)
        self._total = len(entries)
        self._rebuild()
        if entries:
            self.status.setText(self._summary())
        self._refresh()

    def _name(self, entry: ScanEntry) -> str:
        if self._root is not None:
            try:
                return entry.path.relative_to(self._root).as_posix()
            except ValueError:
                pass
        return entry.path.name

    def _append(self, entry: ScanEntry) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(self._name(entry))
        name.setData(Qt.ItemDataRole.UserRole, str(entry.path))
        name.setToolTip(str(entry.path))
        label = tr("batch.error") if entry.bucket == "error" else tr(f"single.{entry.bucket}")
        verdict = QTableWidgetItem(label)
        verdict.setForeground(QColor(TONE_COLORS[_TONES[entry.bucket]]))
        cutoff = _NumberItem(entry.cutoff_hz / 1000 if entry.cutoff_hz else None)
        reason = QTableWidgetItem(localize(entry.headline) if entry.headline else "")
        for column, item in enumerate((name, verdict, cutoff, reason)):
            self.table.setItem(row, column, item)

    def _rebuild(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        choice = self.filter.currentData()
        for entry in self.entries:
            if _visible(entry, choice):
                self._append(entry)
        self.table.setSortingEnabled(True)

    def _open(self, row: int, column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self.verify_requested.emit(Path(item.data(Qt.ItemDataRole.UserRole)))

    def export(self) -> None:
        if not self.entries:
            return
        suggested = (self._root or Path.home()) / "codecdelta_scan.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, tr("batch.export"), str(suggested), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self.write_csv(Path(path))
        except OSError as exc:
            QMessageBox.critical(self, tr("tab.batch"), str(exc))
            return
        self.status.setText(tr("batch.exported", name=Path(path).name))

    def write_csv(self, path: Path) -> None:
        """Tum sonuclar; Excel Turkce karakterleri dogru acsin diye BOM'lu UTF-8."""
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "verdict", "cutoff_khz", "stage", "reason", "seconds"])
            for e in self.entries:
                writer.writerow(
                    [
                        str(e.path),
                        e.bucket,
                        f"{e.cutoff_hz / 1000:.2f}" if e.cutoff_hz else "",
                        e.stage,
                        localize(e.headline) if e.headline else "",
                        f"{e.seconds:.2f}",
                    ]
                )
