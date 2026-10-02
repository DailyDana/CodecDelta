"""Ana pencere: sekmeler, dil menusu, durum cubugunda ffmpeg bilgisi."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QAction, QActionGroup, QCloseEvent
from PyQt6.QtWidgets import QLabel, QMainWindow, QTabWidget, QWidget

from app import __version__
from app.compare.pipeline import Track
from app.compare.result import ComparisonResult
from app.core import settings as settings_mod
from app.core.ffmpeg_locate import FFmpegTools
from app.core.settings import Settings
from app.ui.i18n import LANGUAGES, set_language, tr
from app.ui.tab_abx import AbxPair, AbxTab
from app.ui.tab_analyze import AnalyzeTab
from app.ui.tab_batch import BatchTab
from app.ui.tab_encode import Encoded, EncodeTab

_LANGUAGE_NAMES = {"en": "English", "tr": "Türkçe"}


class MainWindow(QMainWindow):
    def __init__(
        self, tools: FFmpegTools, settings: Settings, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.tools = tools
        self.settings = settings
        self.resize(1180, 860)
        self.setMinimumSize(900, 640)
        self._build()

        language_menu = self.menuBar().addMenu("Language / Dil")
        group = QActionGroup(self)
        self._language_actions: dict[str, QAction] = {}
        for code in LANGUAGES:
            action = QAction(_LANGUAGE_NAMES.get(code, code), self)
            action.setCheckable(True)
            action.setChecked(code == settings.language)
            action.triggered.connect(lambda _=False, c=code: self.set_language(c))
            group.addAction(action)
            language_menu.addAction(action)
            self._language_actions[code] = action

        version = tools.caps.version.split(" ")[0] if tools.caps.version else "?"
        info = QLabel(f"ffmpeg {version}  ·  {tools.directory}")
        info.setObjectName("Muted")
        self.statusBar().addPermanentWidget(info)

    def _build(self) -> None:
        set_language(self.settings.language)
        self.setWindowTitle(f"{tr('app.title')} {__version__}")
        self.tabs = QTabWidget()
        self.analyze = AnalyzeTab(self.tools, self.settings)
        self.encode = EncodeTab(self.tools, self.settings)
        self.encode.encoded.connect(self._on_encoded)
        self.encode.settings_changed.connect(self._on_settings)
        # Kodlama bittiginde Analiz mesgulse karsilastirma kuyruga alinir (D27).
        self._queued: Encoded | None = None
        self.analyze.runner.busy_changed.connect(self._run_queued)
        self.abx = AbxTab(self.tools)
        self.analyze.abx_requested.connect(self._on_abx_requested)
        self.batch = BatchTab(self.tools)
        self.batch.verify_requested.connect(self._on_verify_requested)
        self.tabs.addTab(self.analyze, tr("tab.analyze"))
        self.tabs.addTab(self.encode, tr("tab.encode"))
        self.tabs.addTab(self.abx, tr("tab.abx"))
        self.tabs.addTab(self.batch, tr("tab.batch"))
        self.setCentralWidget(self.tabs)

    def _on_verify_requested(self, path: object) -> None:
        """Taramadan secilen dosyayi Analiz'de tek dosya dogrulamasiyla acar."""
        assert isinstance(path, Path)
        if self.analyze.runner.busy:
            return
        self.analyze.reference.set_path(path)
        self.tabs.setCurrentWidget(self.analyze)
        self.analyze.start_verify()

    def _on_abx_requested(self, result: object, reference: object, test: object) -> None:
        """Analiz'deki karsilastirmayi ABX sekmesine verir ve oraya gecer."""
        assert isinstance(result, ComparisonResult)
        assert isinstance(reference, Track) and isinstance(test, Track)
        self.abx.set_pair(AbxPair(result, reference, test))
        self.tabs.setCurrentWidget(self.abx)

    def _on_settings(self, settings: Settings) -> None:
        """Bir sekme ayar kaydetti: digerinin kopyasi bayat kalmasin (D21).

        Analiz, raporun onerilen klasorunu kendi ayar kopyasindan okuyordu ve
        Kodlama'da secilen klasor oraya hic yansimiyordu.
        """
        self.settings = settings
        self.analyze.settings = settings

    def _run_queued(self, busy: bool) -> None:
        if not busy and self._queued is not None:
            queued, self._queued = self._queued, None
            self._on_encoded(queued)

    def _on_encoded(self, done: Encoded) -> None:
        """Kodlama bitti: istenmisse Analiz'e gec, kaynak/cikti ile karsilastir."""
        self._on_settings(self.encode.settings)
        if not done.compare:
            return
        if self.analyze.runner.busy:
            # Once sessizce atlaniyordu (D27): Analiz'deki is bitince baslar.
            self._queued = done
            self.analyze.stage.setText(tr("encode.compare_queued", name=done.output.name))
            return
        self.analyze.load(done.source, done.output)
        self.analyze.reference.select_stream(done.stream_index)
        self.tabs.setCurrentWidget(self.analyze)
        self.analyze.start_compare()

    def set_language(self, code: str) -> None:
        """Dili degistirir ve arayuzu yeniden kurar. Yuklu dosyalar korunur."""
        abx_active = self.abx.session is not None and not self.abx.session.finished
        if (
            code == self.settings.language
            or self.analyze.runner.busy
            or self.encode.runner.busy
            or self.abx.runner.busy
            or self.batch.runner.busy
            or abx_active
        ):
            # Reddedildi: menu tiklananin isaretli kalmasin, gecerli dil isaretli
            # olsun (denetim D20).
            current = self._language_actions.get(self.settings.language)
            if current is not None:
                current.setChecked(True)
            return
        analyze = self.analyze.state()
        abx_pair = self.abx.pair
        batch = (self.batch.folder.text(), self.batch._root, self.batch.entries)
        source, source_stream = self.encode.source.path, self.encode.source.stream_index
        choice, folder = self.encode.choice(), self.encode.folder.text()
        compare_after = self.encode.compare_after.isChecked()
        current_tab = self.tabs.currentIndex()
        self.settings = self.encode.settings
        self.settings = Settings(**{**self.settings.__dict__, "language": code}).clamped()
        settings_mod.save(self.settings)
        self._build()
        self.analyze.restore(analyze)
        self.abx.set_pair(abx_pair)
        self.batch.restore(*batch)
        if source is not None:
            self.encode.source.set_path(source)
            self.encode.source.select_stream(source_stream)
        self.encode.apply_choice(choice)
        self.encode.folder.setText(folder)
        self.encode.compare_after.setChecked(compare_after)
        self.tabs.setCurrentIndex(current_tab)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.abx.player.release()
        for runner in (self.analyze.runner, self.encode.runner, self.abx.runner, self.batch.runner):
            if runner.busy:
                runner.cancel()
                runner.wait(10000)
        super().closeEvent(event)
