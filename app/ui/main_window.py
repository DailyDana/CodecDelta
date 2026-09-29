"""Ana pencere: sekmeler, dil menusu, durum cubugunda ffmpeg bilgisi."""

from __future__ import annotations

from PyQt6.QtGui import QAction, QActionGroup, QCloseEvent
from PyQt6.QtWidgets import QLabel, QMainWindow, QTabWidget, QWidget

from app import __version__
from app.core import settings as settings_mod
from app.core.ffmpeg_locate import FFmpegTools
from app.core.settings import Settings
from app.ui.i18n import LANGUAGES, set_language, tr
from app.ui.tab_analyze import AnalyzeTab
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
        for code in LANGUAGES:
            action = QAction(_LANGUAGE_NAMES.get(code, code), self)
            action.setCheckable(True)
            action.setChecked(code == settings.language)
            action.triggered.connect(lambda _=False, c=code: self.set_language(c))
            group.addAction(action)
            language_menu.addAction(action)

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
        self.tabs.addTab(self.analyze, tr("tab.analyze"))
        self.tabs.addTab(self.encode, tr("tab.encode"))
        self.setCentralWidget(self.tabs)

    def _on_encoded(self, done: Encoded) -> None:
        """Kodlama bitti: istenmisse Analiz'e gec, kaynak/cikti ile karsilastir."""
        self.settings = self.encode.settings
        if not done.compare or self.analyze.runner.busy:
            return
        self.analyze.load(done.source, done.output)
        self.analyze.reference.select_stream(done.stream_index)
        self.tabs.setCurrentWidget(self.analyze)
        self.analyze.start_compare()

    def set_language(self, code: str) -> None:
        """Dili degistirir ve arayuzu yeniden kurar. Yuklu dosyalar korunur."""
        if code == self.settings.language or self.analyze.runner.busy or self.encode.runner.busy:
            return
        reference, test = self.analyze.reference.path, self.analyze.test.path
        source = self.encode.source.path
        self.settings = self.encode.settings
        self.settings = Settings(**{**self.settings.__dict__, "language": code}).clamped()
        settings_mod.save(self.settings)
        self._build()
        self.analyze.load(reference, test)
        if source is not None:
            self.encode.source.set_path(source)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        for runner in (self.analyze.runner, self.encode.runner):
            if runner.busy:
                runner.cancel()
                runner.wait(10000)
        super().closeEvent(event)
