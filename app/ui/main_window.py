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
            # Reddedildi: menu tiklananin isaretli kalmasin, gecerli dil isaretli
            # olsun (denetim D20).
            current = self._language_actions.get(self.settings.language)
            if current is not None:
                current.setChecked(True)
            return
        analyze = self.analyze.state()
        source, source_stream = self.encode.source.path, self.encode.source.stream_index
        choice, folder = self.encode.choice(), self.encode.folder.text()
        compare_after = self.encode.compare_after.isChecked()
        current_tab = self.tabs.currentIndex()
        self.settings = self.encode.settings
        self.settings = Settings(**{**self.settings.__dict__, "language": code}).clamped()
        settings_mod.save(self.settings)
        self._build()
        self.analyze.restore(analyze)
        if source is not None:
            self.encode.source.set_path(source)
            self.encode.source.select_stream(source_stream)
        self.encode.apply_choice(choice)
        self.encode.folder.setText(folder)
        self.encode.compare_after.setChecked(compare_after)
        self.tabs.setCurrentIndex(current_tab)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        for runner in (self.analyze.runner, self.encode.runner):
            if runner.busy:
                runner.cancel()
                runner.wait(10000)
        super().closeEvent(event)
