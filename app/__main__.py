"""`python -m app [referans] [test]` -- masaustu uygulamasini baslatir."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    from PyQt6.QtWidgets import QApplication, QMessageBox

    from app.core import settings as settings_mod
    from app.core.errors import CodecDeltaError
    from app.core.ffmpeg_locate import discover
    from app.ui.i18n import set_language, tr
    from app.ui.main_window import MainWindow
    from app.ui.theme import STYLESHEET

    app = QApplication(sys.argv[:1])
    app.setApplicationName("CodecDelta")
    app.setStyleSheet(STYLESHEET)
    settings = settings_mod.load()
    set_language(settings.language)
    try:
        tools = discover(
            explicit=Path(settings.ffmpeg_dir) if settings.ffmpeg_dir else None,
            cache_file=settings_mod.caps_file(),
        )
    except CodecDeltaError as exc:
        QMessageBox.critical(
            None, tr("error.title"), f"{tr('error.ffmpeg')}\n\n{exc.user_message()}"
        )
        return 1

    window = MainWindow(tools, settings)
    paths = [Path(a) for a in args[:2]]
    window.analyze.load(paths[0] if paths else None, paths[1] if len(paths) > 1 else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
