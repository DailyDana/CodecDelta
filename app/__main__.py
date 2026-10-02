"""`python -m app [referans] [test]` -- masaustu uygulamasini baslatir.

`--smoke-test <dosya.json>`: pencereyi kurar, durumu (surum, ffmpeg, sekmeler)
JSON olarak yazar ve olay dongusune girmeden cikar. Paketlenmis exe'nin
(konsolu olmayan) calistigini dogrulamak icin; `tools/build.ps1` kullanir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    # Baslatici bos argumanlari da gecirir (dosya surulmeden cift tik).
    args = [a for a in args if a.strip()]
    smoke: Path | None = None
    if "--smoke-test" in args:
        at = args.index("--smoke-test")
        smoke = Path(args[at + 1]) if at + 1 < len(args) else Path("smoke.json")
        args = args[:at] + args[at + 2 :]

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
    # Paketlenmis hal: exe'nin yanindaki "bin" klasoru de ffmpeg adayi.
    app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else None
    # Sablon ve ses denetimi ffmpeg'den bagimsiz: ffmpeg'siz makinede de kosar.
    checks = _smoke_checks() if smoke is not None else {}
    try:
        tools = discover(
            explicit=Path(settings.ffmpeg_dir) if settings.ffmpeg_dir else None,
            app_dir=app_dir,
            cache_file=settings_mod.caps_file(),
        )
    except CodecDeltaError as exc:
        if smoke is not None:
            _write_smoke(smoke, ffmpeg=None, tabs=0, error=exc.user_message(), **checks)
            return 2
        QMessageBox.critical(
            None, tr("error.title"), f"{tr('error.ffmpeg')}\n\n{exc.user_message()}"
        )
        return 1

    window = MainWindow(tools, settings)
    if smoke is not None:
        _write_smoke(
            smoke,
            ffmpeg=str(tools.directory),
            tabs=window.tabs.count(),
            error=None,
            **checks,
        )
        window.close()
        return 0
    paths = [Path(a) for a in args[:2]]
    window.analyze.load(paths[0] if paths else None, paths[1] if len(paths) > 1 else None)
    window.show()
    return app.exec()


def _smoke_checks() -> dict[str, object]:
    """Paketlemede en kolay kopan iki parca: rapor sablonu ve Qt ses ciktisi."""
    from PyQt6.QtMultimedia import QAudioSink, QMediaDevices

    from app.report import html

    try:
        html.self_check()
        report: object = True
    except Exception as exc:  # duman testi her hatayi raporlar
        report = f"{type(exc).__name__}: {exc}"
    device = QMediaDevices.defaultAudioOutput()
    audio: object = None
    if not device.isNull():
        # Cihaz olmayan makinede (CI) null kalir. Varsa sink push modunda
        # acilip kapatilir: veri yazilmaz, ses cikmaz ama arka uc yuklenir.
        sink = QAudioSink(device, device.preferredFormat())
        opened = sink.start() is not None
        audio = {"device": device.description(), "opened": opened, "error": sink.error().name}
        sink.stop()
    return {"report": report, "audio": audio}


def _write_smoke(
    path: Path, *, ffmpeg: str | None, tabs: int, error: str | None, **extra: object
) -> None:
    from app import __version__

    data = {"version": __version__, "ffmpeg": ffmpeg, "tabs": tabs, "error": error, **extra}
    path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
