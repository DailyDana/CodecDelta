"""Paket iskeletinin ayakta oldugunu dogrulayan duman testi.

Asil degeri katman kuralini kilitlemesi: motor katmanlari Qt'ye bulasirsa
headless test ve CLI yolu kapanir, bu da geri donusu pahali bir hatadir.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import app

ENGINE_PACKAGES = [
    "core",
    "bitstream",
    "dsp",
    "align",
    "compare",
    "psycho",
    "single",
    "encode",
    "batch",
    "report",
    "abx",
]


def test_version_is_set() -> None:
    assert app.__version__


def test_engine_layers_import_without_qt() -> None:
    """Motor katmanlari Qt olmadan import edilebilmeli."""
    for name in ENGINE_PACKAGES:
        importlib.import_module(f"app.{name}")


def test_engine_layers_do_not_import_qt() -> None:
    """Motor katmanlarinin hicbir modulu PyQt6'ya bagimli olmamali.

    Ayri bir surecte: ayni surecte daha once calisan bir arayuz testi PyQt6'yi
    zaten yuklemis olur ve test, sirasina gore yanlis alarm verirdi.
    """
    import subprocess
    import sys

    script = "\n".join(
        [
            "import importlib, pkgutil, sys",
            f"for name in {ENGINE_PACKAGES!r}:",
            "    pkg = importlib.import_module(f'app.{name}')",
            "    for mod in pkgutil.iter_modules(pkg.__path__, prefix=f'app.{name}.'):",
            "        importlib.import_module(mod.name)",
            "print(sorted(m for m in sys.modules if m.startswith('PyQt6')))",
        ]
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]", f"motor katmani Qt import etti: {done.stdout}"


def test_smoke_test_flag(tmp_path: Path) -> None:
    """`--smoke-test` pencereyi acmadan durum JSON'u yazar (build.ps1 buna dayanir)."""
    import json
    import os
    import subprocess
    import sys

    out = tmp_path / "smoke.json"
    # Kullanicinin ayar ve onbellegine dokunmasin.
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "APPDATA": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-m", "app", "--smoke-test", str(out)],
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == app.__version__
    assert data["report"] is True
    if done.returncode == 0:
        assert data["tabs"] == 4
        assert data["error"] is None
    else:
        # ffmpeg'siz makine (CI): okunur hata, pencere kurulmadi.
        assert done.returncode == 2
        assert data["ffmpeg"] is None
        assert data["error"]
    if data["audio"] is not None:
        assert data["audio"]["opened"] is True
