"""Paket iskeletinin ayakta oldugunu dogrulayan duman testi.

Asil degeri katman kuralini kilitlemesi: motor katmanlari Qt'ye bulasirsa
headless test ve CLI yolu kapanir, bu da geri donusu pahali bir hatadir.
"""

from __future__ import annotations

import importlib

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
