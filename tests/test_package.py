"""Paket iskeletinin ayakta oldugunu dogrulayan duman testi.

Asil degeri katman kuralini kilitlemesi: motor katmanlari Qt'ye bulasirsa
headless test ve CLI yolu kapanir, bu da geri donusu pahali bir hatadir.
"""

from __future__ import annotations

import importlib
import pkgutil

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
]


def test_version_is_set() -> None:
    assert app.__version__


def test_engine_layers_import_without_qt() -> None:
    """Motor katmanlari Qt olmadan import edilebilmeli."""
    for name in ENGINE_PACKAGES:
        importlib.import_module(f"app.{name}")


def test_engine_layers_do_not_import_qt() -> None:
    """Motor katmanlarinin hicbir modulu PyQt6'ya bagimli olmamali."""
    import sys

    for name in ENGINE_PACKAGES:
        pkg = importlib.import_module(f"app.{name}")
        for mod in pkgutil.iter_modules(pkg.__path__, prefix=f"app.{name}."):
            importlib.import_module(mod.name)

    qt_modules = [m for m in sys.modules if m.startswith("PyQt6")]
    assert not qt_modules, f"motor katmani Qt import etti: {qt_modules}"
