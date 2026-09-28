"""Koyu tema: Catppuccin Mocha paleti + Aniflow'un mor vurgusu.

Tek `COLORS` sozlugu ve tek `STYLESHEET`; widget'larda renk sabit yazilmaz.
Grafikler (pyqtgraph) da ayni sozlukten renk alir.
"""

from __future__ import annotations

COLORS = {
    "crust": "#11111b",
    "mantle": "#181825",
    "base": "#1e1e2e",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "overlay1": "#7f849c",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "text": "#cdd6f4",
    "accent": "#8b5cf6",
    "accent_hover": "#a78bfa",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "red": "#f38ba8",
}

TONE_COLORS = {
    "good": COLORS["green"],
    "warn": COLORS["yellow"],
    "bad": COLORS["red"],
    "neutral": COLORS["accent"],
}

C = COLORS

STYLESHEET = f"""
QWidget {{
    background: {C["base"]};
    color: {C["text"]};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
}}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {C["base"]}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: {C["mantle"]}; color: {C["subtext0"]};
    padding: 8px 18px; border: none; margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: {C["surface0"]}; color: {C["text"]}; }}
QStatusBar {{ background: {C["mantle"]}; color: {C["subtext0"]}; }}
QMenuBar {{ background: {C["mantle"]}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {C["surface1"]}; }}
QMenu {{ background: {C["mantle"]}; border: 1px solid {C["surface1"]}; }}

QFrame#Card {{
    background: {C["mantle"]};
    border: 1px solid {C["surface0"]};
    border-radius: 10px;
}}
QFrame#DropSlot {{
    background: {C["mantle"]};
    border: 2px dashed {C["surface2"]};
    border-radius: 10px;
}}
QFrame#DropSlot[filled="true"] {{ border: 2px solid {C["surface1"]}; }}
QFrame#DropSlot[hover="true"] {{ border: 2px dashed {C["accent"]}; background: {C["surface0"]}; }}
QLabel {{ background: transparent; }}
QLabel#SlotTitle {{ color: {C["subtext0"]}; font-size: 9pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#SlotName {{ font-size: 11pt; font-weight: 600; }}
QLabel#SlotInfo, QLabel#Muted {{ color: {C["subtext0"]}; }}
QLabel#HeadlineTitle {{ font-size: 18pt; font-weight: 700; }}
QLabel#HeadlineDetail {{ color: {C["subtext1"]}; }}
QLabel#SectionTitle {{ color: {C["subtext0"]}; font-size: 9pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#Stage {{ color: {C["subtext1"]}; }}

QPushButton {{
    background: {C["surface0"]}; color: {C["text"]};
    border: 1px solid {C["surface1"]}; border-radius: 6px; padding: 7px 16px;
}}
QPushButton:hover {{ background: {C["surface1"]}; }}
QPushButton:disabled {{ color: {C["overlay0"]}; background: {C["mantle"]}; border-color: {C["surface0"]}; }}
QPushButton#Primary {{ background: {C["accent"]}; border: none; color: white; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {C["accent_hover"]}; }}
QPushButton#Primary:disabled {{ background: {C["surface1"]}; color: {C["overlay0"]}; }}
QPushButton#Link {{ background: transparent; border: none; color: {C["subtext0"]}; padding: 2px 6px; }}
QPushButton#Link:hover {{ color: {C["text"]}; }}

QComboBox {{
    background: {C["surface0"]}; border: 1px solid {C["surface1"]};
    border-radius: 5px; padding: 3px 8px;
}}
QComboBox QAbstractItemView {{ background: {C["mantle"]}; selection-background-color: {C["surface1"]}; }}
QProgressBar {{
    background: {C["surface0"]}; border: none; border-radius: 3px; max-height: 6px;
}}
QProgressBar::chunk {{ background: {C["accent"]}; border-radius: 3px; }}

QTableWidget {{
    background: {C["mantle"]}; border: none; gridline-color: {C["surface0"]};
    alternate-background-color: {C["base"]};
}}
QHeaderView::section {{
    background: {C["mantle"]}; color: {C["subtext0"]}; border: none;
    border-bottom: 1px solid {C["surface1"]}; padding: 4px 8px; font-weight: 600;
}}
QScrollBar:vertical {{ background: {C["base"]}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C["surface1"]}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
