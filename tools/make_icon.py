"""Uygulama ikonunu uretir: `app/ui/assets/codecdelta.ico`.

Tasarim: koyu yuvarlak kare, mor delta (fark), icinde birbirini kesen iki
dalga -- referans (camgobegi) ve daha basik, faz kaymali kodlanmis sinyal
(mor). Dalgalar ucgenin icine kirpilir.

Kucuk boyutlar ayri cizilir: 24 px ve altinda ucgenin ic alani birkac
pikseldir, iki dalga tek bir lekeye doner. Orada tek, kalin dalga kalir.
Cizgiler boyut kuculdukce kalinlasir.

SVG yerine QPainter: Qt'nin SVG motoru (SVG Tiny) clipPath desteklemiyor,
dalgalar ucgenin disina tasiyordu.

ICO her boyutu PNG olarak tasir (Windows Vista'dan beri desteklenir).

    .venv\\Scripts\\python tools\\make_icon.py            # ikonu yeniden uret
    .venv\\Scripts\\python tools\\make_icon.py --preview onizleme.png
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QGradient,
    QGuiApplication,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "ui" / "assets" / "codecdelta.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

TILE = "#1e1e2e"
DELTA = ("#c4b5fd", "#8b5cf6")
REFERENCE = "#67e8f9"
ENCODED = "#a78bfa"
ENCODED_LIGHT = "#e9d5ff"

# 256 birimlik tuvalde ucgenin kose noktalari
_TOP, _RIGHT, _LEFT = QPointF(128, 40), QPointF(226, 212), QPointF(30, 212)


def _triangle(inset: float = 0.0) -> QPainterPath:
    """Agirlik merkezine dogru `inset` oraninda kucultulmus ucgen."""
    cx = (_TOP.x() + _RIGHT.x() + _LEFT.x()) / 3
    cy = (_TOP.y() + _RIGHT.y() + _LEFT.y()) / 3
    k = 1.0 - inset
    points = [QPointF(cx + (p.x() - cx) * k, cy + (p.y() - cy) * k) for p in (_TOP, _RIGHT, _LEFT)]
    path = QPainterPath()
    path.moveTo(points[0])
    path.lineTo(points[1])
    path.lineTo(points[2])
    path.closeSubpath()
    return path


def _wave(amp: float, phase: float, cy: float, period: float) -> QPainterPath:
    path = QPainterPath()
    for i, x in enumerate(range(0, 257, 2)):
        y = cy - amp * math.sin((x - 128) / period * 2 * math.pi + phase)
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    return path


def _pen(brush: QBrush, width: float) -> QPen:
    pen = QPen(brush, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def draw(painter: QPainter, size: int) -> None:
    """Ikonu `size` x `size` piksele cizer."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 256, size / 256)
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(8, 8, 240, 240), 56, 56)
    painter.fillPath(tile, QBrush(QColor(TILE)))

    gradient = QLinearGradient(0, 0, 256, 256)
    gradient.setColorAt(0, QColor(DELTA[0]))
    gradient.setColorAt(1, QColor(DELTA[1]))
    gradient.setSpread(QGradient.Spread.PadSpread)

    painter.save()
    painter.setClipPath(_triangle(inset=0.10))
    if size <= 24:
        # Tek, kalin dalga: iki cizgi bu boyutta ayirt edilemez.
        painter.setPen(_pen(QBrush(QColor(REFERENCE)), 30))
        painter.drawPath(_wave(26, 0.0, 168, 124))
        tri_width = 34.0
    else:
        mid = size <= 64
        width = 14.0 if mid else 10.0
        cy, period = (166.0, 96.0) if mid else (164.0, 72.0)
        # Orta boyutta ikinci dalga daha acik ve daha kaymis: yoksa deltanin
        # moruna karisip camgobeginin altinda kayboluyordu.
        painter.setPen(_pen(QBrush(QColor(ENCODED_LIGHT if mid else ENCODED)), width))
        painter.drawPath(_wave(14, 1.7 if mid else 1.0, cy, period))
        painter.setPen(_pen(QBrush(QColor(REFERENCE)), width))
        painter.drawPath(_wave(26, 0.0, cy, period))
        tri_width = 26.0 if size <= 48 else 18.0  # 64 px: ince kenar, orta dalga
    painter.restore()

    painter.setPen(_pen(QBrush(gradient), tri_width))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(_triangle())


def render(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    draw(painter, size)
    painter.end()
    return image


def _png(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data.data())


def ico_bytes(sizes: tuple[int, ...] = SIZES) -> bytes:
    """Her boyutu PNG olarak tasiyan ICO dosyasi."""
    images = [_png(render(size)) for size in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries = b""
    for size, data in zip(sizes, images, strict=True):
        side = 0 if size >= 256 else size  # ICO'da 0 = 256
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return header + entries + b"".join(images)


def preview(path: Path) -> None:
    """Tum boyutlar, koyu ve acik zemin uzerinde yan yana."""
    width = sum(SIZES) + 20 * (len(SIZES) + 1)
    sheet = QImage(width, 2 * 296, QImage.Format.Format_ARGB32)
    painter = QPainter(sheet)
    for row, background in enumerate(("#202020", "#f3f3f3")):
        painter.fillRect(0, row * 296, width, 296, QColor(background))
        x = 20
        for size in SIZES:
            painter.drawImage(x, row * 296 + 276 - size, render(size))
            x += size + 20
    painter.end()
    sheet.save(str(path))


def main(argv: list[str]) -> int:
    app = QGuiApplication(argv[:1])  # QPainter metin disinda da bir uygulama ister
    if len(argv) > 2 and argv[1] == "--preview":
        preview(Path(argv[2]))
        print(f"onizleme: {argv[2]}")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(ico_bytes())
        print(f"{TARGET} ({TARGET.stat().st_size} bayt, {len(SIZES)} boyut)")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
