"""Grafikler: bant basina S/N ve zaman icinde NMR spektrogrami (pyqtgraph).

Renkler temadan gelir. Spektrogram 0 dB'de (gurultu = maskeleme esigi) renk
degistiren iki yonlu bir olcek kullanir: altinda soguk/koyu (maskeleniyor),
ustunde sicak (maskeyi asiyor). Olcek -20..+20 dB'de kirpilir; NMR'in mutlak
degeri model secimine gore 15 dB oynadigi icin (DECISIONS) daha ince bir
olcek yaniltici bir kesinlik gosterirdi.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from app.psycho.nmr import NmrSummary
from app.ui.present import BandRow, frequency_text
from app.ui.theme import COLORS

pg.setConfigOptions(antialias=True, background=COLORS["mantle"], foreground=COLORS["subtext0"])

NMR_RANGE_DB = 20.0


def _configure(widget: pg.PlotWidget, *, min_height: int) -> None:
    """Ortak eksen/izgara ayari; yakinlastirma kapali (rapor grafigi, gezinme degil)."""
    widget.setMinimumHeight(min_height)
    widget.setMenuEnabled(False)
    widget.setMouseEnabled(x=False, y=False)
    widget.hideButtons()
    item = widget.getPlotItem()
    item.showGrid(x=False, y=True, alpha=0.15)
    for axis in ("left", "bottom"):
        item.getAxis(axis).setPen(pg.mkPen(COLORS["surface1"]))
        item.getAxis(axis).setTextPen(pg.mkPen(COLORS["subtext0"]))


class BandChart(pg.PlotWidget):
    """Bant basina mid/side S/N ve olcum tabani."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        _configure(self, min_height=220)
        item = self.getPlotItem()
        item.setLabel("left", "SNR (dB)")
        item.addLegend(offset=(-10, 10), labelTextColor=COLORS["subtext1"])

    def set_rows(self, rows: Sequence[BandRow]) -> None:
        item = self.getPlotItem()
        item.clear()
        if not rows:
            return
        x = np.arange(len(rows), dtype=float)
        mid = np.array([r.mid_db for r in rows], dtype=float)
        side = np.array([r.side_db for r in rows], dtype=float)
        floor = np.array([r.floor_db for r in rows], dtype=float)
        finite = np.concatenate([v[np.isfinite(v)] for v in (mid, side)] or [np.zeros(1)])
        top = float(np.nanmax(finite)) if finite.size else 30.0
        bottom = float(np.nanmin(finite)) if finite.size else 0.0
        # Taban cogu zaman 90-150 dB; grafigi ezmesin diye gorunur araliga kirpilir.
        ceiling = top + 12.0
        item.addItem(
            pg.BarGraphItem(
                x=x,
                height=np.nan_to_num(mid),
                width=0.55,
                brush=COLORS["accent"],
                pen=None,
                name="Mid",
            )
        )
        side_ok = np.isfinite(side)
        if side_ok.any():
            item.plot(
                x[side_ok],
                side[side_ok],
                pen=pg.mkPen(COLORS["blue"], width=2),
                symbol="o",
                symbolSize=6,
                symbolBrush=COLORS["blue"],
                symbolPen=None,
                name="Side",
            )
        # Taban cogu zaman 90-150 dB: gorunur araligin disindaysa cizilmez
        # (tepede kesilmis bir cizgi bilgi degil gurultu olurdu). Deger tabloda.
        visible = np.isfinite(floor) & (floor <= ceiling)
        if visible.any():
            item.plot(
                x[visible],
                floor[visible],
                pen=pg.mkPen(COLORS["overlay1"], width=1, style=Qt.PenStyle.DashLine),
                name="Floor",
            )
        for i, row in enumerate(rows):
            if not row.measurable:
                marker = pg.TextItem("×", color=COLORS["yellow"], anchor=(0.5, 1.0))
                marker.setPos(i, float(np.nan_to_num(mid[i])))
                item.addItem(marker)
        axis = item.getAxis("bottom")
        axis.setTicks([[(i, r.label) for i, r in enumerate(rows)]])
        item.setYRange(min(0.0, bottom - 3.0), ceiling, padding=0)
        item.setXRange(-0.6, len(rows) - 0.4, padding=0)


def _nmr_colormap() -> pg.ColorMap:
    positions = np.array([0.0, 0.35, 0.5, 0.65, 1.0])
    colors = [
        COLORS["crust"],
        COLORS["surface1"],
        COLORS["overlay1"],
        COLORS["peach"],
        COLORS["red"],
    ]
    return pg.ColorMap(positions, [pg.mkColor(c) for c in colors])


class NmrView(pg.PlotWidget):
    """Bant x zaman NMR (dB). Hucre: o zaman araligindaki en kotu deger."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        _configure(self, min_height=240)
        self.getPlotItem().showGrid(x=False, y=False)
        self._image = pg.ImageItem()
        self._image.setColorMap(_nmr_colormap())
        item = self.getPlotItem()
        item.addItem(self._image)
        item.setLabel("bottom", "s")
        self._bar = pg.ColorBarItem(
            values=(-NMR_RANGE_DB, NMR_RANGE_DB),
            colorMap=_nmr_colormap(),
            interactive=False,
            width=12,
            label="NMR (dB)",
        )
        self._bar.setImageItem(self._image, insert_in=item)

    def set_summary(self, summary: NmrSummary | None) -> None:
        if summary is None or summary.grid_db.size == 0:
            self._image.clear()
            return
        grid = np.clip(
            np.nan_to_num(summary.grid_db, neginf=-NMR_RANGE_DB), -NMR_RANGE_DB, NMR_RANGE_DB
        )
        # ImageItem (x, y) bekler: zaman x, bant y.
        self._image.setImage(grid.T, levels=(-NMR_RANGE_DB, NMR_RANGE_DB), autoLevels=False)
        width = grid.shape[1] * summary.seconds_per_column
        self._image.setRect(0.0, 0.0, width, float(grid.shape[0]))
        centres = summary.centre_hz
        step = max(1, len(centres) // 8)
        ticks = [(i + 0.5, frequency_text(centres[i])) for i in range(0, len(centres), step)]
        self.getPlotItem().getAxis("left").setTicks([ticks])
        self.getPlotItem().setXRange(0, width, padding=0)
        self.getPlotItem().setYRange(0, grid.shape[0], padding=0)


class SweepChart(pg.PlotWidget):
    """Bit hizi taramasi: codec S/N'in bitrate'e gore egrisi."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        _configure(self, min_height=200)
        self.setLabel("left", "SNR (dB)")
        self.setLabel("bottom", "kbps")

    def show_points(self, bitrates: list[int], snr_db: list[float]) -> None:
        self.clear()
        finite = [
            (b, s)
            for b, s in zip(bitrates, snr_db, strict=True)
            if s == s and abs(s) != float("inf")
        ]
        if not finite:
            return
        xs, ys = zip(*finite, strict=True)
        self.plot(
            list(xs),
            list(ys),
            pen=pg.mkPen(COLORS["accent"], width=2),
            symbol="o",
            symbolBrush=COLORS["accent"],
            symbolPen=None,
        )
        self.getPlotItem().getAxis("bottom").setTicks([[(b, str(b)) for b in bitrates]])
