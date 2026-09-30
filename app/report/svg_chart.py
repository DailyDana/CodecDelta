"""Elle uretilen inline SVG grafikler.

matplotlib yok (plan: ~60 MB, soguk import 1.2 s, tema kilitli PNG). Burada
uretilen SVG'ler renk tasimaz: cizgiler ve metinler CSS sinifi alir, renkler
raporun stil sayfasindaki degiskenlerden gelir. Ayni SVG acik ve koyu temada
dogru gorunur, her yakinlastirmada keskin kalir ve metni aranabilir.

Tum metin XML'den KACIRILIR; etiketler bizim olsa da SVG dogrudan HTML'e
gomuldugu icin bir `<` belgeyi bozar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class Series:
    name: str
    values: Sequence[float]
    css: str
    kind: str = "line"  # "line" | "bar" | "dashed"


def _nice_step(span: float, target: int = 6) -> float:
    raw = span / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def _fmt(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"


def category_chart(
    labels: Sequence[str],
    series: Sequence[Series],
    *,
    y_label: str,
    width: int = 760,
    height: int = 280,
    y_max: float | None = None,
    marks: Sequence[bool] = (),
) -> str:
    """Kategori ekseni (bantlar) uzerinde cubuk + cizgi grafik.

    `marks[i]` dogruysa o kategorinin ustune "olculemez" isareti konur.
    NaN/sonsuz degerler cizilmez (bosluk birakir).
    """
    left, right, top, bottom = 52, 16, 14, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    finite = [v for s in series for v in s.values if math.isfinite(v)]
    lo = min([0.0, *finite]) if finite else 0.0
    hi = max([1.0, *finite]) if finite else 1.0
    if y_max is not None:
        hi = min(hi, y_max)
    step = _nice_step(hi - lo)
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    n = max(1, len(labels))
    band = plot_w / n

    def x(i: float) -> float:
        return left + band * (i + 0.5)

    def y(v: float) -> float:
        v = min(max(v, lo), hi)
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(y_label)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    tick = lo
    while tick <= hi + 1e-9:
        yy = y(tick)
        grid = "axis" if abs(tick) < 1e-9 else "grid"
        out.append(
            f'<line class="{grid}" x1="{left}" x2="{width - right}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 6}" y="{yy + 4:.1f}" '
            f'text-anchor="end">{_fmt(tick)}</text>'
        )
        tick += step
    out.append(
        f'<text class="label" transform="translate(14 {top + plot_h / 2:.0f}) rotate(-90)" '
        f'text-anchor="middle">{escape(y_label)}</text>'
    )
    for i, label in enumerate(labels):
        out.append(
            f'<text class="tick" x="{x(i):.1f}" y="{height - bottom + 18}" text-anchor="middle">'
            f"{escape(label)}</text>"
        )

    zero = y(max(lo, 0.0))
    for s in series:
        if s.kind == "bar":
            for i, v in enumerate(s.values):
                if not math.isfinite(v):
                    continue
                yy = y(v)
                top_y, h = (yy, zero - yy) if yy < zero else (zero, yy - zero)
                out.append(
                    f'<rect class="{s.css}" x="{x(i) - band * 0.28:.1f}" y="{top_y:.1f}" '
                    f'width="{band * 0.56:.1f}" height="{max(h, 0.5):.1f}" rx="2"/>'
                )
            continue
        runs: list[list[str]] = [[]]
        for i, v in enumerate(s.values):
            if math.isfinite(v) and (y_max is None or v <= y_max):
                runs[-1].append(f"{x(i):.1f},{y(v):.1f}")
            elif runs[-1]:
                runs.append([])
        dashed = ' stroke-dasharray="5 4"' if s.kind == "dashed" else ""
        for run in runs:
            if len(run) > 1:
                out.append(
                    f'<polyline class="{s.css}" fill="none" points="{" ".join(run)}"{dashed}/>'
                )
            if s.kind == "line":
                for point in run:
                    px, py = point.split(",")
                    out.append(f'<circle class="{s.css}-dot" cx="{px}" cy="{py}" r="3"/>')
    for i, marked in enumerate(marks):
        if marked:
            out.append(
                f'<text class="mark" x="{x(i):.1f}" y="{top + 12}" text-anchor="middle">×</text>'
            )

    legend_x = left + 8
    for s in series:
        if s.kind == "bar":
            swatch = (
                f'<rect class="{s.css}" x="{legend_x}" y="{height - 14}" '
                f'width="12" height="8" rx="1"/>'
            )
        else:
            dash = ' stroke-dasharray="4 3"' if s.kind == "dashed" else ""
            swatch = (
                f'<line class="{s.css}" x1="{legend_x}" x2="{legend_x + 14}" '
                f'y1="{height - 10}" y2="{height - 10}"{dash}/>'
            )
        out.append(swatch)
        out.append(
            f'<text class="legend" x="{legend_x + 18}" y="{height - 6}">{escape(s.name)}</text>'
        )
        legend_x += 26 + 7 * len(s.name)
    out.append("</svg>")
    return "".join(out)


def spectrum_chart(
    freqs_hz: Sequence[float],
    level_db: Sequence[float],
    *,
    y_label: str,
    markers: Sequence[tuple[float, str]] = (),
    width: int = 760,
    height: int = 260,
    floor_db: float = -120.0,
) -> str:
    """Log frekans ekseninde uzun donem spektrum (tek dosya dogrulamasi).

    `markers`: (frekans, etiket) dikey cizgiler -- kesim ve diz.
    """
    left, right, top, bottom = 52, 16, 14, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [
        (f, v) for f, v in zip(freqs_hz, level_db, strict=False) if f >= 20.0 and math.isfinite(v)
    ]
    if not points:
        return ""
    f_lo, f_hi = 20.0, max(f for f, _ in points)
    hi = float(math.ceil(max(v for _, v in points) / 10) * 10)
    lo = max(floor_db, float(math.floor(min(v for _, v in points) / 10) * 10))

    def x(f: float) -> float:
        return left + plot_w * math.log10(f / f_lo) / math.log10(f_hi / f_lo)

    def y(v: float) -> float:
        v = min(max(v, lo), hi)
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(y_label)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    step = _nice_step(hi - lo, 5)
    tick = hi
    while tick >= lo - 1e-9:
        yy = y(tick)
        out.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
        )
        out.append(
            f'<text class="tick" x="{left - 6}" y="{yy + 4:.1f}" '
            f'text-anchor="end">{_fmt(tick)}</text>'
        )
        tick -= step
    for tick_hz in (100, 1000, 5000, 10000, 16000, 20000):
        if f_lo <= tick_hz <= f_hi:
            label = f"{tick_hz // 1000}k" if tick_hz >= 1000 else str(tick_hz)
            out.append(
                f'<line class="grid" x1="{x(tick_hz):.1f}" x2="{x(tick_hz):.1f}" '
                f'y1="{top}" y2="{top + plot_h}"/>'
            )
            out.append(
                f'<text class="tick" x="{x(tick_hz):.1f}" y="{height - 12}" '
                f'text-anchor="middle">{label}</text>'
            )
    # Cizgiyi ~800 noktaya seyrelt: 2049 binlik spektrum gereksiz buyuk SVG uretir.
    stride = max(1, len(points) // 800)
    path = " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v in points[::stride])
    out.append(f'<polyline class="series-a" fill="none" points="{path}"/>')
    for f, label in markers:
        if f_lo <= f <= f_hi:
            out.append(
                f'<line class="marker" x1="{x(f):.1f}" x2="{x(f):.1f}" '
                f'y1="{top}" y2="{top + plot_h}"/>'
            )
            out.append(
                f'<text class="marker-text" x="{x(f) - 4:.1f}" y="{top + 12}" text-anchor="end">'
                f"{escape(label)}</text>"
            )
    out.append(
        f'<text class="label" transform="translate(14 {top + plot_h / 2:.0f}) rotate(-90)" '
        f'text-anchor="middle">{escape(y_label)}</text>'
    )
    out.append("</svg>")
    return "".join(out)
