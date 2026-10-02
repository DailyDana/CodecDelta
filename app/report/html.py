"""Tek dosyalik, paylasilabilir HTML rapor.

Ilkeler:
- Kendi kendine yeter: dis stil, betik, yazi tipi, goruntu yok. Grafikler
  inline SVG, NMR izgarasi data: URI'li PNG.
- Jinja2 `autoescape=True` ACIKCA verilir (Jinja2'de varsayilan kapali).
  Dosya adi ya da bir etiket icindeki `<script>` gercek bir XSS olurdu.
- Varsayilan olarak paylasima hazir: mutlak yol, kullanici ve makine adi
  hic yazilmaz; dosya kimligi ad + icerik ozeti. Yazmadan once
  `privacy.audit()` calisir ve bir sey bulursa dosya YAZILMAZ.
- Metin arayuzun dilinde (`app.ui.i18n`, Qt'siz).

Bu modul Qt IMPORT ETMEZ.
"""

from __future__ import annotations

import base64
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import jinja2

from app import __version__
from app.compare.ladder import LadderVerdict
from app.compare.result import ComparisonResult, FileSummary
from app.core import privacy
from app.core.errors import CodecDeltaError
from app.core.messages import Message
from app.core.probe import Probe
from app.report import png, svg_chart
from app.single.verdict import Verdict
from app.ui import present
from app.ui.i18n import language, localize, tr

_TEMPLATES = Path(__file__).with_name("templates")
_DATA_URI = re.compile(r"data:[a-z/+.-]+;base64,[A-Za-z0-9+/=]+")


class ReportPrivacyError(CodecDeltaError):
    """Raporda kisisel iz kaldi; dosya yazilmadi."""

    def __init__(self, leaks: list[str]) -> None:
        self.leaks = leaks
        # Ilk birkaci gosterilir: kullanici neyin engellendigini bilmeli (D4).
        # Ileti yalnizca yerel arayuzde gorunur; rapor zaten yazilmadi.
        shown = ", ".join(dict.fromkeys(leaks[:3]))
        super().__init__(
            Message(
                "report.privacy",
                "The report still contains {count} personal trace(s) ({shown}); it was not "
                "written.",
                count=len(leaks),
                shown=shown,
            )
        )


@dataclass(frozen=True)
class FileCard:
    role: str
    name: str
    identity: str
    details: str


def _environment() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )


def self_check() -> None:
    """Sablon bulunup derlenebiliyor mu (paketlenmis yapinin duman testi)."""
    _environment().get_template("report.html.j2")


def _identity(path: Path) -> str:
    try:
        return privacy.content_id(path)
    except OSError:
        return "--"


def _duration(seconds: float | None) -> str:
    if not seconds:
        return "--"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"


def _card_from_summary(role: str, summary: FileSummary) -> FileCard:
    details = "  ·  ".join(
        [
            summary.codec,
            f"{summary.sample_rate / 1000:g} kHz",
            f"{summary.channels} ch",
            _duration(summary.duration),
            summary.container.split(",")[0],
        ]
    )
    return FileCard(role, privacy.scrub_path(summary.path), _identity(summary.path), details)


def _card_from_probe(role: str, info: Probe, stream_index: int) -> FileCard:
    stream = info.stream(stream_index)
    parts = [stream.codec, f"{stream.sample_rate / 1000:g} kHz"]
    if stream.bits_per_raw_sample:
        parts.append(f"{stream.bits_per_raw_sample}-bit")
    parts += [f"{stream.channels} ch", _duration(info.duration), info.container.split(",")[0]]
    return FileCard(role, privacy.scrub_path(info.path), _identity(info.path), "  ·  ".join(parts))


def _common(title: str) -> dict[str, object]:
    return {
        "lang": language(),
        "title": title,
        "t": tr,
        "version": __version__,
        "generated": dt.date.today().isoformat(),
    }


def render_comparison(result: ComparisonResult, *, ladder: LadderVerdict | None = None) -> str:
    """Karsilastirma raporu (HTML dizesi)."""
    rows = present.band_rows(result)
    chart = ""
    if rows:
        ceiling = max((r.mid_db for r in rows if r.mid_db == r.mid_db), default=30.0) + 12.0
        # Taban cogu zaman 90-150 dB: gorunur araligin disindaysa ne cizilir ne
        # lejantta gorunur (deger tabloda). Lejantta olup grafikte olmayan bir
        # seri okuyucuya cizgiyi aratir.
        floors = [r.floor_db for r in rows]
        floor_series = (
            [svg_chart.Series(tr("bands.floor"), floors, "series-c", "dashed")]
            if any(f == f and f <= ceiling for f in floors)
            else []
        )
        chart = svg_chart.category_chart(
            [r.label for r in rows],
            [
                svg_chart.Series(tr("bands.mid"), [r.mid_db for r in rows], "series-a", "bar"),
                svg_chart.Series(tr("bands.side"), [r.side_db for r in rows], "series-b", "line"),
                *floor_series,
            ],
            y_label="SNR (dB)",
            y_max=ceiling,
            marks=[not r.measurable for r in rows],
        )
    nmr_image = ""
    nmr_seconds = 0.0
    nmr_ticks: list[str] = []
    if result.nmr is not None and result.nmr.frames and result.nmr.grid_db.size:
        data = base64.b64encode(png.nmr_png(result.nmr.grid_db)).decode("ascii")
        nmr_image = f"data:image/png;base64,{data}"
        nmr_seconds = result.nmr.grid_db.shape[1] * result.nmr.seconds_per_column
        centres = result.nmr.centre_hz
        nmr_ticks = [
            present.frequency_text(centres[i]) for i in (len(centres) - 1, len(centres) // 2, 0)
        ]
    ladder_rows = []
    if ladder is not None:
        ladder_rows = [
            (f"{r.bitrate_kbps} kbps", present.db_text(r.snr_db), present.db_text(r.nmr_p95_db))
            for r in ladder.rungs
        ]
    headline = present.comparison_headline(result)
    context = _common(tr("report.title_compare"))
    context.update(
        kind="compare",
        headline=headline,
        files=[
            _card_from_summary(tr("slot.reference"), result.reference),
            _card_from_summary(tr("slot.test"), result.test),
        ],
        summary=present.summary_rows(result),
        bands=rows,
        chart=chart,
        nmr_image=nmr_image,
        nmr_seconds=f"{nmr_seconds:.0f}",
        nmr_ticks=nmr_ticks,
        nmr_range=f"{png.NMR_RANGE_DB:.0f}",
        ladder_text=present.ladder_text(ladder) if ladder is not None else "",
        ladder_rows=ladder_rows,
        notes=[localize(n) for n in (*result.plan.reasons, *result.notes)],
        reasons=[],
        counters=[],
        spectrum="",
    )
    return _environment().get_template("report.html.j2").render(**context)


def render_verification(verdict: Verdict, info: Probe, *, stream_index: int = 0) -> str:
    """Tek dosya dogrulama raporu (HTML dizesi)."""
    evidence = verdict.spectral
    spectrum = ""
    if evidence is not None and evidence.frames:
        markers = [(evidence.cutoff_median_hz, tr("report.cutoff_marker"))]
        if abs(evidence.knee_hz - evidence.cutoff_median_hz) > 400:
            markers.append((evidence.knee_hz, tr("report.knee_marker")))
        spectrum = svg_chart.spectrum_chart(
            list(evidence.freqs_hz), list(evidence.ltas_rel_db), y_label="dB", markers=markers
        )
    context = _common(tr("report.title_verify"))
    context.update(
        kind="verify",
        headline=present.verdict_headline(verdict),
        files=[_card_from_probe(tr("report.file"), info, stream_index)],
        summary=present.verdict_rows(verdict),
        bands=[],
        chart="",
        nmr_image="",
        nmr_seconds="0",
        nmr_ticks=[],
        nmr_range="0",
        ladder_text="",
        ladder_rows=[],
        notes=[localize(n) for n in verdict.notes],
        reasons=[localize(r) for r in verdict.reasons],
        counters=[localize(c) for c in verdict.counter_reasons],
        spectrum=spectrum,
    )
    return _environment().get_template("report.html.j2").render(**context)


def check_privacy(html: str) -> list[str]:
    """Raporda kalan kisisel izler. data: URI'leri disarida birakilir.

    Goruntu verisi bizim urettigimiz base64'tur ve metin tasimaz; icinde
    tesadufen bir kullanici adi harf dizisi gecebilir (sahte alarm).
    """
    return privacy.audit(_DATA_URI.sub("data:", html))


def write(path: Path, html: str) -> Path:
    """Raporu yazar. Kisisel iz kaldiysa YAZMAZ ve `ReportPrivacyError` firlatir."""
    leaks = check_privacy(html)
    if leaks:
        raise ReportPrivacyError(leaks)
    path.write_text(html, encoding="utf-8")
    return path
