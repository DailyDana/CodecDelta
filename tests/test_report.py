"""HTML rapor testleri.

Raporun uc sozu var ve her biri ayri sinaniyor:
- Kendi kendine yeter: dis kaynak, betik yok.
- Guvenli: kullanici verisi (dosya adi, etiket, not) HTML olarak calismaz.
- Paylasima hazir: yol, kullanici ve makine adi yok; sizinti varsa dosya YAZILMAZ.
"""

from __future__ import annotations

import dataclasses
import os
import re
import struct
import subprocess
import zlib
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from app.compare.pipeline import compare, open_track
from app.compare.result import ComparisonResult
from app.core.ffmpeg_locate import FFmpegTools
from app.core.messages import Message
from app.core.probe import probe
from app.report import html, png, svg_chart
from app.single import verdict
from app.ui import present
from app.ui.i18n import set_language


@pytest.fixture(autouse=True)
def _english() -> object:
    set_language("en")
    yield
    set_language("en")


# -- parcalar -------------------------------------------------------------------


def test_png_is_valid_and_has_the_right_size() -> None:
    image = np.zeros((7, 11, 3), dtype=np.uint8)
    image[..., 0] = 255
    data = png.encode_rgb(image)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (11, 7)
    # IDAT'i ac ve ilk pikselin kirmizi oldugunu dogrula
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8 : start - 4])[0]
    raw = zlib.decompress(data[start : start + length])
    assert raw[0] == 0 and raw[1:4] == b"\xff\x00\x00"


def test_png_rejects_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        png.encode_rgb(np.zeros((4, 4), dtype=np.uint8))


def test_nmr_colours_follow_the_scale() -> None:
    grid = np.array([[-50.0, 0.0, 50.0, np.nan]])
    rgb = png.colormap(grid, low=-20, high=20, stops=png.NMR_STOPS)[0]
    assert tuple(rgb[0]) == (0x11, 0x11, 0x1B)  # esigin cok alti: en koyu
    assert tuple(rgb[1]) == (0x7F, 0x84, 0x9C)  # tam esik: orta gri
    assert tuple(rgb[2]) == (0xF3, 0x8B, 0xA8)  # esigin cok ustu: en sicak
    assert tuple(rgb[3]) == tuple(rgb[0])  # NaN -> olcek alti


def test_svg_charts_are_well_formed_and_escape_text() -> None:
    chart = svg_chart.category_chart(
        ["<b>", "2", "3"],
        [
            svg_chart.Series("mid & more", [1.0, float("nan"), 3.0], "series-a", "bar"),
            svg_chart.Series("side", [0.5, 1.0, float("inf")], "series-b", "line"),
            svg_chart.Series("floor", [9.0, 9.0, 9.0], "series-c", "dashed"),
        ],
        y_label="SNR <dB>",
        marks=[False, True, False],
    )
    ElementTree.fromstring(chart)
    assert "<b>" not in chart and "&lt;b&gt;" in chart
    spectrum = svg_chart.spectrum_chart(
        list(np.linspace(0, 22050, 2049)),
        list(np.linspace(0, -90, 2049)),
        y_label="dB",
        markers=[(16000.0, "cut <here>")],
    )
    ElementTree.fromstring(spectrum)
    assert spectrum.count("<polyline") == 1
    assert "cut &lt;here&gt;" in spectrum


# -- gercek rapor ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison(
    ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory
) -> ComparisonResult:
    root = tmp_path_factory.mktemp("report")
    ff = str(ffmpeg_tools.ffmpeg)
    reference = root / "ref.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=12:seed=8,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(reference),
        ],
        check=True,
    )
    test = root / "test.mp3"
    subprocess.run(
        [ff, "-v", "error", "-i", str(reference), "-c:a", "libmp3lame", "-b:a", "128k", str(test)],
        check=True,
    )
    return compare(
        ffmpeg_tools.ffmpeg,
        open_track(ffmpeg_tools.ffprobe, reference),
        open_track(ffmpeg_tools.ffprobe, test),
    )


@pytest.mark.needs_ffmpeg
def test_report_is_self_contained(comparison: ComparisonResult) -> None:
    text = html.render_comparison(comparison)
    assert "<script" not in text.lower()
    assert not re.search(r'(src|href)\s*=\s*"(https?:)?//', text)
    assert "@import" not in text and "url(http" not in text
    assert text.count("<svg") >= 1
    assert "data:image/png;base64," in text


@pytest.mark.needs_ffmpeg
def test_user_text_cannot_inject_html(comparison: ComparisonResult) -> None:
    """Autoescape: bir not ya da etiket icindeki isaretleme METIN olarak cikmali."""
    hostile = Message("x", "<script>alert(1)</script><img src=x onerror=alert(2)>")
    poisoned = dataclasses.replace(comparison, notes=(*comparison.notes, hostile))
    text = html.render_comparison(poisoned)
    assert "<script>alert(1)" not in text and "<img src=x" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


@pytest.mark.needs_ffmpeg
def test_report_leaks_no_path_or_user(comparison: ComparisonResult) -> None:
    text = html.render_comparison(comparison)
    assert html.check_privacy(text) == []
    assert str(comparison.reference.path.parent) not in text
    assert comparison.reference.path.name in text


@pytest.mark.needs_ffmpeg
def test_a_leaking_report_is_not_written(comparison: ComparisonResult, tmp_path: Path) -> None:
    """Yol sizan bir rapor diske HIC yazilmamali."""
    leak = str(Path(os.environ.get("USERPROFILE", "C:\\Users\\someone")) / "Music" / "a.flac")
    poisoned = dataclasses.replace(comparison, notes=(f"loaded from {leak}",))
    target = tmp_path / "leak.html"
    with pytest.raises(html.ReportPrivacyError) as caught:
        html.write(target, html.render_comparison(poisoned))
    assert caught.value.leaks
    assert not target.exists()


@pytest.mark.needs_ffmpeg
def test_report_is_written_in_the_interface_language(
    comparison: ComparisonResult, tmp_path: Path
) -> None:
    set_language("tr")
    text = html.render_comparison(comparison)
    assert 'lang="tr"' in text and "Karşılaştırma raporu" in text
    written = html.write(tmp_path / "r.html", text)
    assert written.read_text(encoding="utf-8") == text


@pytest.mark.needs_ffmpeg
def test_verification_report_shows_the_spectrum_and_evidence(
    ffmpeg_tools: FFmpegTools, tmp_path: Path
) -> None:
    ff = str(ffmpeg_tools.ffmpeg)
    clean = tmp_path / "clean.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=12:seed=9",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(clean),
        ],
        check=True,
    )
    lossy = tmp_path / "t.mp3"
    subprocess.run(
        [ff, "-v", "error", "-i", str(clean), "-c:a", "libmp3lame", "-b:a", "128k", str(lossy)],
        check=True,
    )
    fake = tmp_path / "fake.flac"
    subprocess.run([ff, "-v", "error", "-i", str(lossy), "-c:a", "flac", str(fake)], check=True)
    info = probe(ffmpeg_tools.ffprobe, fake)
    result = verdict.verify(ffmpeg_tools.ffmpeg, info)
    text = html.render_verification(result, info)
    assert "Consistent with a lossy source" in text
    assert 'class="for"' in text and "brickwall" in text
    assert "content stops" in text  # grafikteki isaret
    assert html.check_privacy(text) == []


@pytest.mark.needs_ffmpeg
def test_a_partial_measurement_is_flagged(comparison: ComparisonResult) -> None:
    """Hizasiz kisim disarida birakildiysa baslik uyari tonunda ve sure gorunur (D1)."""
    partial = dataclasses.replace(comparison, excluded_s=12.0)
    headline = present.comparison_headline(partial)
    assert headline.tone == "warn" and "12 s" in headline.detail
    rows = dict(present.summary_rows(partial))
    assert any("12 s left out" in value for value in rows.values())
    set_language("tr")
    assert "12 s dışarıda" in html.render_comparison(partial)


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize("name", ["test", "user", "mark"])
def test_a_common_username_does_not_block_the_report(
    comparison: ComparisonResult, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    """`USERNAME=test` iken her rapor reddediliyordu (D4)."""
    monkeypatch.setenv("USERNAME", name)
    text = html.render_comparison(comparison)
    assert html.check_privacy(text) == []
    assert html.write(tmp_path / "r.html", text).exists()


@pytest.mark.needs_ffmpeg
def test_a_blocked_report_says_what_was_found(comparison: ComparisonResult) -> None:
    poisoned = dataclasses.replace(comparison, notes=(r"loaded from D:\Private\a.flac",))
    with pytest.raises(html.ReportPrivacyError) as caught:
        html.write(Path("unused.html"), html.render_comparison(poisoned))
    assert r"D:\Private" in caught.value.user_message()
