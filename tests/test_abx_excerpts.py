"""ABX kesit hazirlama: B, A'nin zaman ekseninde ve seviyesinde olmali."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.abx import excerpts
from app.compare.pipeline import compare, open_track
from app.core.ffmpeg_locate import FFmpegTools

SOURCE = "anoisesrc=color=pink:sample_rate=44100:duration=30:seed=12,tremolo=f=0.9:d=0.7"


def _residual_db(a: np.ndarray, b: np.ndarray) -> float:
    """b'nin a'dan farkinin a'ya gore seviyesi (dB, kucuk = iyi)."""
    core = slice(2000, -2000)
    diff = np.sum((a[core] - b[core]) ** 2)
    return float(10 * np.log10(diff / np.sum(a[core] ** 2)))


@pytest.fixture(scope="module")
def pair(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("abx")
    ff = str(ffmpeg_tools.ffmpeg)
    ref = root / "ref.flac"
    subprocess.run(
        [ff, "-v", "error", "-f", "lavfi", "-i", SOURCE, "-c:a", "flac", str(ref)], check=True
    )
    # Kayipsiz ama 370 ms gecikmeli, -2 dB ve polaritesi ters
    shifted = root / "shifted.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-i",
            str(ref),
            "-af",
            "adelay=370,volume=-0.7943",
            "-c:a",
            "flac",
            "-sample_fmt",
            "s32",
            str(shifted),
        ],
        check=True,
    )
    return {"ref": ref, "shifted": shifted}


def _run(tools: FFmpegTools, ref: Path, test: Path):  # type: ignore[no-untyped-def]
    a, b = open_track(tools.ffprobe, ref), open_track(tools.ffprobe, test)
    return a, b, compare(tools.ffmpeg, a, b)


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize("rate", [None, 48000])
def test_b_lines_up_with_a_in_time_level_and_polarity(
    ffmpeg_tools: FFmpegTools, pair: dict[str, Path], rate: int | None
) -> None:
    reference, test, result = _run(ffmpeg_tools, pair["ref"], pair["shifted"])
    assert result.status == "measured" and result.polarity == -1
    audio = excerpts.prepare(ffmpeg_tools.ffmpeg, reference, test, result, 8.0, 6.0, rate=rate)
    assert audio.a.shape == audio.b.shape and audio.a.dtype == np.float32
    assert audio.rate == (rate or 44100)
    # Kayipsiz kopya: hizalama ve seviye dogruysa fark cok kucuk kalir
    assert _residual_db(audio.a, audio.b) < -60.0
    assert audio.b_gain_db == pytest.approx(2.0, abs=0.05)
    assert max(np.abs(audio.a).max(), np.abs(audio.b).max()) <= 0.98 + 1e-6


@pytest.mark.needs_ffmpeg
def test_excerpt_starts_stay_inside_the_overlap(
    ffmpeg_tools: FFmpegTools, pair: dict[str, Path]
) -> None:
    _, _, result = _run(ffmpeg_tools, pair["ref"], pair["shifted"])
    span = excerpts.valid_range(result, 15.0)
    assert span is not None and span[0] < span[1]
    critical = excerpts.critical_start(result, 15.0)
    assert critical is not None and span[0] <= critical <= span[1]
    first = excerpts.random_start(result, 15.0, random.Random(1))
    again = excerpts.random_start(result, 15.0, random.Random(1))
    assert first == again and first is not None and span[0] <= first <= span[1]
    assert excerpts.valid_range(result, 100.0) is None
