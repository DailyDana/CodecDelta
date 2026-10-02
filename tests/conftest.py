"""Paylasilan pytest fixture'lari."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.core.errors import CodecDeltaError
from app.core.ffmpeg_locate import FFmpegTools, discover


@pytest.fixture(scope="session")
def ffmpeg_tools() -> FFmpegTools:
    """Sistemde bulunan ffmpeg+ffprobe cifti.

    CI runner'inda ffmpeg yok; bu fixture'i kullanan testler `needs_ffmpeg`
    ile isaretlidir ve orada atlanir.
    """
    try:
        return discover()
    except CodecDeltaError as exc:
        pytest.skip(f"ffmpeg bulunamadi: {exc}")


def silent_middle(ffmpeg: Path, out: Path) -> Path:
    """40 s muzik benzeri gurultu + 40 s dijital sessizlik + 40 s gurultu (FLAC).

    Ortadaki 30 s'lik kesit tamamen sessiz kalir (denetim D7, D13).
    """
    noise = "anoisesrc=color=pink:sample_rate=44100:duration=40:seed={s},tremolo=f=0.7:d=0.6"
    graph = (
        f"{noise.format(s=1)}[a];anullsrc=r=44100:cl=mono:d=40[z];{noise.format(s=2)}[b];"
        "[a][z][b]concat=n=3:v=0:a=1"
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-filter_complex",
            graph,
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(out),
        ],
        check=True,
    )
    return out
