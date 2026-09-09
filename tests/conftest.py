"""Paylasilan pytest fixture'lari."""

from __future__ import annotations

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
