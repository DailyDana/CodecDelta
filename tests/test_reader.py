"""Pencere okuyucu sozlesme testleri.

Sozlesme tek cumle: okunan pencere, dosyanin bastan tam cozumunun ilgili
dilimiyle ornek ornek AYNI. Hem hizli (arama) hem guvenli (bastan cozme)
yol, yeniden ornekleme ile ve olmadan, izgaraya oturan ve oturmayan
baslangiclarla sinaniyor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.compare.reader import FFmpegWindowReader, Source, seek_exact
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_stream import open_pcm
from app.core.probe import probe


def test_seek_whitelist_is_measured_pairs_only() -> None:
    assert seek_exact("flac", "flac")
    assert seek_exact("ogg", "opus")
    assert seek_exact("mp3", "mp3")
    assert seek_exact("wav", "pcm_s16le")
    # Olculdu ve ornek-dogru DEGIL:
    assert not seek_exact("ogg", "vorbis")
    assert not seek_exact("mov,mp4,m4a,3gp,3g2,mj2", "aac")
    assert not seek_exact("matroska,webm", "opus")
    assert not seek_exact("matroska,webm", "flac")
    # Bilinmeyen -> guvenli yol
    assert not seek_exact("asf", "wmav2")


# Her girdi: (dosya adi, ffmpeg kodlayici argumanlari, hizli yolda mi olmali)
FORMATS = [
    ("t.wav", ["-c:a", "pcm_s16le"], True),
    ("t.flac", ["-c:a", "flac"], True),
    ("t.mp3", ["-c:a", "libmp3lame", "-q:a", "2"], True),
    ("t.opus", ["-c:a", "libopus", "-b:a", "96k"], True),
    ("t.m4a", ["-c:a", "aac", "-b:a", "128k"], False),
    ("t.ogg", ["-c:a", "libvorbis", "-q:a", "4"], False),
    ("t.webm", ["-c:a", "libopus", "-b:a", "96k"], False),
]


@pytest.fixture(scope="module")
def encoded(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("reader")
    for name, args, _ in FORMATS:
        subprocess.run(
            [
                str(ffmpeg_tools.ffmpeg),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=pink:sample_rate=44100:duration=12:seed=3,tremolo=f=1.3:d=0.8",
                "-ac",
                "2",
                *args,
                str(root / name),
            ],
            check=True,
        )
    return root


def _full(ffmpeg_tools: FFmpegTools, source: Source) -> np.ndarray:
    blocks = []
    with open_pcm(
        ffmpeg_tools.ffmpeg,
        source.path,
        sample_rate=source.rate,
        channels=source.channels,
        stream_index=source.stream_index,
        rate=source.target_rate,
    ) as stream:
        for block in stream.blocks(1 << 16):
            blocks.append(block.copy())
    return np.concatenate(blocks)


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize(("name", "fast"), [(n, f) for n, _, f in FORMATS])
@pytest.mark.parametrize("target_rate", [None, 48000])
def test_window_equals_slice_of_full_decode(
    ffmpeg_tools: FFmpegTools, encoded: Path, name: str, fast: bool, target_rate: int | None
) -> None:
    path = encoded / name
    info = probe(ffmpeg_tools.ffprobe, path)
    stream = info.audio[0]
    exact = seek_exact(info.container, stream.codec)
    assert exact is fast, f"{info.container}/{stream.codec} beyaz listede beklenmedik durumda"

    source = Source(
        path=path,
        stream_index=0,
        channels=stream.channels,
        source_rate=stream.sample_rate,
        target_rate=target_rate,
        exact_seek=exact,
    )
    full = _full(ffmpeg_tools, source)
    reader = FFmpegWindowReader(ffmpeg_tools.ffmpeg, source)
    rate = source.rate
    # 13.3712 s: 44.1/48 kHz ortak izgarasinin disinda; kirpma dogru mu?
    for start_s in (0.0, 0.2, 5.0, 13.3712 * 0.5, 9.0):
        start = round(start_s * rate)
        count = rate
        window = reader(start, count)
        expected = full[start : start + count]
        assert window.shape == expected.shape
        assert np.array_equal(window, expected), f"{name} @ {start_s} s"


@pytest.mark.needs_ffmpeg
def test_window_past_end_is_short_not_padded(ffmpeg_tools: FFmpegTools, encoded: Path) -> None:
    path = encoded / "t.flac"
    source = Source(path, 0, 2, 44100, None, exact_seek=True)
    reader = FFmpegWindowReader(ffmpeg_tools.ffmpeg, source)
    window = reader(11 * 44100, 5 * 44100)
    assert window.shape == (44100, 2)
    assert reader(20 * 44100, 1000).shape == (0, 2)


def test_rejects_negative_arguments() -> None:
    reader = FFmpegWindowReader(Path("ffmpeg"), Source(Path("x.flac"), 0, 2, 44100))
    with pytest.raises(ValueError, match="negatif"):
        reader(-1, 10)
    assert reader(0, 0).shape == (0, 2)
