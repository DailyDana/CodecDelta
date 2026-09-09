"""ffmpeg keşfi ve yetenek ayristirma testleri."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.ffmpeg_locate import (
    Capabilities,
    _pair_in,
    _parse_listing,
    candidate_dirs,
    load_cached_capabilities,
    parse_version,
    save_cached_capabilities,
)

EXE = ".exe" if os.name == "nt" else ""

# Gercek `ffmpeg -version` ciktisinin kisaltilmis hali. Onemli olan uc sey
# korundu: surum satirinin bicimi, --enable-* bayraklari ve bir --disable-*.
VERSION_SAMPLE = """ffmpeg version N-125875-g5d4d3bdc61-20260731 Copyright (c) 2000-2026
built with gcc 15.2.0 (crosstool-NG 1.28.0.23_185f348)
configuration: --enable-gpl --enable-libopus --enable-libsoxr --disable-libfdk-aac
libavutil      61.  5.100 / 61.  5.100
"""

ENCODERS_SAMPLE = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 A....D aac                  AAC (Advanced Audio Coding)
 A....D libopus              libopus Opus (codec opus)
 A....D flac                 FLAC (Free Lossless Audio Codec)
"""

FILTERS_SAMPLE = """Filters:
  T.. = Timeline support
  ... = etc
  -----
 .S showspectrumpic  A->V       Convert input audio to a spectrum video output.
 .. aresample        A->A       Resample audio data.
 .. axcorrelate      AA->A      Cross-correlate two audio streams.
"""


def test_parse_version_extracts_version_and_enable_flags() -> None:
    version, flags = parse_version(VERSION_SAMPLE)
    assert version == "N-125875-g5d4d3bdc61-20260731"
    assert "libopus" in flags
    assert "libsoxr" in flags
    # --disable-* bayraklari --enable-* olarak sayilmamali
    assert "libfdk-aac" not in flags


def test_parse_listing_reads_name_column() -> None:
    assert _parse_listing(ENCODERS_SAMPLE) == {"aac", "libopus", "flac"}
    assert _parse_listing(FILTERS_SAMPLE) == {"showspectrumpic", "aresample", "axcorrelate"}


def test_parse_listing_ignores_header_before_separator() -> None:
    """Ayrac satirindan onceki aciklama satirlari isim olarak sayilmamali."""
    names = _parse_listing(FILTERS_SAMPLE)
    assert "=" not in names
    assert "Timeline" not in names


def test_pair_requires_ffprobe_in_same_directory(tmp_path: Path) -> None:
    """Altin kural: ffprobe'suz dizin reddedilir.

    Bu, PATH'te duran `--disable-ffprobe` ile derlenmis MPV kopyasinin
    secilmesini engelleyen tek mekanizma.
    """
    (tmp_path / f"ffmpeg{EXE}").write_bytes(b"stub")
    assert _pair_in(tmp_path) is None, "ffprobe yokken cift kabul edildi"

    (tmp_path / f"ffprobe{EXE}").write_bytes(b"stub")
    pair = _pair_in(tmp_path)
    assert pair is not None
    assert pair[0].name == f"ffmpeg{EXE}"
    assert pair[1].name == f"ffprobe{EXE}"


def test_pair_rejects_directory_with_only_ffprobe(tmp_path: Path) -> None:
    (tmp_path / f"ffprobe{EXE}").write_bytes(b"stub")
    assert _pair_in(tmp_path) is None


def test_candidate_dirs_puts_explicit_first(tmp_path: Path) -> None:
    dirs = candidate_dirs(explicit=tmp_path)
    assert dirs[0] == tmp_path.resolve()


def test_candidate_dirs_deduplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), str(tmp_path)]))
    dirs = candidate_dirs(explicit=tmp_path)
    assert dirs.count(tmp_path.resolve()) == 1


def test_capabilities_missing_required_reports_sorted() -> None:
    caps = Capabilities(
        version="x", build_flags=frozenset(), encoders=frozenset(), filters=frozenset()
    )
    assert caps.missing_required() == ["aresample", "astats"]


def test_capability_cache_roundtrip(tmp_path: Path) -> None:
    exe = tmp_path / f"ffmpeg{EXE}"
    exe.write_bytes(b"stub")
    cache = tmp_path / "caps.json"
    caps = Capabilities(
        version="N-1",
        build_flags=frozenset({"libopus"}),
        encoders=frozenset({"libopus", "flac"}),
        filters=frozenset({"aresample", "astats"}),
    )
    save_cached_capabilities(exe, caps, cache)
    assert load_cached_capabilities(exe, cache) == caps


def test_capability_cache_invalidates_when_binary_changes(tmp_path: Path) -> None:
    """Onbellek anahtari boyut+mtime iceriyor; ffmpeg guncellenirse dusmeli."""
    exe = tmp_path / f"ffmpeg{EXE}"
    exe.write_bytes(b"stub")
    cache = tmp_path / "caps.json"
    caps = Capabilities(
        version="N-1",
        build_flags=frozenset(),
        encoders=frozenset(),
        filters=frozenset({"aresample", "astats"}),
    )
    save_cached_capabilities(exe, caps, cache)

    exe.write_bytes(b"a different build entirely")
    assert load_cached_capabilities(exe, cache) is None


def test_capability_cache_survives_unreadable_file(tmp_path: Path) -> None:
    """Bozuk onbellek hata degil, sadece onbellek yok demektir."""
    exe = tmp_path / f"ffmpeg{EXE}"
    exe.write_bytes(b"stub")
    cache = tmp_path / "caps.json"
    cache.write_text("{ this is not json", encoding="utf-8")
    assert load_cached_capabilities(exe, cache) is None


@pytest.mark.needs_ffmpeg
def test_discovered_build_has_expected_capabilities(ffmpeg_tools: object) -> None:
    """Gercek makinede bulunan derleme analiz icin yeterli mi."""
    from app.core.ffmpeg_locate import FFmpegTools

    assert isinstance(ffmpeg_tools, FFmpegTools)
    assert ffmpeg_tools.ffprobe.is_file()
    assert not ffmpeg_tools.caps.missing_required()
    assert ffmpeg_tools.caps.version
