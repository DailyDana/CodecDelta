"""Ogg Vorbis baslik testleri."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.test_ogg import make_page

from app.bitstream import vorbis
from app.bitstream.vorbis import parse_comment, parse_identification
from app.core.ffmpeg_locate import FFmpegTools


def make_identification(
    *,
    channels: int = 2,
    rate: int = 44100,
    nominal: int = 128000,
    maximum: int = -1,
    minimum: int = -1,
    short_exp: int = 8,
    long_exp: int = 11,
) -> bytes:
    return (
        bytes([1])
        + b"vorbis"
        + (0).to_bytes(4, "little")
        + bytes([channels])
        + rate.to_bytes(4, "little")
        + maximum.to_bytes(4, "little", signed=True)
        + nominal.to_bytes(4, "little", signed=True)
        + minimum.to_bytes(4, "little", signed=True)
        + bytes([(long_exp << 4) | short_exp])
        + bytes([1])  # framing
    )


def make_comment(vendor: str, comments: list[str]) -> bytes:
    out = bytearray(bytes([3]) + b"vorbis")
    raw = vendor.encode()
    out += len(raw).to_bytes(4, "little") + raw
    out += len(comments).to_bytes(4, "little")
    for c in comments:
        data = c.encode()
        out += len(data).to_bytes(4, "little") + data
    out += bytes([1])  # framing
    return bytes(out)


def test_identification_fields() -> None:
    info = parse_identification(make_identification())
    assert info is not None
    assert info.channels == 2
    assert info.sample_rate == 44100
    assert info.bitrate_nominal == 128000
    assert info.nominal_kbps == 128.0
    # Blok boyutlari us olarak saklanir: 2^8 ve 2^11
    assert info.blocksize_short == 256
    assert info.blocksize_long == 2048


def test_unset_bitrate_is_signed_minus_one() -> None:
    """Bitrate alanlari isaretlidir; -1 "belirtilmemis" demektir.

    Isaretsiz okunursa 4294967295 gibi absurt bir deger cikar.
    """
    info = parse_identification(make_identification(nominal=-1))
    assert info is not None
    assert info.bitrate_nominal == -1
    assert info.nominal_kbps is None


def test_identification_rejects_foreign_packet() -> None:
    assert parse_identification(b"\x01opus" + b"\x00" * 40) is None
    assert parse_identification(b"\x01vorbis") is None


def test_comment_header() -> None:
    vendor, comments = parse_comment(make_comment("Xiph.Org libVorbis I", ["TITLE=T", "a=b"]))
    assert vendor == "Xiph.Org libVorbis I"
    assert comments["TITLE"] == "T"
    assert comments["A"] == "b"


def test_comment_rejects_wrong_packet_type() -> None:
    assert parse_comment(make_identification()) == ("", {})


def test_scan_synthetic(tmp_path: Path) -> None:
    path = tmp_path / "s.ogg"
    pages = [
        make_page([make_identification()], sequence=0),
        make_page([make_comment("libVorbis", ["A=1"])], sequence=1),
        make_page([bytes([5]) + b"vorbis" + b"\x00" * 50], sequence=2),
    ]
    for i in range(4):
        pages.append(make_page([b"\x00" * 200], sequence=3 + i, granule=44100 * (i + 1)))
    path.write_bytes(b"".join(pages))

    info = vorbis.scan(path)
    assert info is not None
    assert info.identification.sample_rate == 44100
    assert info.vendor == "libVorbis"
    assert info.comments["A"] == "1"
    # Ilk uc paket basliktir, geri kalani ses
    assert info.audio_packets == 4
    assert info.duration_s == pytest.approx(4.0)


def test_scan_rejects_opus(tmp_path: Path) -> None:
    path = tmp_path / "x.opus"
    path.write_bytes(make_page([b"OpusHead" + b"\x00" * 11]))
    assert vorbis.scan(path) is None


@pytest.mark.needs_ffmpeg
def test_scan_real_vorbis(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    if not ffmpeg_tools.caps.has_encoder("libvorbis"):
        pytest.skip("bu ffmpeg derlemesinde libvorbis yok")
    media = tmp_path / "tone.ogg"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=5",
            "-ac",
            "2",
            "-c:a",
            "libvorbis",
            "-b:a",
            "128k",
            str(media),
        ],
        check=True,
        capture_output=True,
    )

    info = vorbis.scan(media)
    assert info is not None
    assert info.identification.channels == 2
    assert info.identification.sample_rate == 44100
    assert info.vendor  # libVorbis kendi imzasini yazar
    assert info.duration_s == pytest.approx(5.0, abs=0.1)
    assert info.audio_packets > 0
    assert info.bad_crc_pages == 0
    # Olculen bitrate konteynerin altinda kalmali (sayfa basliklari haric)
    container_kbps = media.stat().st_size * 8 / info.duration_s / 1000
    assert info.kbps is not None
    assert info.kbps < container_kbps
