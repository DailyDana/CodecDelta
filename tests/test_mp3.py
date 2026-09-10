"""MP3 cerceve basligi, Xing/Info ve LAME etiketi testleri."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.bitstream import mp3
from app.bitstream.mp3 import find_first_frame, parse_frame_header, parse_lame_tag
from app.core.ffmpeg_locate import FFmpegTools


def frame_header(
    *,
    version: int = 3,  # MPEG1
    layer_bits: int = 1,  # Layer III
    bitrate_index: int = 9,  # MPEG1 L3 -> 128 kbps
    rate_index: int = 0,  # 44100
    padding: bool = False,
    channel_mode: int = 0,  # stereo
    protected: bool = False,
) -> bytes:
    b1 = 0xE0 | (version << 3) | (layer_bits << 1) | (0 if protected else 1)
    b2 = (bitrate_index << 4) | (rate_index << 2) | (int(padding) << 1)
    b3 = channel_mode << 6
    return bytes([0xFF, b1, b2, b3])


def test_frame_header_fields() -> None:
    header = parse_frame_header(frame_header())
    assert header is not None
    assert header.is_layer3
    assert header.bitrate_kbps == 128
    assert header.sample_rate == 44100
    assert header.channel_mode == "stereo"
    assert header.samples_per_frame == 1152
    # MPEG1 L3: 144 * 128000 / 44100 = 417 (dolgu yok)
    assert header.frame_bytes == 417


def test_frame_size_includes_padding() -> None:
    plain = parse_frame_header(frame_header())
    padded = parse_frame_header(frame_header(padding=True))
    assert plain is not None and padded is not None
    assert padded.frame_bytes == plain.frame_bytes + 1


def test_mpeg2_uses_half_the_samples() -> None:
    """MPEG2/2.5 Layer III cercevesi 576 ornek tasir, 1152 degil."""
    header = parse_frame_header(frame_header(version=2, bitrate_index=8, rate_index=0))
    assert header is not None
    assert header.sample_rate == 22050
    assert header.samples_per_frame == 576
    assert header.bitrate_kbps == 64


def test_channel_modes() -> None:
    modes = [parse_frame_header(frame_header(channel_mode=m)) for m in range(4)]
    assert [m.channel_mode for m in modes] == [  # type: ignore[union-attr]
        "stereo",
        "joint stereo",
        "dual channel",
        "mono",
    ]
    assert modes[3].is_mono  # type: ignore[union-attr]
    assert modes[1].is_joint_stereo  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "bad",
    [
        frame_header(version=1),  # ayrilmis surum
        frame_header(layer_bits=0),  # ayrilmis layer
        frame_header(bitrate_index=0),  # "free" bitrate
        frame_header(bitrate_index=15),  # gecersiz
        frame_header(rate_index=3),  # gecersiz hiz
        b"\xff\x00\x00\x00",  # sync eksik
        b"\xff",  # kirpik
    ],
)
def test_invalid_headers_are_rejected(bad: bytes) -> None:
    """Sync deseni ses verisinde sik gecer; gecersiz kombinasyonlar elenmeli."""
    assert parse_frame_header(bad) is None


def test_find_first_frame_requires_a_following_frame() -> None:
    """Tek bir gecerli baslik yetmez: zincir kontrolu yanlis senkronu eler."""
    header = frame_header()
    size = 417
    lone = b"\x00" * 10 + header + b"\x00" * 50
    assert find_first_frame(lone) is None

    chained = b"\x00" * 10 + header + b"\x00" * (size - 4) + header + b"\x00" * (size - 4)
    found = find_first_frame(chained)
    assert found is not None
    assert found[0] == 10


# -- LAME etiketi -----------------------------------------------------------


def make_lame_tag(
    *,
    encoder: str = "LAME3.100",
    vbr_code: int = 4,
    lowpass_hz: int = 19000,
    delay: int = 576,
    padding: int = 1728,
    ath: int = 4,
    abr: int = 0,
) -> bytes:
    body = bytearray(36)
    body[0:9] = encoder.encode("latin-1").ljust(9, b"\x00")
    body[9] = vbr_code
    body[10] = lowpass_hz // 100
    body[19] = ath
    body[20] = abr
    body[21:24] = ((delay << 12) | padding).to_bytes(3, "big")
    return bytes(body)


def test_lame_tag_fields() -> None:
    """Gercek bir LAME etiketi; ffmpeg bunlari yazmadigi icin sentetik.

    Alcak geciren 100 Hz biriminde saklanir: 19000 Hz -> 190.
    """
    tag = parse_lame_tag(make_lame_tag(), 0)
    assert tag is not None
    assert tag.encoder == "LAME3.100"
    assert tag.is_lame
    assert tag.vbr_method == "VBR (new/mtrh)"
    assert tag.lowpass_hz == 19000
    assert tag.encoder_delay == 576
    assert tag.encoder_padding == 1728
    assert tag.ath_type == 4


def test_lowpass_zero_means_unknown() -> None:
    """Sifir "0 Hz'de kesilmis" degil "yazilmamis" demektir."""
    tag = parse_lame_tag(make_lame_tag(lowpass_hz=0), 0)
    assert tag is not None
    assert tag.lowpass_hz is None


def test_vbr_method_names() -> None:
    assert parse_lame_tag(make_lame_tag(vbr_code=1), 0).vbr_method == "CBR"  # type: ignore[union-attr]
    assert parse_lame_tag(make_lame_tag(vbr_code=2), 0).vbr_method == "ABR"  # type: ignore[union-attr]
    assert "unknown" in parse_lame_tag(make_lame_tag(vbr_code=7), 0).vbr_method  # type: ignore[union-attr]


def test_lame_tag_rejects_garbage() -> None:
    assert parse_lame_tag(b"\x00" * 36, 0) is None
    assert parse_lame_tag(b"\x00" * 10, 0) is None


def test_ffmpeg_encoder_string_is_not_lame() -> None:
    """ffmpeg "Lavc*" yazar; LAME'e ozgu alanlari doldurmadigi icin ayrim onemli."""
    tag = parse_lame_tag(make_lame_tag(encoder="Lavc63.7.", vbr_code=0, lowpass_hz=0), 0)
    assert tag is not None
    assert tag.is_lame is False
    assert tag.lowpass_hz is None


# -- gercek ffmpeg ----------------------------------------------------------


def _encode(tools: FFmpegTools, out: Path, *args: str, duration: int = 6) -> None:
    subprocess.run(
        [
            str(tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=color=pink:sample_rate=44100:duration={duration}",
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            *args,
            str(out),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.needs_ffmpeg
def test_cbr_file(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    if not ffmpeg_tools.caps.has_encoder("libmp3lame"):
        pytest.skip("bu ffmpeg derlemesinde libmp3lame yok")
    media = tmp_path / "cbr.mp3"
    _encode(ffmpeg_tools, media, "-b:a", "128k")

    info = mp3.scan(media)
    assert info is not None
    assert info.first_frame.sample_rate == 44100
    assert info.nominal_bitrate_kbps == 128
    assert info.is_vbr is False
    assert info.xing is not None
    # CBR dosyada blok "Info" olur, "Xing" degil
    assert info.xing.magic == "Info"
    assert info.duration_s == pytest.approx(6.0, abs=0.1)


@pytest.mark.needs_ffmpeg
def test_vbr_file(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    if not ffmpeg_tools.caps.has_encoder("libmp3lame"):
        pytest.skip("bu ffmpeg derlemesinde libmp3lame yok")
    media = tmp_path / "vbr.mp3"
    _encode(ffmpeg_tools, media, "-q:a", "0")

    info = mp3.scan(media)
    assert info is not None
    assert info.xing is not None
    assert info.xing.magic == "Xing"
    assert info.is_vbr is True
    average = info.average_bitrate_kbps
    assert average is not None
    # V0 nominal olarak ilk cercevenin bitrate'inden cok daha yuksek surer
    assert average > info.nominal_bitrate_kbps


@pytest.mark.needs_ffmpeg
def test_ffmpeg_leaves_lame_fields_empty(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Olculen sinir: ffmpeg gercek bir LAME etiketi yazmaz.

    Etiket dogru yerde ve encoder dizesi ile gecikme/dolgu okunabiliyor, ama
    alcak geciren ve VBR yontemi alanlari sifir. Bu yuzden ffmpeg ciktisinda
    kesim frekansi yalnizca OLCUMLE bulunabilir.
    """
    if not ffmpeg_tools.caps.has_encoder("libmp3lame"):
        pytest.skip("bu ffmpeg derlemesinde libmp3lame yok")
    media = tmp_path / "x.mp3"
    _encode(ffmpeg_tools, media, "-b:a", "192k")

    info = mp3.scan(media)
    assert info is not None
    assert info.xing is not None and info.xing.lame is not None
    lame = info.xing.lame
    assert lame.encoder.startswith("Lavc")
    assert lame.is_lame is False
    # Gecikme/dolgu dogru okunuyor -- ofset zincirinin dogrulugunun kaniti
    assert lame.encoder_delay == 576
    assert lame.encoder_padding > 0
    # Ama LAME'e ozgu alanlar bos
    assert lame.lowpass_hz is None
    assert info.declared_lowpass_hz is None


@pytest.mark.needs_ffmpeg
def test_mono_side_info_offset(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Xing blogunun yeri kanal sayisina gore degisir; mono yolu da tutmali."""
    if not ffmpeg_tools.caps.has_encoder("libmp3lame"):
        pytest.skip("bu ffmpeg derlemesinde libmp3lame yok")
    media = tmp_path / "mono.mp3"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=3",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(media),
        ],
        check=True,
        capture_output=True,
    )
    info = mp3.scan(media)
    assert info is not None
    assert info.first_frame.is_mono
    assert info.xing is not None
    assert info.duration_s == pytest.approx(3.0, abs=0.1)


@pytest.mark.needs_ffmpeg
def test_scan_rejects_non_mp3(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    media = tmp_path / "tone.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1",
            str(media),
        ],
        check=True,
        capture_output=True,
    )
    assert mp3.scan(media) is None
