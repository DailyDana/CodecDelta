"""FLAC metadata testleri."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.bitstream import flac
from app.bitstream.flac import parse_stream_info, parse_vorbis_comment
from app.core.ffmpeg_locate import FFmpegTools


def pack_stream_info(
    *,
    min_bs: int = 4096,
    max_bs: int = 4096,
    min_fs: int = 1000,
    max_fs: int = 14000,
    rate: int = 44100,
    channels: int = 2,
    bps: int = 16,
    total: int = 25090269,
    md5: bytes = b"\x11" * 16,
) -> bytes:
    """34 baytlik STREAMINFO govdesini elle kurar.

    Bit yerlesimi testte bagimsiz olarak yazildi: 20 bit hiz, 3 bit kanal-1,
    5 bit bps-1, 36 bit toplam ornek. Ayristirici bunu tersinden okumali.
    """
    packed = (rate << 44) | ((channels - 1) << 41) | ((bps - 1) << 36) | total
    return (
        min_bs.to_bytes(2, "big")
        + max_bs.to_bytes(2, "big")
        + min_fs.to_bytes(3, "big")
        + max_fs.to_bytes(3, "big")
        + packed.to_bytes(8, "big")
        + md5
    )


def make_block(block_type: int, body: bytes, *, last: bool = False) -> bytes:
    head = bytes([(0x80 if last else 0) | block_type]) + len(body).to_bytes(3, "big")
    return head + body


def make_vorbis_comment(vendor: str, comments: list[str]) -> bytes:
    out = bytearray()
    raw = vendor.encode()
    out += len(raw).to_bytes(4, "little") + raw
    out += len(comments).to_bytes(4, "little")
    for c in comments:
        data = c.encode()
        out += len(data).to_bytes(4, "little") + data
    return bytes(out)


def make_flac(
    *,
    stream_info: bytes | None = None,
    extra_blocks: list[bytes] | None = None,
    audio: bytes = b"\xff" * 1000,
    id3: bytes = b"",
) -> bytes:
    blocks = [make_block(flac.BLOCK_STREAMINFO, stream_info or pack_stream_info())]
    blocks += extra_blocks or []
    # Son blogun "last" bayragi kalkik olmali
    body = bytearray(b"".join(blocks))
    header_pos = 0
    positions = []
    for block in blocks:
        positions.append(header_pos)
        header_pos += len(block)
    body[positions[-1]] |= 0x80
    return id3 + flac.MAGIC + bytes(body) + audio


# -- STREAMINFO -------------------------------------------------------------


def test_stream_info_bit_layout() -> None:
    info = parse_stream_info(pack_stream_info(rate=44100, channels=2, bps=16, total=25090269))
    assert info is not None
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.bits_per_sample == 16
    assert info.total_samples == 25090269
    assert info.duration_s == pytest.approx(568.940, abs=1e-3)


def test_stream_info_handles_high_resolution() -> None:
    """24/192 mono da dogru cozulmeli; alanlar bitisik paketlenmis."""
    info = parse_stream_info(pack_stream_info(rate=192000, channels=1, bps=24, total=1))
    assert info is not None
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (192000, 1, 24)


def test_stream_info_rejects_short_block() -> None:
    assert parse_stream_info(b"\x00" * 20) is None


def test_missing_md5_is_reported_as_unknown() -> None:
    """Sifir MD5 "bozuk" degil "yazilmamis" demektir."""
    info = parse_stream_info(pack_stream_info(md5=b"\x00" * 16))
    assert info is not None
    assert info.md5_present is False


def test_raw_pcm_bytes() -> None:
    info = parse_stream_info(pack_stream_info(total=25090269, channels=2, bps=16))
    assert info is not None
    assert info.raw_pcm_bytes == 25090269 * 2 * 2


# -- VORBIS_COMMENT ---------------------------------------------------------


def test_vorbis_comment_is_little_endian() -> None:
    """FLAC big endian ama bu blok little endian; kolay atlanan ayrinti."""
    vendor, comments = parse_vorbis_comment(
        make_vorbis_comment("reference libFLAC 1.4.3", ["TITLE=X", "date=2006"])
    )
    assert vendor == "reference libFLAC 1.4.3"
    assert comments["TITLE"] == "X"
    assert comments["DATE"] == "2006"


def test_vorbis_comment_survives_truncation() -> None:
    raw = make_vorbis_comment("v", ["A=1", "B=2"])
    vendor, comments = parse_vorbis_comment(raw[:-3])
    assert vendor == "v"
    assert comments.get("A") == "1"


# -- tam tarama -------------------------------------------------------------


def test_scan_reads_blocks_and_vendor(tmp_path: Path) -> None:
    path = tmp_path / "a.flac"
    path.write_bytes(
        make_flac(
            extra_blocks=[
                make_block(flac.BLOCK_VORBIS_COMMENT, make_vorbis_comment("libFLAC", ["A=1"])),
                make_block(flac.BLOCK_PADDING, b"\x00" * 512),
            ]
        )
    )
    info = flac.scan(path)
    assert info is not None
    assert info.vendor == "libFLAC"
    assert info.comments["A"] == "1"
    assert info.has_block["PADDING"] is True
    assert info.has_block["SEEKTABLE"] is False


def test_scan_skips_id3_tag(tmp_path: Path) -> None:
    """Basta ID3v2 varsa "fLaC" imzasi kaymis olur; atlanmazsa dosya
    FLAC degil sanilir. Boyut alani syncsafe (her baytin alt 7 biti)."""
    payload = b"\x00" * 100
    size = len(payload)
    syncsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    id3 = b"ID3\x04\x00\x00" + syncsafe + payload

    path = tmp_path / "tagged.flac"
    path.write_bytes(make_flac(id3=id3))
    info = flac.scan(path)
    assert info is not None
    assert info.id3_bytes == 10 + size


def test_scan_rejects_non_flac(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"RIFFxxxxWAVEfmt ")
    assert flac.scan(path) is None


def test_compression_ratio_excludes_embedded_picture(tmp_path: Path) -> None:
    """Kapak resmi orana karismamali.

    Gercek dosyada 296 KB'lik kapak orani ~%0.5 kaydiriyordu; buradaki abartili
    kapak farki gorunur kiliyor.
    """
    total = 100_000
    audio = b"\xaa" * 50_000
    picture = make_block(flac.BLOCK_PICTURE, b"\x00" * 200_000)

    plain = tmp_path / "plain.flac"
    plain.write_bytes(make_flac(stream_info=pack_stream_info(total=total), audio=audio))
    with_pic = tmp_path / "pic.flac"
    with_pic.write_bytes(
        make_flac(stream_info=pack_stream_info(total=total), extra_blocks=[picture], audio=audio)
    )

    a = flac.scan(plain)
    b = flac.scan(with_pic)
    assert a is not None and b is not None
    assert a.audio_bytes == b.audio_bytes == len(audio)
    assert a.compression_ratio == b.compression_ratio


def test_encoder_family_comes_from_vendor(tmp_path: Path) -> None:
    """Kodlayiciyi vendor belirler, blok boyutu DEGIL."""

    def scan_with(vendor: str) -> str | None:
        path = tmp_path / f"{abs(hash(vendor))}.flac"
        path.write_bytes(
            make_flac(
                extra_blocks=[
                    make_block(flac.BLOCK_VORBIS_COMMENT, make_vorbis_comment(vendor, []))
                ]
            )
        )
        info = flac.scan(path)
        assert info is not None
        return info.encoder_family

    assert scan_with("reference libFLAC 1.4.3 20230623") == "reference libFLAC"
    assert scan_with("Lavf58.45.100") == "ffmpeg"
    assert scan_with("SomeOtherEncoder 2.0") == "SomeOtherEncoder 2.0"
    assert scan_with("") is None


def test_blocksize_is_data_not_a_fingerprint(tmp_path: Path) -> None:
    """Blok boyutundan kodlayici cikarilmaz -- olculdu.

    Ayni ffmpeg derlemesi lavfi kaynagindan 1024, wav dosyasindan 4096,
    -frame_size 4608 ile 4608 uretti. 4096 "referans libFLAC varsayilani"
    olarak bilinen degerdir; boyuta bakan bir kural modern ffmpeg ciktilarini
    "referans rip" diye etiketlerdi. Bu test o kuralin geri eklenmesini
    engellemek icin var.
    """
    path = tmp_path / "ff4096.flac"
    path.write_bytes(
        make_flac(
            stream_info=pack_stream_info(min_bs=4096, max_bs=4096),
            extra_blocks=[
                make_block(flac.BLOCK_VORBIS_COMMENT, make_vorbis_comment("Lavf63.5.101", []))
            ],
        )
    )
    info = flac.scan(path)
    assert info is not None
    # 4096 blok boyutuna ragmen kodlayici ffmpeg olarak dogru tespit edilmeli
    assert info.stream_info.max_blocksize == 4096
    assert info.encoder_family == "ffmpeg"


# -- gercek ffmpeg ----------------------------------------------------------


def _encode(tools: FFmpegTools, source: str, out: Path, *args: str) -> None:
    subprocess.run(
        [
            str(tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-ac",
            "2",
            *args,
            str(out),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.needs_ffmpeg
def test_scan_and_verify_real_flac(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    media = tmp_path / "tone.flac"
    _encode(ffmpeg_tools, "sine=frequency=440:sample_rate=44100:duration=3", media)

    info = flac.scan(media)
    assert info is not None
    si = info.stream_info
    assert (si.sample_rate, si.channels, si.bits_per_sample) == (44100, 2, 16)
    assert si.total_samples == 3 * 44100
    assert si.md5_present
    assert info.vendor.startswith("Lavf")
    assert info.encoder_family == "ffmpeg"
    assert flac.verify_md5(ffmpeg_tools.ffmpeg, media, si) is True


@pytest.mark.needs_ffmpeg
def test_verify_md5_skipped_when_absent(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """MD5 yoksa sonuc False degil None: dogrulanamadi, basarisiz degil."""
    media = tmp_path / "tone.flac"
    _encode(ffmpeg_tools, "sine=frequency=440:sample_rate=44100:duration=1", media)
    info = flac.scan(media)
    assert info is not None
    blank = type(info.stream_info)(
        **{**info.stream_info.__dict__, "md5": b"\x00" * 16}  # type: ignore[arg-type]
    )
    assert flac.verify_md5(ffmpeg_tools.ffmpeg, media, blank) is None


@pytest.mark.needs_ffmpeg
def test_lossy_source_compresses_better_than_real_music(
    ffmpeg_tools: FFmpegTools, tmp_path: Path
) -> None:
    """Kayipli kaynaktan gelen FLAC daha iyi sikisir.

    Mekanizma: kayipli kodlayici yuksek frekanslari siler, silinen bant
    neredeyse sifir olur ve FLAC bunu cok ucuza saklar. Burada bir bant
    sinirlamasi ile taklit ediliyor -- gercek bir MP3 zinciri de ayni yone
    iter. Bu test mekanizmayi kilitliyor, mutlak esikleri DEGIL: gercek
    esikler kullanicinin kendi kutuphanesiyle kalibre edilmeli.
    """
    if not ffmpeg_tools.caps.has_filter("lowpass"):
        pytest.skip("bu ffmpeg derlemesinde lowpass filtresi yok")

    full = tmp_path / "full.flac"
    limited = tmp_path / "limited.flac"
    noise = "anoisesrc=color=pink:sample_rate=44100:duration=5"
    _encode(ffmpeg_tools, noise, full)
    _encode(ffmpeg_tools, noise, limited, "-af", "lowpass=f=15000")

    a = flac.scan(full)
    b = flac.scan(limited)
    assert a is not None and b is not None
    assert a.compression_ratio is not None and b.compression_ratio is not None
    assert b.compression_ratio < a.compression_ratio
