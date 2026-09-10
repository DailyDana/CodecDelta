"""Opus bit akisi testleri.

TOC cozumu RFC 6716 bolum 3.1'e gore elle dogrulanmis config degerleriyle
sinaniyor; baslik paketleri testin icinde kuruluyor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.test_ogg import make_page

from app.bitstream import opus
from app.bitstream.opus import decode_toc, parse_head, parse_tags
from app.core.ffmpeg_locate import FFmpegTools


def toc_byte(config: int, *, stereo: bool = True, count_code: int = 0) -> int:
    return (config << 3) | (int(stereo) << 2) | count_code


def make_head(
    *,
    channels: int = 2,
    pre_skip: int = 312,
    input_rate: int = 48000,
    gain_q78: int = 0,
    family: int = 0,
) -> bytes:
    return (
        b"OpusHead"
        + bytes([1, channels])
        + pre_skip.to_bytes(2, "little")
        + input_rate.to_bytes(4, "little")
        + gain_q78.to_bytes(2, "little", signed=True)
        + bytes([family])
    )


def make_tags(vendor: str, comments: list[str]) -> bytes:
    out = bytearray(b"OpusTags")
    raw = vendor.encode()
    out += len(raw).to_bytes(4, "little") + raw
    out += len(comments).to_bytes(4, "little")
    for c in comments:
        data = c.encode()
        out += len(data).to_bytes(4, "little") + data
    return bytes(out)


# -- TOC --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "mode", "bandwidth", "frame_ms"),
    [
        (0, "SILK", "NB", 10.0),
        (3, "SILK", "NB", 60.0),
        (8, "SILK", "WB", 10.0),
        (12, "Hybrid", "SWB", 10.0),
        (15, "Hybrid", "FB", 20.0),
        (16, "CELT", "NB", 2.5),
        (20, "CELT", "WB", 2.5),
        (28, "CELT", "FB", 2.5),
        (31, "CELT", "FB", 20.0),
    ],
)
def test_config_table_matches_rfc(config: int, mode: str, bandwidth: str, frame_ms: float) -> None:
    toc = decode_toc(bytes([toc_byte(config)]))
    assert toc is not None
    assert (toc.mode, toc.bandwidth, toc.frame_ms) == (mode, bandwidth, frame_ms)


def test_celt_has_no_medium_band() -> None:
    """CELT'te MB yoktur; tabloda o satirin atlanmis olmasi bilincli."""
    celt_bandwidths = {decode_toc(bytes([toc_byte(c)])).bandwidth for c in range(16, 32)}  # type: ignore[union-attr]
    assert "MB" not in celt_bandwidths
    assert celt_bandwidths == {"NB", "WB", "SWB", "FB"}


def test_stereo_flag() -> None:
    assert decode_toc(bytes([toc_byte(31, stereo=True)])).stereo is True  # type: ignore[union-attr]
    assert decode_toc(bytes([toc_byte(31, stereo=False)])).stereo is False  # type: ignore[union-attr]


def test_frame_count_codes() -> None:
    assert decode_toc(bytes([toc_byte(31, count_code=0)])).frames == 1  # type: ignore[union-attr]
    assert decode_toc(bytes([toc_byte(31, count_code=1)])).frames == 2  # type: ignore[union-attr]
    assert decode_toc(bytes([toc_byte(31, count_code=2)])).frames == 2  # type: ignore[union-attr]
    # Kod 3: sayi ikinci baytin alt 6 bitinde
    toc = decode_toc(bytes([toc_byte(31, count_code=3), 0x03]))
    assert toc is not None
    assert toc.frames == 3
    assert toc.duration_ms == 60.0


def test_frame_count_code_three_ignores_upper_flag_bits() -> None:
    """Ust iki bit VBR ve dolgu bayragidir, cerceve sayisina karismamali."""
    toc = decode_toc(bytes([toc_byte(31, count_code=3), 0xC3]))
    assert toc is not None
    assert toc.frames == 3


def test_undecodable_packets_return_none() -> None:
    assert decode_toc(b"") is None
    # Kod 3 ama ikinci bayt yok
    assert decode_toc(bytes([toc_byte(31, count_code=3)])) is None
    # Kod 3 ve sifir cerceve
    assert decode_toc(bytes([toc_byte(31, count_code=3), 0x00])) is None


# -- baslik paketleri -------------------------------------------------------


def test_parse_head_reads_pre_skip_and_family() -> None:
    head = parse_head(make_head(pre_skip=312))
    assert head is not None
    assert head.channels == 2
    assert head.pre_skip == 312
    assert head.input_sample_rate == 48000
    assert head.mapping_family == 0
    assert head.coupled_count == 1


def test_parse_head_gain_is_signed_q78() -> None:
    """Cikis kazanci ISARETLI Q7.8; isaretsiz okunursa -6 dB, +250 dB gorunur."""
    head = parse_head(make_head(gain_q78=-1536))
    assert head is not None
    assert head.output_gain_db == pytest.approx(-6.0)


def test_parse_head_rejects_foreign_packet() -> None:
    assert parse_head(b"NotOpusHead....") is None
    assert parse_head(b"OpusHead") is None


def test_parse_tags_reads_vendor_and_comments() -> None:
    vendor, comments = parse_tags(
        make_tags("Lavf62.12.102", ["TITLE=Song", "artist=Someone", "no-equals-sign"])
    )
    assert vendor == "Lavf62.12.102"
    assert comments["TITLE"] == "Song"
    # Anahtarlar buyuk harfe normalize edilir
    assert comments["ARTIST"] == "Someone"
    # "=" icermeyen girdi yok sayilir
    assert len(comments) == 2


def test_parse_tags_survives_truncation() -> None:
    """Kirpilmis bir etiket paketi cokme degil, eksik sozluk uretmeli."""
    raw = make_tags("vendor", ["A=1", "B=2"])
    vendor, comments = parse_tags(raw[:-4])
    assert vendor == "vendor"
    assert comments.get("A") == "1"


def test_parse_tags_ignores_absurd_comment_count() -> None:
    """Bozuk bir sayac milyonlarca donguye yol acmamali."""
    raw = bytearray(make_tags("v", []))
    raw[8 + 4 + 1 : 8 + 4 + 1 + 4] = (2**31).to_bytes(4, "little")
    vendor, comments = parse_tags(bytes(raw))
    assert vendor == "v"
    assert comments == {}


# -- tam tarama -------------------------------------------------------------


def _opus_stream(packets: list[bytes], *, granule: int) -> bytes:
    pages = [
        make_page([make_head()], sequence=0),
        make_page([make_tags("libopus 1.5", ["TITLE=T"])], sequence=1),
    ]
    for i, packet in enumerate(packets):
        pages.append(make_page([packet], sequence=2 + i, granule=granule * (i + 1)))
    return b"".join(pages)


def test_scan_synthetic_stream(tmp_path: Path) -> None:
    path = tmp_path / "s.opus"
    # 20 ms'lik CELT/FB paketleri, degisken boyutlu -> VBR
    packets = [bytes([toc_byte(31)]) + b"\x00" * (100 + i) for i in range(10)]
    # granule 48 kHz ornek cinsinden: 20 ms = 960 ornek
    path.write_bytes(_opus_stream(packets, granule=960))

    info = opus.scan(path)
    assert info is not None
    assert info.head.pre_skip == 312
    assert info.vendor == "libopus 1.5"
    assert info.comments["TITLE"] == "T"
    assert info.packets == 10
    assert info.undecodable_packets == 0
    assert info.dominant_mode == "CELT"
    assert info.dominant_bandwidth == "FB"
    assert info.nominal_bandwidth_hz == 20000
    assert info.is_vbr is True
    assert info.total_duration_ms == pytest.approx(200.0)


def test_scan_detects_cbr(tmp_path: Path) -> None:
    path = tmp_path / "cbr.opus"
    packets = [bytes([toc_byte(31)]) + b"\x00" * 100 for _ in range(8)]
    path.write_bytes(_opus_stream(packets, granule=960))
    info = opus.scan(path)
    assert info is not None
    assert info.is_vbr is False


def test_granule_duration_subtracts_pre_skip(tmp_path: Path) -> None:
    """Granule pre-skip'i icerir; cikarilmazsa sure birkac ms uzun gorunur."""
    path = tmp_path / "g.opus"
    packets = [bytes([toc_byte(31)]) + b"\x00" * 100]
    # Tek sayfa, granule 48000 -> pre_skip 312 dusulunce 0.99350 s
    path.write_bytes(_opus_stream(packets, granule=48000))
    info = opus.scan(path)
    assert info is not None
    assert info.granule_duration_s == pytest.approx((48000 - 312) / 48000)


def test_scan_rejects_non_opus(tmp_path: Path) -> None:
    path = tmp_path / "not.ogg"
    path.write_bytes(make_page([b"\x01vorbis and then some"]))
    assert opus.scan(path) is None


@pytest.mark.needs_ffmpeg
def test_scan_real_libopus_file(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Gercek bir libopus ciktisinda TOC istatistikleri tutarli olmali."""
    if not ffmpeg_tools.caps.has_encoder("libopus"):
        pytest.skip("bu ffmpeg derlemesinde libopus yok")
    media = tmp_path / "tone.opus"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=48000:duration=5",
            "-ac",
            "2",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            str(media),
        ],
        check=True,
        capture_output=True,
    )

    info = opus.scan(media)
    assert info is not None
    assert info.head.channels == 2
    assert info.head.pre_skip > 0
    # Vendor kodlayiciyi DEGIL paketleyiciyi soyler: bu dosya libopus ile
    # kodlandi ama Ogg'u ffmpeg'in kendi muxer'i yazdigi icin imza "Lavf*".
    # Yani vendor "kim kodladi" sorusunu cevaplamaz; YouTube'dan indirilen
    # dosyalarda da ayni sekilde orijinal imza kaybolur.
    assert info.vendor.startswith("Lavf")
    assert info.dominant_mode == "CELT"
    assert info.dominant_bandwidth == "FB"
    # libopus varsayilani 20 ms cerceve
    assert info.frame_ms_hist.most_common(1)[0][0] == 20.0
    assert info.granule_duration_s == pytest.approx(5.0, abs=0.05)
    assert info.bad_crc_pages == 0

    # Bitrate hedefe DEGIL, konteynere gore dogrulanir. `-b:a 128k` bir hedef,
    # tavan degil: bu build'de saf 1 kHz sinus 170 kbps, pembe gurultu 88 kbps
    # uretti. (Ters gibi gorunur ama degil -- saf ton bir donusum kodeki icin
    # pahalidir, tek bini yuksek hassasiyetle tasimak gerekir; gurultu ucuzdur.)
    #
    # Gercek degismez: paket verisi konteynerin biraz ALTINDA olmali, cunku
    # Ogg sayfa basliklarini icermez. Olculen fark ~1.9 kbps.
    duration = info.granule_duration_s
    assert duration is not None
    container_kbps = media.stat().st_size * 8 / duration / 1000
    assert info.kbps < container_kbps
    assert info.kbps > container_kbps * 0.95
