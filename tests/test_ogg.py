"""Ogg sayfa/paket okuyucu testleri.

Sayfalar test icinde sifirdan kuruluyor. Bu, CRC uygulamasini iki yonlu
sinar: yazarken hesaplanan deger okurken dogrulanmali, ve bilerek bozulan
bir bayt yakalanmali.
"""

from __future__ import annotations

from pathlib import Path

from app.bitstream import ogg
from app.bitstream.ogg import iter_packets, iter_pages, page_crc, parse_page


def lacing(packets: list[bytes]) -> bytes:
    """Paket uzunluklarini Ogg segment tablosuna cevirir.

    255'lik her segment "paket devam ediyor" demektir; 255'ten kisa olan
    paketi bitirir. Tam 255'in kati uzunluktaki bir paket bu yuzden sonuna
    bir sifir segment almak zorundadir.
    """
    out = bytearray()
    for packet in packets:
        n = len(packet)
        out += b"\xff" * (n // 255)
        out.append(n % 255)
    return bytes(out)


def make_page(
    packets: list[bytes],
    *,
    serial: int = 1,
    sequence: int = 0,
    granule: int = 0,
    flags: int = 0,
    segments: bytes | None = None,
    body: bytes | None = None,
) -> bytes:
    """Gecerli CRC'li bir Ogg sayfasi kurar.

    `segments`/`body` verilirse paket listesi yerine onlar kullanilir; sayfa
    sinirini asan paketleri elle kurabilmek icin.
    """
    table = segments if segments is not None else lacing(packets)
    payload = body if body is not None else b"".join(packets)
    header = bytearray(b"OggS")
    header.append(0)  # surum
    header.append(flags)
    header += granule.to_bytes(8, "little")
    header += serial.to_bytes(4, "little")
    header += sequence.to_bytes(4, "little")
    header += b"\x00\x00\x00\x00"  # CRC alani once sifir
    header.append(len(table))
    page = bytearray(header + table + payload)
    page[22:26] = page_crc(bytes(page)).to_bytes(4, "little")
    return bytes(page)


def test_roundtrip_single_page() -> None:
    raw = make_page([b"hello", b"world"], granule=960, serial=7, sequence=3)
    page = parse_page(raw)
    assert page is not None
    assert page.crc_ok is True
    assert page.granule == 960
    assert page.serial == 7
    assert page.sequence == 3
    assert page.total_size == len(raw)
    assert list(iter_packets(iter_pages(raw))) == [b"hello", b"world"]


def test_corrupted_byte_is_detected() -> None:
    raw = bytearray(make_page([b"payload"]))
    raw[-1] ^= 0xFF
    page = parse_page(bytes(raw))
    assert page is not None
    assert page.crc_ok is False


def test_crc_can_be_skipped() -> None:
    """Atlanan CRC "saglam" degil, "bilinmiyor" anlamina gelir."""
    raw = make_page([b"payload"])
    page = parse_page(raw, verify_crc=False)
    assert page is not None
    assert page.crc_ok is None


def test_truncated_page_returns_none() -> None:
    raw = make_page([b"payload"])
    assert parse_page(raw[:-3]) is None
    assert parse_page(raw[:10]) is None


def test_non_ogg_data_returns_none() -> None:
    assert parse_page(b"not an ogg page at all") is None


def test_packet_spanning_two_pages() -> None:
    """Sayfa sinirini asan paket yeniden birlestirilmeli.

    255 uzunlugundaki bir segment paketin devam ettigini soyler; ikinci sayfa
    CONTINUED bayragiyla gelir ve iki parca tek pakete birlesir.
    """
    part_a = bytes(range(256)) * 2  # 512 bayt
    first = make_page([], segments=b"\xff\xff", body=part_a[:510])
    second = make_page(
        [],
        segments=bytes([2]),
        body=part_a[510:],
        flags=ogg.FLAG_CONTINUED,
        sequence=1,
    )
    packets = list(iter_packets(iter_pages(first + second)))
    assert packets == [part_a]


def test_packet_of_exact_multiple_of_255_needs_zero_segment() -> None:
    packet = b"x" * 255
    raw = make_page([packet])
    assert list(iter_packets(iter_pages(raw))) == [packet]


def test_leading_continuation_is_discarded() -> None:
    """Dosyanin ortasindan baslandiginda yarim paket atilmali.

    Yarim bir paketin ilk bayti TOC degildir; istatistige katilirsa mod ve
    bant histogramlarini sessizce kirletir.
    """
    orphan_tail = make_page([], segments=bytes([4]), body=b"tail", flags=ogg.FLAG_CONTINUED)
    good = make_page([b"whole"], sequence=1)
    assert list(iter_packets(iter_pages(orphan_tail + good))) == [b"whole"]


def test_trailing_incomplete_packet_is_discarded() -> None:
    """Kirpilmis dosyanin sonundaki yarim paket sayilmamali."""
    raw = make_page([], segments=b"\xff", body=b"y" * 255)
    assert list(iter_packets(iter_pages(raw))) == []


def test_find_page_skips_fake_magic_in_audio_data() -> None:
    """Ses verisinin icindeki "OggS" dizisi yanlis senkrona yol acmamali.

    Bu, ornekleme yapilan buyuk dosyalarda gercek bir risk: CRC olmadan
    yeniden senkron rastgele bir noktaya kilitlenebilir.
    """
    decoy = b"OggS" + b"\x00" * 40
    real = make_page([b"real packet"], sequence=9)
    found = ogg.find_page(decoy + real)
    assert found is not None
    assert found.offset == len(decoy)
    assert found.sequence == 9


def test_iter_pages_resyncs_after_garbage() -> None:
    good_a = make_page([b"aaa"], sequence=0)
    good_b = make_page([b"bbb"], sequence=1)
    stream = good_a + b"\x00" * 37 + good_b
    packets = list(iter_packets(iter_pages(stream)))
    assert packets == [b"aaa", b"bbb"]


def test_stats_track_pages_and_crc_coverage() -> None:
    pages = b"".join(make_page([f"p{i}".encode()], sequence=i, granule=i * 960) for i in range(5))
    stats = ogg.ScanStats()
    packets = list(iter_packets(iter_pages(pages), stats))
    assert len(packets) == 5
    assert stats.pages == 5
    assert stats.crc_checked_pages == 5  # ilk 16 sayfa her zaman dogrulanir
    assert stats.bad_crc_pages == 0
    assert stats.first_granule == 0
    assert stats.last_granule == 4 * 960
    assert stats.serials == {1}


def test_crc_sampling_reduces_coverage_on_long_streams() -> None:
    """Uzun akista her sayfa dogrulanmaz; rapor bunu bilmek zorunda."""
    pages = b"".join(make_page([b"x"], sequence=i) for i in range(200))
    stats = ogg.ScanStats()
    list(iter_packets(iter_pages(pages), stats))
    assert stats.pages == 200
    assert stats.crc_checked_pages < stats.pages
    assert stats.crc_checked_pages >= 16


def test_scan_packets_reads_whole_small_file(tmp_path: Path) -> None:
    path = tmp_path / "small.ogg"
    path.write_bytes(b"".join(make_page([b"packet"], sequence=i) for i in range(3)))
    packets, stats = ogg.scan_packets(path)
    assert packets == [b"packet"] * 3
    assert stats.sampled is False


def test_scan_packets_samples_large_file(tmp_path: Path) -> None:
    """Esik asilinca ornekleme devreye girer ve bayragi kaldirir."""
    path = tmp_path / "big.ogg"
    path.write_bytes(b"".join(make_page([b"packet"], sequence=i) for i in range(4000)))
    packets, stats = ogg.scan_packets(path, full_scan_limit=1024)
    assert stats.sampled is True
    assert packets  # ornek bolgelerden paket cikmis olmali
    assert stats.bytes_read <= path.stat().st_size
