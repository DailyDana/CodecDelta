"""Ogg Vorbis baslik paketleri.

Opus'un aksine Vorbis'te paket basina mod okunamaz: mod numarasi paketin ilk
baytindadir ama kac bit oldugu setup basligindaki (codebook) mod sayisina
baglidir ve setup basligini cozmek Huffman kod kitaplarini cozmek demektir.
Bu, saf Python'da tasinmaya degmeyecek bir maliyet -- ve karsiliginda elde
edilecek bilgi (blok boyutu degisimi) analiz icin Opus'un mod/bant bilgisi
kadar degerli degil.

Bu yuzden burada baslik paketleri ve gercek bitrate okunur. Onemli alan
`bitrate_nominal`: kodlayicinin HEDEFI. Olculen bitrate ile karsilastirmak,
"bu dosya q5 ile mi kodlanmis" turu sorulara dayanak verir.

Referans: Vorbis I spesifikasyonu bolum 4.2 (identification header).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.bitstream.ogg import FULL_SCAN_LIMIT, ScanStats, scan_packets

IDENTIFICATION = 1
COMMENT = 3
SETUP = 5

MAGIC = b"vorbis"


@dataclass(frozen=True)
class VorbisIdentification:
    """Identification header (paket tipi 1)."""

    version: int
    channels: int
    sample_rate: int
    bitrate_maximum: int
    bitrate_nominal: int
    bitrate_minimum: int
    blocksize_short: int
    blocksize_long: int

    @property
    def nominal_kbps(self) -> float | None:
        """Kodlayicinin hedef bitrate'i. 0 veya negatif "belirtilmemis" demek.

        Kalite tabanli (`-q`) kodlamada bu alan sik sik dolu olur ama bir
        taahhut degildir; gercek bitrate paketlerden olculur.
        """
        return self.bitrate_nominal / 1000 if self.bitrate_nominal > 0 else None


def parse_identification(packet: bytes) -> VorbisIdentification | None:
    """Identification header'i cozer. Alanlar LITTLE endian."""
    if len(packet) < 30 or packet[0] != IDENTIFICATION or packet[1:7] != MAGIC:
        return None
    blocksizes = packet[28]
    return VorbisIdentification(
        version=int.from_bytes(packet[7:11], "little"),
        channels=packet[11],
        sample_rate=int.from_bytes(packet[12:16], "little"),
        # Bitrate alanlari ISARETLI: -1 "belirtilmemis" icin kullanilir.
        bitrate_maximum=int.from_bytes(packet[16:20], "little", signed=True),
        bitrate_nominal=int.from_bytes(packet[20:24], "little", signed=True),
        bitrate_minimum=int.from_bytes(packet[24:28], "little", signed=True),
        # Blok boyutlari 2'nin kuvveti olarak, us degeri saklanir.
        blocksize_short=1 << (blocksizes & 0x0F),
        blocksize_long=1 << ((blocksizes >> 4) & 0x0F),
    )


def parse_comment(packet: bytes) -> tuple[str, dict[str, str]]:
    """Comment header'dan vendor ve etiketleri cikarir.

    Yerlesim OpusTags ile ayni (Vorbis'ten miras); tek fark bastaki paket tipi
    ve "vorbis" imzasi.
    """
    if len(packet) < 11 or packet[0] != COMMENT or packet[1:7] != MAGIC:
        return "", {}
    pos = 7
    vendor_len = int.from_bytes(packet[pos : pos + 4], "little")
    pos += 4
    vendor = packet[pos : pos + vendor_len].decode("utf-8", errors="replace")
    pos += vendor_len

    comments: dict[str, str] = {}
    if len(packet) < pos + 4:
        return vendor, comments
    count = int.from_bytes(packet[pos : pos + 4], "little")
    pos += 4
    for _ in range(min(count, 4096)):
        if len(packet) < pos + 4:
            break
        length = int.from_bytes(packet[pos : pos + 4], "little")
        pos += 4
        if len(packet) < pos + length:
            break
        entry = packet[pos : pos + length].decode("utf-8", errors="replace")
        pos += length
        key, sep, value = entry.partition("=")
        if sep:
            comments[key.upper()] = value
    return vendor, comments


@dataclass
class VorbisInfo:
    """Bir Ogg Vorbis dosyasinin bit akisi duzeyindeki tanimi."""

    identification: VorbisIdentification
    vendor: str = ""
    comments: dict[str, str] = field(default_factory=dict)
    audio_packets: int = 0
    audio_bytes: int = 0
    duration_s: float | None = None
    sampled: bool = False
    crc_checked_pages: int = 0
    bad_crc_pages: int = 0

    @property
    def kbps(self) -> float | None:
        """Paket verisinden olculen gercek bitrate (kapsayici yuku haric)."""
        if not self.duration_s:
            return None
        return self.audio_bytes * 8 / self.duration_s / 1000


def scan(path: Path, *, full_scan_limit: int = FULL_SCAN_LIMIT) -> VorbisInfo | None:
    """Bir Ogg Vorbis dosyasini tarar. Vorbis degilse None."""
    packets, stats = scan_packets(path, full_scan_limit=full_scan_limit)
    if not packets:
        return None
    identification = parse_identification(packets[0])
    if identification is None:
        return None

    vendor: str = ""
    comments: dict[str, str] = {}
    for packet in packets[1:3]:
        if packet[:1] == bytes([COMMENT]):
            vendor, comments = parse_comment(packet)
            break

    # Ilk uc paket basliktir (identification, comment, setup); geri kalani ses.
    audio = [p for p in packets[3:] if p]
    return VorbisInfo(
        identification=identification,
        vendor=vendor,
        comments=comments,
        audio_packets=len(audio),
        audio_bytes=sum(len(p) for p in audio),
        duration_s=_duration(stats, identification.sample_rate),
        sampled=stats.sampled,
        crc_checked_pages=stats.crc_checked_pages,
        bad_crc_pages=stats.bad_crc_pages,
    )


def _duration(stats: ScanStats, sample_rate: int) -> float | None:
    """Vorbis'te granule dogrudan ornek sayisidir; Opus'taki pre-skip yok."""
    if stats.last_granule is None or sample_rate <= 0:
        return None
    return stats.last_granule / sample_rate
