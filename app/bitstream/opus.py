"""Ogg Opus bit akisi: OpusHead, OpusTags ve paket TOC istatistikleri.

ffprobe bir Opus dosyasi icin "opus, 48000 Hz, stereo" der ve durur. Kodlayicinin
gercekte ne yaptigi -- her paketin modu (SILK/Hybrid/CELT), bant genisligi,
cerceve suresi, VBR olup olmadigi -- yalnizca paketlerin ilk baytindan, TOC'tan
okunur.

Bunun analiz icin dogrudan degeri var: TOC'un bildirdigi bant genisligi, olculen
alcak geciren kesim frekansiyla KARSILASTIRILABILIR. Fullband bir akis 20 kHz
soz verir; spektrumda 20.1 kHz'de duvar gormek codec'in kendi siniridir, bir
transcode belirtisi degildir. Bu iki bagimsiz olcum birbirini dogrular.

Referans: RFC 6716 bolum 3.1 (TOC bayti), RFC 7845 (Ogg kapsayicisi).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.bitstream.ogg import FULL_SCAN_LIMIT, ScanStats, scan_packets

HEAD_MAGIC = b"OpusHead"
TAGS_MAGIC = b"OpusTags"

Mode = Literal["SILK", "Hybrid", "CELT"]
Bandwidth = Literal["NB", "MB", "WB", "SWB", "FB"]

# Opus daima 48 kHz'de calisir. Kapsayici baska bir hiz bildiriyorsa o, kaynagin
# ORIJINAL hizidir (bilgi amacli), cozulen akisin hizi degil.
OPUS_RATE = 48000

# Bant genisliginin nominal ust frekansi (RFC 6716 tablo 1). Kesim frekansi
# olcumuyle karsilastirmak icin.
BANDWIDTH_HZ: dict[str, int] = {
    "NB": 4000,
    "MB": 6000,
    "WB": 8000,
    "SWB": 12000,
    "FB": 20000,
}

# TOC config (5 bit) -> (mod, bant, cerceve suresi ms). RFC 6716 bolum 3.1.
# CELT'te MB yoktur; atlanan satir bilincli.
_CONFIG_TABLE: tuple[tuple[Mode, Bandwidth, float], ...] = (
    *[("SILK", "NB", ms) for ms in (10.0, 20.0, 40.0, 60.0)],
    *[("SILK", "MB", ms) for ms in (10.0, 20.0, 40.0, 60.0)],
    *[("SILK", "WB", ms) for ms in (10.0, 20.0, 40.0, 60.0)],
    *[("Hybrid", "SWB", ms) for ms in (10.0, 20.0)],
    *[("Hybrid", "FB", ms) for ms in (10.0, 20.0)],
    *[("CELT", "NB", ms) for ms in (2.5, 5.0, 10.0, 20.0)],
    *[("CELT", "WB", ms) for ms in (2.5, 5.0, 10.0, 20.0)],
    *[("CELT", "SWB", ms) for ms in (2.5, 5.0, 10.0, 20.0)],
    *[("CELT", "FB", ms) for ms in (2.5, 5.0, 10.0, 20.0)],
)


@dataclass(frozen=True)
class Toc:
    """Bir Opus paketinin TOC baytindan cozulen tanimi."""

    config: int
    mode: Mode
    bandwidth: Bandwidth
    frame_ms: float
    stereo: bool
    frames: int

    @property
    def duration_ms(self) -> float:
        return self.frame_ms * self.frames


def decode_toc(packet: bytes) -> Toc | None:
    """Paketin ilk baytini (ve gerekirse ikincisini) cozer.

    Bos paket veya kod 3'te eksik ikinci bayt None dondurur; bozuk bir akista
    tahmin yurutmek istatistigi sessizce kirletir.
    """
    if not packet:
        return None
    toc = packet[0]
    config = toc >> 3
    stereo = bool((toc >> 2) & 1)
    count_code = toc & 0x03

    if count_code == 0:
        frames = 1
    elif count_code in (1, 2):
        frames = 2
    else:
        if len(packet) < 2:
            return None
        # Kod 3: ikinci baytin alt 6 biti cerceve sayisi. Ust iki bit VBR ve
        # dolgu bayraklaridir, sayiya dahil degil.
        frames = packet[1] & 0x3F
        if frames == 0:
            return None

    mode, bandwidth, frame_ms = _CONFIG_TABLE[config]
    return Toc(
        config=config,
        mode=mode,
        bandwidth=bandwidth,
        frame_ms=frame_ms,
        stereo=stereo,
        frames=frames,
    )


@dataclass(frozen=True)
class OpusHead:
    """OpusHead baslik paketi (RFC 7845 bolum 5.1)."""

    version: int
    channels: int
    pre_skip: int
    input_sample_rate: int
    output_gain_db: float
    mapping_family: int
    stream_count: int
    coupled_count: int


def parse_head(packet: bytes) -> OpusHead | None:
    """OpusHead paketini cozer."""
    if not packet.startswith(HEAD_MAGIC) or len(packet) < 19:
        return None
    # output_gain Q7.8 formatinda ISARETLI: -32768..32767 -> -128..+128 dB.
    gain_raw = int.from_bytes(packet[16:18], "little", signed=True)
    family = packet[18]
    # Aile 0'da akis duzeni sabittir: tek akis, stereo ise eslenmis.
    stream_count = 1
    coupled_count = 1 if packet[9] == 2 else 0
    if family != 0 and len(packet) >= 21:
        stream_count = packet[19]
        coupled_count = packet[20]
    return OpusHead(
        version=packet[8],
        channels=packet[9],
        pre_skip=int.from_bytes(packet[10:12], "little"),
        input_sample_rate=int.from_bytes(packet[12:16], "little"),
        output_gain_db=gain_raw / 256.0,
        mapping_family=family,
        stream_count=stream_count,
        coupled_count=coupled_count,
    )


def parse_tags(packet: bytes) -> tuple[str, dict[str, str]]:
    """OpusTags paketinden vendor dizesini ve yorumlari cikarir.

    Vendor dizesi kokeni ele verir: `Lavf*` ffmpeg ile yeniden paketlendigini,
    `libopus *` dogrudan referans kodlayiciyi gosterir. YouTube'dan yt-dlp ile
    indirilen bir dosyada orijinal imza remux sirasinda silinir ve `Lavf`
    kalir -- yani vendor tek basina "kim kodladi" sorusunu cevaplamaz, ama
    "bu dosya elden gecmis mi" sorusunu cevaplar.
    """
    if not packet.startswith(TAGS_MAGIC) or len(packet) < 12:
        return "", {}
    pos = len(TAGS_MAGIC)
    vendor_len = int.from_bytes(packet[pos : pos + 4], "little")
    pos += 4
    vendor = packet[pos : pos + vendor_len].decode("utf-8", errors="replace")
    pos += vendor_len

    comments: dict[str, str] = {}
    if len(packet) < pos + 4:
        return vendor, comments
    count = int.from_bytes(packet[pos : pos + 4], "little")
    pos += 4
    # Sayac dosyadan geliyor; bozuk bir deger milyonlarca donguye yol acmasin
    # diye kalan uzunluk her adimda kontrol ediliyor.
    for _ in range(min(count, 1024)):
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
class OpusInfo:
    """Bir Ogg Opus dosyasinin bit akisi duzeyindeki tanimi."""

    head: OpusHead
    vendor: str
    comments: dict[str, str]

    packets: int = 0
    undecodable_packets: int = 0
    mode_hist: Counter[str] = field(default_factory=Counter)
    bandwidth_hist: Counter[str] = field(default_factory=Counter)
    frame_ms_hist: Counter[float] = field(default_factory=Counter)
    frames_per_packet_hist: Counter[int] = field(default_factory=Counter)

    total_packet_bytes: int = 0
    total_duration_ms: float = 0.0
    min_packet_bytes: int = 0
    max_packet_bytes: int = 0

    granule_duration_s: float | None = None
    sampled: bool = False
    crc_checked_pages: int = 0
    bad_crc_pages: int = 0

    @property
    def mean_packet_bytes(self) -> float:
        return self.total_packet_bytes / self.packets if self.packets else 0.0

    @property
    def kbps(self) -> float:
        """Paket verisinden hesaplanan gercek bitrate.

        Kapsayici yukunu (Ogg sayfa basliklari) ICERMEZ, yani ffprobe'un
        bildirdigi konteyner bitrate'inden bir miktar dusuktur. 5 s'lik bir
        test dosyasinda olculen fark ~1.9 kbps (170.0 karsi 171.7).

        Bu sayi kodlayicinin HEDEFINE esit degildir ve olmasi da gerekmez:
        `-b:a 128k` ile uretilen bir dosyada saf 1 kHz sinus 170 kbps, pembe
        gurultu 88 kbps olctu. VBR'de hedef bir tavan degil bir yonlendirmedir,
        ve saf ton bir donusum kodeki icin gurultuden PAHALIDIR. Arayuz
        "128 kbps" yazan bir dosyanin gercekten 128 kbps oldugunu varsaymamali;
        gosterilecek sayi budur.
        """
        if self.total_duration_ms <= 0:
            return 0.0
        return self.total_packet_bytes * 8 / self.total_duration_ms

    @property
    def is_vbr(self) -> bool:
        """Paket boyutlari degisiyorsa VBR.

        CBR bir Opus akisinda her paket ayni bayt sayisindadir; tek bir
        istisna bile (son paket haric) VBR demektir. Sinir olarak boyut
        cesitliligine bakiliyor cunku Opus'ta akista bir "CBR bayragi" yok.
        """
        return self.max_packet_bytes > self.min_packet_bytes

    @property
    def dominant_mode(self) -> str:
        return self.mode_hist.most_common(1)[0][0] if self.mode_hist else ""

    @property
    def dominant_bandwidth(self) -> str:
        return self.bandwidth_hist.most_common(1)[0][0] if self.bandwidth_hist else ""

    @property
    def nominal_bandwidth_hz(self) -> int | None:
        """Baskin bant genisliginin nominal ust frekansi.

        Olculen alcak geciren kesimle karsilastirilir: fullband bir akista
        ~20 kHz'lik duvar codec'in kendi siniridir, transcode belirtisi degil.
        """
        name = self.dominant_bandwidth
        return BANDWIDTH_HZ.get(name) if name else None


def _apply_stats(info: OpusInfo, packets: list[bytes]) -> None:
    sizes: list[int] = []
    for packet in packets:
        toc = decode_toc(packet)
        if toc is None:
            info.undecodable_packets += 1
            continue
        info.packets += 1
        info.mode_hist[toc.mode] += 1
        info.bandwidth_hist[toc.bandwidth] += 1
        info.frame_ms_hist[toc.frame_ms] += 1
        info.frames_per_packet_hist[toc.frames] += 1
        info.total_duration_ms += toc.duration_ms
        info.total_packet_bytes += len(packet)
        sizes.append(len(packet))
    if sizes:
        info.min_packet_bytes = min(sizes)
        info.max_packet_bytes = max(sizes)


def scan(path: Path, *, full_scan_limit: int = FULL_SCAN_LIMIT) -> OpusInfo | None:
    """Bir Ogg Opus dosyasini tarar. Opus degilse None dondurur."""
    packets, stats = scan_packets(path, full_scan_limit=full_scan_limit)
    if not packets or not packets[0].startswith(HEAD_MAGIC):
        return None

    head = parse_head(packets[0])
    if head is None:
        return None
    vendor, comments = parse_tags(packets[1]) if len(packets) > 1 else ("", {})

    info = OpusInfo(head=head, vendor=vendor, comments=comments)
    info.sampled = stats.sampled
    info.crc_checked_pages = stats.crc_checked_pages
    info.bad_crc_pages = stats.bad_crc_pages
    _apply_stats(info, packets[2:])
    info.granule_duration_s = _granule_duration(stats, head.pre_skip)
    return info


def _granule_duration(stats: ScanStats, pre_skip: int) -> float | None:
    """Son granule konumundan gercek sureyi hesaplar.

    Opus'ta granule 48 kHz ornek cinsindendir ve pre-skip'i ICERIR; cikarilmazsa
    sure birkac milisaniye uzun gorunur. Ornekleme yapildiysa son bolge dosyanin
    sonunu kapsadigi icin bu deger yine gecerlidir.
    """
    if stats.last_granule is None:
        return None
    samples = stats.last_granule - pre_skip
    return samples / OPUS_RATE if samples > 0 else None
