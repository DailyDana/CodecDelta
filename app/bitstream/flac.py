"""FLAC metadata bloklari.

Buradan cikan bilgiler "bu FLAC gercekten kayipsiz bir kaynaktan mi geliyor"
sorusunun BAGLAMSAL kanitlarini olusturur. Spektral kanitlar (kesim frekansi,
diz keskinligi, kesim ustu taban) asil delildir; bunlar onlari destekler:

  - vendor dizesi: `reference libFLAC*` bir rip aracindan, `Lavf*` ffmpeg'den
    gecmis demektir. Ikincisi tek basina sucluyucu degil ama "EAC ile ripledim"
    iddiasiyla celisir. Kodlayiciyi belirleyen TEK guvenilir alan budur.
  - seektable/padding yoklugu: gercek rip araclari ikisini de yazar.
  - sikistirma orani: neredeyse bedava ve sasirtici derecede ayirt edici.
    Gercek 44.1/16 muzikte ~0.55-0.70; kayipli bir kaynaktan gelen FLAC sik sik
    0.40-0.55 bandindadir, cunku silinmis yuksek frekanslar cok iyi sikisir.

STREAMINFO MD5 hakkinda bir uyari: eslesmesi dosyanin KENDI ICINDE tutarli
oldugunu gosterir (kodlayici sakladigi PCM'in ozetini dogru yazmis), kaynagin
kayipsiz oldugunu DEGIL. Bir MP3'u cozup FLAC'a kodlarsaniz MD5 yine tutar.

Blok boyutu bir kodlayici parmak izi DEGILDIR -- bu olculdu. Ayni ffmpeg
derlemesi ayni surumde uc farkli deger uretti:

    lavfi kaynagindan dogrudan  -> 1024   (girdi filtresinin cerceve boyutu)
    wav dosyasindan             -> 4096
    -frame_size 4608 ile        -> 4608

4096, "referans libFLAC'in varsayilani" diye bilinen degerdir; yani boyuta
bakan bir kural modern ffmpeg ciktilarinin cogunu "referans rip" diye
etiketlerdi. Blok boyutu VERI olarak raporlanir (1024 gibi alisildik olmayan
bir deger dikkat cekicidir) ama ondan kodlayici cikarilmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.ffmpeg_runner import CancelToken, run_capture

MAGIC = b"fLaC"

BLOCK_STREAMINFO = 0
BLOCK_PADDING = 1
BLOCK_APPLICATION = 2
BLOCK_SEEKTABLE = 3
BLOCK_VORBIS_COMMENT = 4
BLOCK_CUESHEET = 5
BLOCK_PICTURE = 6

BLOCK_NAMES = {
    BLOCK_STREAMINFO: "STREAMINFO",
    BLOCK_PADDING: "PADDING",
    BLOCK_APPLICATION: "APPLICATION",
    BLOCK_SEEKTABLE: "SEEKTABLE",
    BLOCK_VORBIS_COMMENT: "VORBIS_COMMENT",
    BLOCK_CUESHEET: "CUESHEET",
    BLOCK_PICTURE: "PICTURE",
}

# Metadata bloklari dosyanin basindadir; bu kadari her zaman yeter. Gomulu
# kapak resmi birkac yuz KB olabilir, bu yuzden cok dar tutulmuyor.
_HEADER_READ_BYTES = 4 << 20

_ZERO_MD5 = b"\x00" * 16


@dataclass(frozen=True)
class StreamInfo:
    """STREAMINFO blogu: akisin temel tanimi."""

    min_blocksize: int
    max_blocksize: int
    min_framesize: int
    max_framesize: int
    sample_rate: int
    channels: int
    bits_per_sample: int
    total_samples: int
    md5: bytes

    @property
    def duration_s(self) -> float | None:
        if not self.sample_rate or not self.total_samples:
            return None
        return self.total_samples / self.sample_rate

    @property
    def md5_present(self) -> bool:
        """MD5 alani sifirsa kodlayici hic yazmamistir.

        Sifir olmasi bozukluk degil, "bilinmiyor" demektir; dogrulama o zaman
        yapilamaz ve rapor bunu "atlandi" diye soylemek zorundadir.
        """
        return self.md5 != _ZERO_MD5

    @property
    def fixed_blocksize(self) -> bool:
        return self.min_blocksize == self.max_blocksize

    @property
    def raw_pcm_bytes(self) -> int:
        """Sikistirilmamis hali kac bayt olurdu."""
        return self.total_samples * self.channels * self.bits_per_sample // 8


@dataclass(frozen=True)
class MetadataBlock:
    """Bir metadata blogunun yerini ve boyutunu tanimlar."""

    block_type: int
    length: int
    offset: int
    last: bool

    @property
    def name(self) -> str:
        return BLOCK_NAMES.get(self.block_type, f"RESERVED_{self.block_type}")


@dataclass
class FlacInfo:
    """Bir FLAC dosyasinin metadata duzeyindeki tanimi."""

    stream_info: StreamInfo
    vendor: str = ""
    comments: dict[str, str] = field(default_factory=dict)
    blocks: tuple[MetadataBlock, ...] = ()
    file_size: int = 0
    metadata_bytes: int = 0
    id3_bytes: int = 0

    @property
    def audio_bytes(self) -> int:
        """Yalnizca ses cerceveleri: metadata ve gomulu kapak haric.

        Kapak resmini dahil etmek sikistirma oranini sessizce bozar; 58 MB'lik
        bir dosyada 296 KB'lik kapak orani ~%0.5 kaydiriyordu.
        """
        return max(0, self.file_size - self.metadata_bytes - self.id3_bytes)

    @property
    def compression_ratio(self) -> float | None:
        """Ses cercevelerinin ham PCM'e orani.

        Gercek 44.1/16 muzikte tipik olarak 0.55-0.70. Belirgin sekilde dusuk
        bir oran, silinmis yuksek frekanslarin cok iyi sikismasindan gelebilir
        -- yani kayipli bir kaynak isareti. Tek basina kanit DEGIL: sessiz,
        seyrek veya dogal olarak karanlik kayitlar da dusuk oran verir.
        """
        raw = self.stream_info.raw_pcm_bytes
        if raw <= 0 or self.audio_bytes <= 0:
            return None
        return self.audio_bytes / raw

    @property
    def has_block(self) -> dict[str, bool]:
        present = {b.name for b in self.blocks}
        return {name: name in present for name in BLOCK_NAMES.values()}

    @property
    def encoder_family(self) -> str | None:
        """Vendor dizesinden kodlayici ailesi.

        Kodlayiciyi belirleyen tek guvenilir alan budur; blok boyutu DEGIL
        (gerekcesi modul basindaki olcume bak). Vendor bosca None doner --
        "bilinmiyor", "referans kodlayici" degil.
        """
        if not self.vendor:
            return None
        lowered = self.vendor.lower()
        if lowered.startswith("reference libflac"):
            return "reference libFLAC"
        if lowered.startswith("lavf") or lowered.startswith("libavcodec"):
            return "ffmpeg"
        return self.vendor


def parse_stream_info(data: bytes) -> StreamInfo | None:
    """34 baytlik STREAMINFO govdesini cozer."""
    if len(data) < 34:
        return None
    min_bs = int.from_bytes(data[0:2], "big")
    max_bs = int.from_bytes(data[2:4], "big")
    min_fs = int.from_bytes(data[4:7], "big")
    max_fs = int.from_bytes(data[7:10], "big")
    # Sonraki 64 bit sikistirilmis: 20 bit hiz, 3 bit kanal-1, 5 bit bps-1,
    # 36 bit toplam ornek.
    packed = int.from_bytes(data[10:18], "big")
    return StreamInfo(
        min_blocksize=min_bs,
        max_blocksize=max_bs,
        min_framesize=min_fs,
        max_framesize=max_fs,
        sample_rate=packed >> 44,
        channels=((packed >> 41) & 0x07) + 1,
        bits_per_sample=((packed >> 36) & 0x1F) + 1,
        total_samples=packed & ((1 << 36) - 1),
        md5=data[18:34],
    )


def parse_vorbis_comment(data: bytes) -> tuple[str, dict[str, str]]:
    """VORBIS_COMMENT blogu. Alan uzunluklari LITTLE endian.

    FLAC'in geri kalani big endian oldugu icin bu, gozden kacmasi kolay bir
    ayrintidir: yanlis okunursa vendor uzunlugu absurt cikar.
    """
    if len(data) < 4:
        return "", {}
    pos = 0
    vendor_len = int.from_bytes(data[pos : pos + 4], "little")
    pos += 4
    vendor = data[pos : pos + vendor_len].decode("utf-8", errors="replace")
    pos += vendor_len

    comments: dict[str, str] = {}
    if len(data) < pos + 4:
        return vendor, comments
    count = int.from_bytes(data[pos : pos + 4], "little")
    pos += 4
    for _ in range(min(count, 4096)):
        if len(data) < pos + 4:
            break
        length = int.from_bytes(data[pos : pos + 4], "little")
        pos += 4
        if len(data) < pos + length:
            break
        entry = data[pos : pos + length].decode("utf-8", errors="replace")
        pos += length
        key, sep, value = entry.partition("=")
        if sep:
            comments[key.upper()] = value
    return vendor, comments


def _id3_length(data: bytes) -> int:
    """Basta bir ID3v2 etiketi varsa uzunlugunu dondurur.

    FLAC'ta ID3 standart disidir ama gercekte karsilasilir (bazi etiketleyiciler
    yazar). Atlanmazsa "fLaC" imzasi bulunamaz ve dosya FLAC degil sanilir.
    """
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    # Boyut "syncsafe": her baytin yalnizca alt 7 biti kullanilir.
    size = 0
    for byte in data[6:10]:
        size = (size << 7) | (byte & 0x7F)
    return 10 + size


def scan(path: Path) -> FlacInfo | None:
    """Bir FLAC dosyasinin metadata bloklarini okur. FLAC degilse None."""
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        data = fh.read(_HEADER_READ_BYTES)

    id3 = _id3_length(data)
    if data[id3 : id3 + 4] != MAGIC:
        return None

    pos = id3 + 4
    stream_info: StreamInfo | None = None
    vendor = ""
    comments: dict[str, str] = {}
    blocks: list[MetadataBlock] = []

    while pos + 4 <= len(data):
        header = data[pos : pos + 4]
        last = bool(header[0] & 0x80)
        block_type = header[0] & 0x7F
        length = int.from_bytes(header[1:4], "big")
        body_start = pos + 4
        body = data[body_start : body_start + length]
        blocks.append(MetadataBlock(block_type=block_type, length=length, offset=pos, last=last))

        if block_type == BLOCK_STREAMINFO:
            stream_info = parse_stream_info(body)
        elif block_type == BLOCK_VORBIS_COMMENT and len(body) == length:
            vendor, comments = parse_vorbis_comment(body)

        pos = body_start + length
        if last:
            break

    if stream_info is None:
        return None

    return FlacInfo(
        stream_info=stream_info,
        vendor=vendor,
        comments=comments,
        blocks=tuple(blocks),
        file_size=file_size,
        # Blok basliklari (4 bayt) + govdeleri, "fLaC" imzasi dahil.
        metadata_bytes=pos - id3,
        id3_bytes=id3,
    )


def verify_md5(
    ffmpeg: Path,
    path: Path,
    stream_info: StreamInfo,
    *,
    cancel: CancelToken | None = None,
    timeout: float = 600.0,
) -> bool | None:
    """Cozulen PCM'in MD5'ini STREAMINFO ile karsilastirir.

    None dondurur: STREAMINFO'da MD5 yoksa (dogrulanamaz) veya cozme
    basarisizsa. True/False yalnizca gercekten karsilastirildiginda.

    NE KANITLAR: dosya kendi icinde tutarli, ses verisi bozulmamis.
    NE KANITLAMAZ: kaynagin kayipsiz oldugunu. Bir MP3'u cozup FLAC'a
    kodlarsaniz bu kontrol yine gecer. Kayipli kaynak sorusu spektral
    kanitlarla cevaplanir, bununla degil.
    """
    if not stream_info.md5_present:
        return None
    run = run_capture(
        ffmpeg,
        [
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "md5",
            "-",
        ],
        timeout=timeout,
        cancel=cancel,
        check=False,
    )
    if not run.ok:
        return None
    text = run.stdout_text().strip()
    _, _, digest = text.partition("=")
    return digest.strip().lower() == stream_info.md5.hex()
