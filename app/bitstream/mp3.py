"""MP3 cerceve basliklari, Xing/Info ve LAME etiketi.

Bu modulun en degerli ciktisi LAME etiketindeki ALCAK GECIREN alanidir:
kodlayici uyguladigi kesim frekansini 100 Hz biriminde dosyaya yazar. Yani
"bu MP3 nerede kesilmis" sorusu, spektrumu hic olcmeden, dogrudan cevaplanir.

Bunun analiz icin iki kullanimi var:
  - Bir MP3'u referansla karsilastirirken beklenen kesimi ONCEDEN bilmek.
  - Bir FLAC'in kayipli kaynaktan gelip gelmedigini arastirirken, olculen
    kesimi bilinen LAME degerleriyle karsilastirmak. LAME'in 128 kbps'te
    yazdigi 16 kHz, spektrumda gorulen 16 kHz ile ayni yere duserse bu
    rastlanti degildir.

Kodlayici gecikmesi ve dolgu da buradan okunur; hizalama asamasi bunlari
bilirse capraz korelasyona daha iyi bir baslangic tahmini verilebilir.

ONEMLI SINIR -- olculdu: ffmpeg'in libmp3lame sarmalayicisi GERCEK bir LAME
etiketi yazmaz. Uretilen dosyada etiket dogru yerde (Xing + 0x78) ve encoder
dizesi okunabilir ("Lavc63.7."), gecikme/dolgu da dogru (576/936, LAME'in
standart degerleri) -- ama +9 ile +20 arasindaki tum alanlar SIFIR:

    4c 61 76 63 36 33 2e 37 2e | 00 00 00 ... 00 | 24 03 a8
    <-- encoder "Lavc63.7." -->  <-- vbr/lowpass/ath/abr: hepsi sifir -->

Yani alcak geciren alani yalnizca gercek LAME ikililerinden (lame.exe,
foobar2000, eski rip araclari) uretilmis dosyalarda dolu olur. ffmpeg
ciktilarinda `declared_lowpass_hz` None doner ve kesim yalnizca olcumle
bulunur. Ayristirma spesifikasyona gore yazildi ama DOGRULANMALI: elde gercek
bir LAME kodlayicisi yok, bu yuzden dolu bir alcak geciren alani uctan uca
sinanmadi (sentetik etiketle sinandi).

Referans: MPEG-1/2 Audio Layer III cerceve basligi ve LAME etiketi uzantisi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAGIC_XING = b"Xing"
MAGIC_INFO = b"Info"
MAGIC_VBRI = b"VBRI"

# MPEG surum kimligi (baslikta 2 bit).
_V25, _VRESERVED, _V2, _V1 = 0, 1, 2, 3

# Layer III icin bitrate tablolari (kbps). Indeks 0 "free", 15 gecersiz.
_BITRATES_V1_L3 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
_BITRATES_V2_L3 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)

_RATES = {
    _V1: (44100, 48000, 32000),
    _V2: (22050, 24000, 16000),
    _V25: (11025, 12000, 8000),
}

_CHANNEL_MODES = ("stereo", "joint stereo", "dual channel", "mono")

# LAME etiketindeki VBR yontemi (bayt 9'un alt 4 biti).
VBR_METHODS = {
    0: "unknown",
    1: "CBR",
    2: "ABR",
    3: "VBR (old/rh)",
    4: "VBR (new/mtrh)",
    5: "VBR (mt)",
    6: "VBR (rh2)",
    8: "CBR (2-pass)",
    9: "ABR (2-pass)",
}

# Ilk cerceveyi ararken taranacak en fazla bayt. ID3v2 etiketi zaten atlanir;
# bu, etiketten sonra hala cop varsa diye bir emniyet payi.
_SYNC_SEARCH_BYTES = 1 << 20

# Ortalama bitrate icin okunacak cerceve sayisi. VBR dalgalanmasini
# duzlestirmeye yeter ve buyuk dosyada tum cerceveleri gezmekten kacinir.
_AVG_FRAMES = 400


@dataclass(frozen=True)
class FrameHeader:
    """Bir MPEG audio cerceve basligi."""

    version: int
    layer: int
    bitrate_kbps: int
    sample_rate: int
    padding: bool
    channel_mode: str
    mode_extension: int
    protected: bool

    @property
    def is_layer3(self) -> bool:
        return self.layer == 3

    @property
    def samples_per_frame(self) -> int:
        # MPEG1 Layer III 1152 ornek, MPEG2/2.5 yarisi kadar.
        return 1152 if self.version == _V1 else 576

    @property
    def frame_bytes(self) -> int:
        if not self.bitrate_kbps or not self.sample_rate:
            return 0
        coefficient = 144 if self.version == _V1 else 72
        return coefficient * self.bitrate_kbps * 1000 // self.sample_rate + int(self.padding)

    @property
    def is_mono(self) -> bool:
        return self.channel_mode == "mono"

    @property
    def is_joint_stereo(self) -> bool:
        return self.channel_mode == "joint stereo"


def parse_frame_header(data: bytes, offset: int = 0) -> FrameHeader | None:
    """4 baytlik cerceve basligini cozer. Gecerli degilse None.

    Gecersiz kombinasyonlar (ayrilmis surum, bitrate 15, hiz indeksi 3)
    reddedilir; sync deseni ses verisinde rastlantiyla gectigi icin bu
    kontroller yanlis eslesmeyi eleyen asil mekanizmadir.
    """
    if len(data) < offset + 4:
        return None
    b0, b1, b2, b3 = data[offset : offset + 4]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None

    version = (b1 >> 3) & 0x03
    layer_bits = (b1 >> 1) & 0x03
    if version == _VRESERVED or layer_bits == 0:
        return None
    layer = 4 - layer_bits

    bitrate_index = (b2 >> 4) & 0x0F
    rate_index = (b2 >> 2) & 0x03
    if bitrate_index in (0, 15) or rate_index == 3:
        return None

    table = _BITRATES_V1_L3 if version == _V1 else _BITRATES_V2_L3
    return FrameHeader(
        version=version,
        layer=layer,
        bitrate_kbps=table[bitrate_index],
        sample_rate=_RATES[version][rate_index],
        padding=bool((b2 >> 1) & 1),
        channel_mode=_CHANNEL_MODES[(b3 >> 6) & 0x03],
        mode_extension=(b3 >> 4) & 0x03,
        protected=not (b1 & 0x01),
    )


def find_first_frame(data: bytes, start: int = 0) -> tuple[int, FrameHeader] | None:
    """Ilk gecerli cerceveyi bulur.

    Tek bir gecerli baslik yeterli degil: bir sonraki cercevenin de hesaplanan
    konumda cikmasi aranir. Ses verisinde 0xFF 0xEx dizisi sik gectigi icin
    bu zincir kontrolu yanlis senkronu pratikte tumuyle eler.
    """
    limit = min(len(data), start + _SYNC_SEARCH_BYTES)
    pos = start
    while pos < limit:
        pos = data.find(b"\xff", pos, limit)
        if pos < 0:
            return None
        header = parse_frame_header(data, pos)
        if header is not None and header.is_layer3 and header.frame_bytes:
            nxt = pos + header.frame_bytes
            follower = parse_frame_header(data, nxt)
            if follower is not None and follower.sample_rate == header.sample_rate:
                return pos, header
        pos += 1
    return None


@dataclass(frozen=True)
class LameTag:
    """LAME etiketi uzantisi."""

    encoder: str
    vbr_method: str
    lowpass_hz: int | None
    encoder_delay: int
    encoder_padding: int
    ath_type: int
    abr_bitrate: int

    @property
    def is_lame(self) -> bool:
        return self.encoder.startswith("LAME")


@dataclass(frozen=True)
class XingHeader:
    """Xing/Info blogu. "Info" CBR, "Xing" VBR anlamina gelir."""

    magic: str
    frames: int | None
    byte_count: int | None
    quality: int | None
    lame: LameTag | None

    @property
    def declares_vbr(self) -> bool:
        return self.magic == "Xing"


def _xing_offset(header: FrameHeader) -> int:
    """Xing blogunun cerceve basindan uzakligi.

    Yan bilgi (side info) alani surum ve kanal sayisina gore degisir; Xing
    blogu tam onun ardina yazilir.
    """
    if header.version == _V1:
        return 4 + (17 if header.is_mono else 32)
    return 4 + (9 if header.is_mono else 17)


def parse_lame_tag(data: bytes, offset: int) -> LameTag | None:
    """LAME etiketini cozer. `offset` 9 baytlik surum dizesinin basi."""
    if len(data) < offset + 24:
        return None
    encoder = data[offset : offset + 9].decode("latin-1", errors="replace").rstrip("\x00")
    if not encoder or not encoder[0].isalpha():
        return None

    vbr_code = data[offset + 9] & 0x0F
    # Alcak geciren 100 Hz biriminde saklanir; 0 "bilinmiyor" demektir.
    lowpass_raw = data[offset + 10]
    delay_padding = int.from_bytes(data[offset + 21 : offset + 24], "big")

    return LameTag(
        encoder=encoder,
        vbr_method=VBR_METHODS.get(vbr_code, f"unknown ({vbr_code})"),
        lowpass_hz=lowpass_raw * 100 if lowpass_raw else None,
        # Gecikme 12 bit, dolgu 12 bit, pes pese paketlenmis.
        encoder_delay=delay_padding >> 12,
        encoder_padding=delay_padding & 0xFFF,
        ath_type=data[offset + 19] & 0x0F,
        abr_bitrate=data[offset + 20],
    )


def parse_xing(data: bytes, frame_offset: int, header: FrameHeader) -> XingHeader | None:
    """Xing/Info blogunu ve varsa LAME uzantisini cozer."""
    pos = frame_offset + _xing_offset(header)
    if len(data) < pos + 8:
        return None
    magic = data[pos : pos + 4]
    if magic not in (MAGIC_XING, MAGIC_INFO):
        return None

    flags = int.from_bytes(data[pos + 4 : pos + 8], "big")
    cursor = pos + 8

    def take(n: int) -> int | None:
        nonlocal cursor
        if len(data) < cursor + n:
            return None
        value = int.from_bytes(data[cursor : cursor + n], "big")
        cursor += n
        return value

    frames = take(4) if flags & 0x01 else None
    byte_count = take(4) if flags & 0x02 else None
    if flags & 0x04:
        cursor += 100  # TOC tablosu, aranmiyor
    quality = take(4) if flags & 0x08 else None

    # LAME etiketi opsiyonel alanlarin hemen ardindadir. Tum bayraklar
    # kalkikken bu, Xing imzasindan 0x78 bayt sonrasina denk gelir -- literaturde
    # gecen o sabit, genel kural degil bu ozel durumdur.
    lame = parse_lame_tag(data, cursor)

    return XingHeader(
        magic=magic.decode("ascii"),
        frames=frames,
        byte_count=byte_count,
        quality=quality,
        lame=lame,
    )


@dataclass
class Mp3Info:
    """Bir MP3 dosyasinin bit akisi duzeyindeki tanimi."""

    first_frame: FrameHeader
    xing: XingHeader | None
    id3_bytes: int
    file_size: int
    audio_bytes: int
    sampled_frames: int = 0
    sampled_bitrates: tuple[int, ...] = ()

    @property
    def is_vbr(self) -> bool:
        """Ornekleme sirasinda birden fazla bitrate gorulduyse VBR.

        Xing bloguna tek basina guvenilmiyor: bazi kodlayicilar CBR dosyaya da
        "Xing" yazar. Cerceve basliklarinda gorulen cesitlilik daha dogrudan
        bir kanittir.
        """
        return len(set(self.sampled_bitrates)) > 1

    @property
    def nominal_bitrate_kbps(self) -> int:
        return self.first_frame.bitrate_kbps

    @property
    def average_bitrate_kbps(self) -> float | None:
        if not self.sampled_bitrates:
            return None
        return sum(self.sampled_bitrates) / len(self.sampled_bitrates)

    @property
    def duration_s(self) -> float | None:
        """Sure: once Xing cerceve sayisindan, yoksa ortalama bitrate'ten."""
        rate = self.first_frame.sample_rate
        if self.xing and self.xing.frames and rate:
            return self.xing.frames * self.first_frame.samples_per_frame / rate
        average = self.average_bitrate_kbps
        if average:
            return self.audio_bytes * 8 / (average * 1000)
        return None

    @property
    def declared_lowpass_hz(self) -> int | None:
        """Kodlayicinin yazdigi alcak geciren kesimi.

        Olculen kesim frekansiyla karsilastirilir. Ikisi ortusuyorsa dosyanin
        bant siniri kodlayicidan gelir, kaynagin kendisinden degil.
        """
        return self.xing.lame.lowpass_hz if self.xing and self.xing.lame else None


def _id3_length(data: bytes) -> int:
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size = 0
    for byte in data[6:10]:
        size = (size << 7) | (byte & 0x7F)
    return 10 + size


def scan(path: Path, *, max_frames: int = _AVG_FRAMES) -> Mp3Info | None:
    """Bir MP3 dosyasini tarar. MP3 degilse None dondurur."""
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        data = fh.read(_SYNC_SEARCH_BYTES + 512 * 1024)

    id3 = _id3_length(data)
    found = find_first_frame(data, id3)
    if found is None:
        return None
    offset, header = found

    xing = parse_xing(data, offset, header)

    # Ortalama bitrate icin bastan birkac yuz cerceve gez. Xing blogunu tasiyan
    # ilk cerceve sessizdir ve bitrate'i temsil etmez, o yuzden atlanir.
    bitrates: list[int] = []
    cursor = offset + header.frame_bytes if xing else offset
    for _ in range(max_frames):
        frame = parse_frame_header(data, cursor)
        if frame is None or not frame.frame_bytes:
            break
        bitrates.append(frame.bitrate_kbps)
        cursor += frame.frame_bytes

    return Mp3Info(
        first_frame=header,
        xing=xing,
        id3_bytes=id3,
        file_size=file_size,
        audio_bytes=max(0, file_size - offset),
        sampled_frames=len(bitrates),
        sampled_bitrates=tuple(bitrates),
    )
