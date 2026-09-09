"""ffprobe ile dosya inceleme.

Bir dosya arayuze birakildiginda yapilan TEK is budur ve milisaniyeler surer:
`-show_format -show_streams` yalnizca konteyner basliklarini okur, tek bir video
paketine dokunmaz. 20 GB'lik bir MKV ile 3 MB'lik bir FLAC ayni hizda incelenir.

Bitrate uc kademeli cozulur ve bulunamazsa UYDURULMAZ. Olculen gercek: MKV
icindeki bir Opus izinde ffprobe `bit_rate` alanini hic dondurmuyor. Sirasiyla:
akisin kendi bit_rate'i, tek izli dosyada konteynerin bit_rate'i, son care
olarak kisa bir araliktan paket boyutu ornekleme. Hicbiri olmazsa None kalir ve
arayuz "--" gosterir. (`-count_packets` bilincli olarak kullanilmaz: 20 GB'lik
bir dosyada tum paketleri saymak kabul edilemez.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import ProbeError
from app.core.ffmpeg_runner import CancelToken, run_capture

# Kayipsiz kodekler. Arayuz "referans" tarafi icin bunlari onerir, ve
# referanssiz dogrulama modu yalnizca bunlar icin anlamlidir.
LOSSLESS_CODECS = frozenset(
    {
        "flac",
        "alac",
        "wavpack",
        "tta",
        "ape",
        "tak",
        "shorten",
        "mlp",
        "truehd",
        "dsd_lsbf",
        "dsd_msbf",
        "dsd_lsbf_planar",
        "dsd_msbf_planar",
    }
)

# Paket ornekleme penceresi. 30 s, VBR dalgalanmasini duzlestirmeye yeter ve
# en yavas konteynerde bile bir saniyenin altinda kalir.
_SAMPLE_WINDOW_S = 30.0


@dataclass(frozen=True)
class AudioStreamInfo:
    """Bir ses izinin ffprobe'dan okunan tanimi."""

    # `-map 0:a:<audio_index>` icin kullanilan SIRA numarasi. Konteynerdeki
    # mutlak `index` ile karistirilmamali: MKV'de video 0, ilk ses 1 olabilir
    # ama onun audio_index'i 0'dir.
    audio_index: int
    index: int
    codec: str
    profile: str | None
    sample_rate: int
    channels: int
    channel_layout: str
    sample_fmt: str
    bits_per_raw_sample: int | None
    bit_rate: int | None
    duration: float | None
    language: str | None
    title: str | None
    is_default: bool

    @property
    def is_lossless(self) -> bool:
        return self.codec in LOSSLESS_CODECS or self.codec.startswith("pcm_")

    def label(self) -> str:
        """Iz seciciye yazilacak tek satirlik ozet (Ingilizce, arayuz metni)."""
        parts = [f"a:{self.audio_index}", self.codec]
        parts.append(self.channel_layout or f"{self.channels}ch")
        if self.language:
            parts.append(self.language)
        parts.append(f"{self.bit_rate // 1000} kbps" if self.bit_rate else "--")
        if self.title:
            parts.append(f'"{self.title}"')
        if self.is_default:
            parts.append("(default)")
        return "  ".join(parts)


@dataclass(frozen=True)
class Probe:
    """Bir medya dosyasinin incelenmis hali."""

    path: Path
    container: str
    size: int
    duration: float | None
    audio: tuple[AudioStreamInfo, ...]
    has_video: bool
    # Gomulu kapak resmi. Video izi sayilmaz ama konteyner bitrate'ini sisirir,
    # bu yuzden ayri tutulur (bkz. measure_bit_rate).
    has_attached_pic: bool
    format_bit_rate: int | None
    tags: dict[str, str]

    def stream(self, audio_index: int) -> AudioStreamInfo:
        for s in self.audio:
            if s.audio_index == audio_index:
                return s
        raise ProbeError(f"No audio track a:{audio_index} in {self.path.name}.")


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # NaN ele


def probe(
    ffprobe: Path,
    path: Path,
    *,
    cancel: CancelToken | None = None,
    timeout: float = 30.0,
) -> Probe:
    """Dosyayi inceler. Ses izi yoksa `ProbeError` firlatir."""
    if not path.is_file():
        raise ProbeError(f"File not found: {path}")

    run = run_capture(
        ffprobe,
        [
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=timeout,
        cancel=cancel,
    )
    try:
        data = json.loads(run.stdout_text())
    except ValueError as exc:
        raise ProbeError(f"ffprobe returned unreadable output for {path.name}.") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})
    if not isinstance(streams, list) or not isinstance(fmt, dict):
        raise ProbeError(f"ffprobe returned an unexpected structure for {path.name}.")

    audio: list[AudioStreamInfo] = []
    has_video = False
    has_attached_pic = False
    audio_ordinal = 0
    for raw in streams:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("codec_type")
        if kind == "video":
            # Gomulu kapak resmi bir video izi degildir; iz sayimini bozmasin.
            disposition = raw.get("disposition", {})
            if isinstance(disposition, dict) and disposition.get("attached_pic"):
                has_attached_pic = True
            else:
                has_video = True
            continue
        if kind != "audio":
            continue

        tags = raw.get("tags", {})
        tags = tags if isinstance(tags, dict) else {}
        disposition = raw.get("disposition", {})
        disposition = disposition if isinstance(disposition, dict) else {}

        audio.append(
            AudioStreamInfo(
                audio_index=audio_ordinal,
                index=_as_int(raw.get("index")) or 0,
                codec=str(raw.get("codec_name", "")),
                profile=str(raw["profile"]) if raw.get("profile") else None,
                sample_rate=_as_int(raw.get("sample_rate")) or 0,
                channels=_as_int(raw.get("channels")) or 0,
                channel_layout=str(raw.get("channel_layout", "")),
                sample_fmt=str(raw.get("sample_fmt", "")),
                bits_per_raw_sample=_as_int(raw.get("bits_per_raw_sample")),
                bit_rate=_as_int(raw.get("bit_rate")),
                duration=_as_float(raw.get("duration")),
                language=str(tags.get("language")) if tags.get("language") else None,
                title=str(tags.get("title")) if tags.get("title") else None,
                is_default=bool(disposition.get("default")),
            )
        )
        audio_ordinal += 1

    if not audio:
        raise ProbeError(f"{path.name} contains no audio track.")

    fmt_tags = fmt.get("tags", {})
    fmt_tags = fmt_tags if isinstance(fmt_tags, dict) else {}

    return Probe(
        path=path,
        container=str(fmt.get("format_name", "")),
        size=_as_int(fmt.get("size")) or path.stat().st_size,
        duration=_as_float(fmt.get("duration")),
        audio=tuple(audio),
        has_video=has_video,
        has_attached_pic=has_attached_pic,
        format_bit_rate=_as_int(fmt.get("bit_rate")),
        tags={str(k): str(v) for k, v in fmt_tags.items()},
    )


def default_stream(p: Probe) -> int:
    """Analiz icin varsayilan ses izini secer.

    Sira: `default` isaretli iz, sonra en cok kanalli, sonra ilki. Kanal sayisi
    ikinci olcut cunku bir konser diskinde 5.1 iz genellikle asil kariskimdir
    ve stereo iz onun downmix'idir.
    """
    for s in p.audio:
        if s.is_default:
            return s.audio_index
    best = max(p.audio, key=lambda s: (s.channels, -s.audio_index))
    return best.audio_index


def measure_bit_rate(
    ffprobe: Path,
    p: Probe,
    audio_index: int,
    *,
    cancel: CancelToken | None = None,
    timeout: float = 60.0,
) -> int | None:
    """Bir ses izinin bitrate'ini uc kademede cozer, bulamazsa None dondurur.

    TEMBEL CAGIRILIR. Kademe 1 ve 2 bedavadir ama kademe 3 (paket ornekleme)
    bir arama gerektirir ve maliyeti dosya konumuna ve disk onbellegine baglidir.
    10.6 GB'lik bir MKV'nin ortasina ilk (soguk) arama 12.9 s surdu; ayni arama
    sicakken 1.36 s, dosyanin basina yakin bir arama 0.09 s. Pencere uzunlugu
    neredeyse etkisiz (2 s ile 30 s ayni sureyi ve ayni cevabi verdi), yani
    maliyet tumuyle aramada.

    Sonuc: iz secici doldurulurken TUM izler icin cagrilmaz -- etiket "--"
    gosterir. Yalnizca kullanicinin sectigi iz icin, arayuz thread'i disinda
    cagrilir.
    """
    stream = p.stream(audio_index)
    if stream.bit_rate:
        return stream.bit_rate
    # Konteyner bitrate'i yalnizca dosyada ses disinda HICBIR sey yoksa
    # kullanilabilir. Gomulu kapak resmi olcumu sisirir: 58 MB'lik bir FLAC'ta
    # 296 KB'lik kapak, 815 kbps'i 819 kbps gosteriyordu.
    only_audio = len(p.audio) == 1 and not p.has_video and not p.has_attached_pic
    if only_audio and p.format_bit_rate:
        return p.format_bit_rate

    duration = stream.duration or p.duration
    if not duration or duration <= 1.0:
        return None
    # Bas taraf sessizlik/fade icerebilir; ortadan ornekle.
    start = max(0.0, duration / 2.0 - _SAMPLE_WINDOW_S / 2.0)
    window = min(_SAMPLE_WINDOW_S, duration - start)
    if window <= 1.0:
        return None

    run = run_capture(
        ffprobe,
        [
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-select_streams",
            f"a:{audio_index}",
            "-show_entries",
            "packet=size,duration_time",
            "-read_intervals",
            f"{start:.3f}%+{window:.3f}",
            str(p.path),
        ],
        timeout=timeout,
        cancel=cancel,
        check=False,
    )
    if not run.ok:
        return None
    try:
        packets = json.loads(run.stdout_text()).get("packets", [])
    except ValueError:
        return None
    if not isinstance(packets, list) or not packets:
        return None

    total_bytes = 0
    total_time = 0.0
    for pkt in packets:
        if not isinstance(pkt, dict):
            continue
        size = _as_int(pkt.get("size"))
        span = _as_float(pkt.get("duration_time"))
        if size is None:
            continue
        total_bytes += size
        if span:
            total_time += span

    # Paket suresi gelmiyorsa istenen pencereye duseriz; ornekleme araligi
    # zaten bilindigi icin bu guvenli bir yaklasim.
    span_s = total_time if total_time > 0.5 else window
    if total_bytes == 0:
        return None
    return int(total_bytes * 8 / span_s)
