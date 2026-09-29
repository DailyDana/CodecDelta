"""Kodlama matrisi: hangi codec, hangi ayarlarla, hangi kisitlarla.

Tablo koda yazilidir ama KULLANILABILIRLIK calisma aninda olculur: bir
kodlayici bu ffmpeg derlemesinde yoksa (`Capabilities.encoders`) listede gri
gorunur ve sebebi yazar. Boylece baska bir ffmpeg ile ayni arayuz dogru
davranir; hicbir secenek "tiklayinca hata veren" bir dugme olmaz.

Her kaydin kisitlari DOGRULANDI (bu makinedeki build, 2026-09):
- libopus yalnizca 48/24/16/12/8 kHz kabul eder; 44.1 kaynak analiz
  zincirinin soxr ayarlariyla 48'e cevrilir (bkz. `jobs.build_args`).
- aac_mf (Windows Media Foundation) hicbir AVOption tasimaz ve yalnizca s16
  girdi kabul eder; `-sample_fmt s16` eklenir.
- ffmpeg'in aac kodlayicisinda PNS (algisal gurultu ikamesi) gurultuyu
  SENTEZLER: kulaga dogru gelir ama S/N olcumunu anlamsizlastirir. Varsayilan
  kapali, acilirsa arayuz uyarir.
- libfdk_aac bu build'de YOK (`--disable-libfdk-aac`); listede gri.

Bu modul Qt IMPORT ETMEZ.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from app.core.messages import Message

Mode = Literal["bitrate", "quality", "lossless"]


@dataclass(frozen=True)
class Option:
    """Kodlayiciya ozel bir secenek: ffmpeg argumani ve secilebilir degerler."""

    key: str
    label: str
    arg: str
    # (deger, gorunen ad). Deger None ise arguman hic eklenmez (kodlayici varsayilani).
    choices: tuple[tuple[str | None, str], ...]
    default: str | None
    # Bu degerler secildiginde gosterilecek uyari.
    warn_on: frozenset[str] = frozenset()
    warning: str = ""


@dataclass(frozen=True)
class Quality:
    """Kalite olcegi (VBR). `arg` degeri tamsayi olarak alir."""

    arg: str
    low: int
    high: int
    default: int
    # Kullaniciya gosterilecek ad bicimi, orn. "V{value}" ya da "q{value}".
    label: str
    # Buyuk deger daha IYI mi? (Vorbis evet, LAME V olcegi hayir.)
    higher_is_better: bool


@dataclass(frozen=True)
class CodecSpec:
    key: str
    name: str
    encoder: str
    extension: str
    lossless: bool = False
    bitrates: tuple[int, ...] = ()
    default_bitrate: int | None = None
    quality: Quality | None = None
    options: tuple[Option, ...] = ()
    # Her zaman eklenen argumanlar (kisit geregi).
    fixed: tuple[str, ...] = ()
    note: str = ""
    # Kodlayici eksikse gosterilecek sebep.
    missing_reason: str = ""

    @property
    def modes(self) -> tuple[Mode, ...]:
        if self.lossless:
            return ("lossless",)
        modes: list[Mode] = []
        if self.bitrates:
            modes.append("bitrate")
        if self.quality is not None:
            modes.append("quality")
        return tuple(modes)


ON = Message("encode.on", "on")
OFF = Message("encode.off", "off")
DEFAULT = Message("encode.default", "default")
LOSSLESS_NOTE = Message(
    "encode.lossless_note", "Lossless: the comparison should come out bit-identical (a control)."
)

SPECS: tuple[CodecSpec, ...] = (
    CodecSpec(
        key="opus",
        name="Opus",
        encoder="libopus",
        extension="opus",
        bitrates=(32, 48, 64, 80, 96, 112, 128, 160, 192, 256),
        default_bitrate=128,
        options=(
            Option(
                "vbr",
                Message("encode.opt.vbr", "VBR"),
                "-vbr",
                (
                    ("on", ON),
                    ("constrained", Message("encode.constrained", "constrained")),
                    ("off", Message("encode.cbr", "off (CBR)")),
                ),
                "on",
            ),
            Option(
                "application",
                Message("encode.opt.application", "Application"),
                "-application",
                (
                    ("audio", Message("encode.music", "music")),
                    ("lowdelay", Message("encode.lowdelay", "low delay")),
                    ("voip", Message("encode.speech", "speech (VoIP)")),
                ),
                "audio",
            ),
            Option(
                "frame_duration",
                Message("encode.opt.frame", "Frame (ms)"),
                "-frame_duration",
                (("20", "20"), ("10", "10"), ("40", "40"), ("60", "60")),
                "20",
            ),
        ),
        note=Message(
            "encode.note.opus",
            "Opus runs at 48 kHz only; a 44.1 kHz source is resampled with the analysis "
            "chain's own settings.",
        ),
    ),
    CodecSpec(
        key="aac",
        name="AAC (ffmpeg)",
        encoder="aac",
        extension="m4a",
        bitrates=(96, 128, 160, 192, 256, 320),
        default_bitrate=192,
        options=(
            Option(
                "aac_coder",
                Message("encode.opt.coder", "Coder"),
                "-aac_coder",
                ((None, DEFAULT), ("twoloop", "twoloop"), ("fast", "fast")),
                None,
            ),
            Option(
                "aac_pns",
                Message("encode.opt.pns", "Noise substitution"),
                "-aac_pns",
                (("0", OFF), ("1", ON)),
                "0",
                warn_on=frozenset({"1"}),
                warning=Message(
                    "encode.pns",
                    "Noise substitution synthesises noise: it can sound right but makes SNR "
                    "meaningless.",
                ),
            ),
        ),
    ),
    CodecSpec(
        key="aac_mf",
        name="AAC (Media Foundation)",
        encoder="aac_mf",
        extension="m4a",
        bitrates=(96, 128, 160, 192),
        default_bitrate=192,
        fixed=("-sample_fmt", "s16"),
        note=Message(
            "encode.note.aac_mf",
            "Windows' own AAC encoder: bitrate only, no other options, 16-bit input.",
        ),
    ),
    CodecSpec(
        key="fdk_aac",
        name="AAC (Fraunhofer FDK)",
        encoder="libfdk_aac",
        extension="m4a",
        bitrates=(96, 128, 160, 192, 256),
        default_bitrate=192,
        missing_reason=Message(
            "encode.missing.fdk", "Not in this ffmpeg build (licence: --disable-libfdk-aac)."
        ),
    ),
    CodecSpec(
        key="mp3",
        name="MP3 (LAME)",
        encoder="libmp3lame",
        extension="mp3",
        bitrates=(96, 128, 160, 192, 224, 256, 320),
        default_bitrate=192,
        quality=Quality("-q:a", 0, 9, 2, "V{value}", higher_is_better=False),
        options=(
            Option(
                "joint_stereo",
                Message("encode.opt.joint", "Joint stereo"),
                "-joint_stereo",
                (("1", ON), ("0", OFF)),
                "1",
            ),
        ),
    ),
    CodecSpec(
        key="vorbis",
        name="Vorbis",
        encoder="libvorbis",
        extension="ogg",
        bitrates=(96, 128, 160, 192, 256, 320),
        default_bitrate=160,
        quality=Quality("-q:a", -1, 10, 5, "q{value}", higher_is_better=True),
    ),
    CodecSpec(
        key="ac3",
        name="AC-3",
        encoder="ac3",
        extension="ac3",
        bitrates=(192, 256, 320, 384, 448, 640),
        default_bitrate=448,
    ),
    CodecSpec(
        key="eac3",
        name="E-AC-3",
        encoder="eac3",
        extension="eac3",
        bitrates=(96, 128, 192, 256, 384, 640),
        default_bitrate=256,
    ),
    CodecSpec(
        key="mp2",
        name="MP2",
        encoder="mp2",
        extension="mp2",
        bitrates=(128, 160, 192, 224, 256, 320, 384),
        default_bitrate=256,
    ),
    CodecSpec(
        key="wma",
        name="WMA v2",
        encoder="wmav2",
        extension="wma",
        bitrates=(64, 96, 128, 160, 192),
        default_bitrate=128,
    ),
    CodecSpec(
        key="flac",
        name="FLAC",
        encoder="flac",
        extension="flac",
        lossless=True,
        options=(
            Option(
                "compression_level",
                Message("encode.opt.level", "Compression level"),
                "-compression_level",
                tuple((str(n), str(n)) for n in range(0, 13)),
                "5",
            ),
        ),
        note=LOSSLESS_NOTE,
    ),
    CodecSpec(
        key="alac", name="ALAC", encoder="alac", extension="m4a", lossless=True, note=LOSSLESS_NOTE
    ),
    CodecSpec(
        key="wavpack",
        name="WavPack",
        encoder="wavpack",
        extension="wv",
        lossless=True,
        note=LOSSLESS_NOTE,
    ),
)


def by_key(key: str) -> CodecSpec:
    for spec in SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def availability(spec: CodecSpec, encoders: frozenset[str]) -> str | None:
    """Kullanilabilirse None, degilse sebep."""
    if spec.encoder in encoders:
        return None
    return spec.missing_reason or Message(
        "encode.missing", "The {encoder} encoder is not in this ffmpeg build.", encoder=spec.encoder
    )


@dataclass(frozen=True)
class EncodeSettings:
    """Kullanicinin secimi. `options` eksik anahtarlar icin spec varsayilani kullanilir."""

    codec: str
    mode: Mode
    bitrate_kbps: int | None = None
    quality: int | None = None
    options: Mapping[str, str | None] = field(default_factory=dict)

    @property
    def spec(self) -> CodecSpec:
        return by_key(self.codec)


def validate(settings: EncodeSettings) -> None:
    """Secimin spec ile uyumlu oldugunu dogrular; degilse ValueError."""
    spec = settings.spec
    if settings.mode not in spec.modes:
        raise ValueError(f"{spec.name} does not support {settings.mode} mode")
    if settings.mode == "bitrate" and settings.bitrate_kbps is None:
        raise ValueError("bitrate mode needs a bitrate")
    if settings.mode == "quality":
        q = spec.quality
        if q is None or settings.quality is None or not q.low <= settings.quality <= q.high:
            raise ValueError("quality out of range")
    known = {o.key: o for o in spec.options}
    for key, value in settings.options.items():
        if key not in known:
            raise ValueError(f"unknown option {key} for {spec.name}")
        if value not in {c[0] for c in known[key].choices}:
            raise ValueError(f"invalid value {value!r} for {key}")


def extra_args(settings: EncodeSettings) -> tuple[str, ...]:
    """Spec'e ozel sabit argumanlar + secilen secenekler + kalite olcegi."""
    validate(settings)
    spec = settings.spec
    args: list[str] = list(spec.fixed)
    for option in spec.options:
        value = settings.options.get(option.key, option.default)
        if value is not None:
            args += [option.arg, value]
    if settings.mode == "quality" and spec.quality is not None and settings.quality is not None:
        args += [spec.quality.arg, str(settings.quality)]
    return tuple(args)


def warnings(settings: EncodeSettings) -> tuple[str, ...]:
    out: list[str] = []
    for option in settings.spec.options:
        value = settings.options.get(option.key, option.default)
        if value is not None and value in option.warn_on:
            out.append(option.warning)
    return tuple(out)


def describe(settings: EncodeSettings) -> str:
    """Dosya adinda kullanilacak kisa etiket: "opus128k", "mp3V2", "flac"."""
    spec = settings.spec
    if settings.mode == "bitrate":
        return f"{spec.key}{settings.bitrate_kbps}k"
    if settings.mode == "quality" and spec.quality is not None:
        return spec.key + spec.quality.label.format(value=settings.quality)
    return spec.key
