"""ffmpeg ile kodlama isi -- capa merdiveninin ve (ileride) encode panelinin motoru.

Bir kaynagi tek bir hedef dosyaya kodlar. Panel duzeyindeki secenek matrisi
(VBR modu, uygulama tipi, kalite olcegi) burada degil; burada yalnizca her
kodlamanin ortak, degismeyen kisimlari var.

Opus ve ornekleme hizi: libopus yalnizca 48/24/16/12/8 kHz kabul eder ve 44.1
kHz'lik bir kaynagi ffmpeg SESSIZCE kendi varsayilan resampler'iyla 48'e
cevirir. O zaman codec'i degil iki farkli resampler'i olcersiniz. Bu yuzden
Opus'a giden 44.1 kHz kaynak, analiz zincirinin kullandigi AYNI soxr
ayarlariyla (`ResampleCfg`, cutoff dahil) 48'e cevrilir; karsilastirma
referansi da ayni ayarla cevrildigi icin iki taraf birebir ortusur ve fark
yalnizca codec'ten gelir.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from app.core.errors import CodecDeltaError, FFmpegFailedError
from app.core.ffmpeg_runner import CancelToken, StderrCollector, kill_tree, run_capture, spawn
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, ResampleCfg
from app.core.messages import Message

# libopus'un kabul ettigi hizlar; digerleri 48 kHz'e cevrilir.
OPUS_RATES = frozenset({48000, 24000, 16000, 12000, 8000})

# Uzun dosyada kodlama dakikalar surebilir; run_capture'in 60 s varsayilani az.
DEFAULT_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class EncodeJob:
    source: Path
    output: Path
    # ffmpeg kodlayici adi: libopus, libmp3lame, aac, libvorbis, flac ...
    codec: str
    bitrate_kbps: int | None = None
    stream_index: int = 0
    # Kaynagin ornekleme hizi; Opus icin yeniden ornekleme kararini belirler.
    source_rate: int | None = None
    # Kodlayiciya ozel ek argumanlar, oldugu gibi eklenir.
    extra: tuple[str, ...] = ()
    # `prepare` sonucu: kodlamadan once acikca cevrilecek hiz ve ek filtreler.
    target_rate: int | None = None
    filters: tuple[str, ...] = ()


def build_args(
    job: EncodeJob, *, resample: ResampleCfg = DEFAULT_RESAMPLE, progress: bool = False
) -> list[str]:
    """Kodlama icin ffmpeg arguman listesi. `-vn -sn -dn` ve `-map` zorunlu."""
    args: list[str] = [
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        str(job.source),
        "-vn",
        "-sn",
        "-dn",
        "-map",
        f"0:a:{job.stream_index}",
    ]
    chain = list(job.filters)
    if job.target_rate is not None:
        chain.append(resample.filter_expr(job.target_rate))
    elif (
        job.codec == "libopus" and job.source_rate is not None and job.source_rate not in OPUS_RATES
    ):
        chain.append(resample.filter_expr(48000))
    if chain:
        args += ["-af", ",".join(chain)]
    args += ["-c:a", job.codec]
    if job.bitrate_kbps is not None:
        args += ["-b:a", f"{job.bitrate_kbps}k"]
    args += list(job.extra)
    if progress:
        # Anahtar=deger satirlari stdout'a; `-nostats` stderr'i sade tutar.
        args += ["-progress", "pipe:1", "-nostats"]
    args.append(str(job.output))
    return args


def run(
    ffmpeg: Path,
    job: EncodeJob,
    *,
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    timeout: float = DEFAULT_TIMEOUT_S,
    cancel: CancelToken | None = None,
) -> Path:
    """Isi calistirir; basarisizlikta `FFmpegFailedError`. Cikti yolunu dondurur."""
    job.output.parent.mkdir(parents=True, exist_ok=True)
    run_capture(ffmpeg, build_args(job, resample=resample), timeout=timeout, cancel=cancel)
    return job.output


def parse_progress(line: str) -> float | None:
    """`-progress` satirindan islenmis sure (s). Ilgisiz satirda None.

    ffmpeg hem `out_time_us` hem (tarihsel bir hatayla yine mikro saniye olan)
    `out_time_ms` yazar; ikisi de mikro saniye kabul edilir. Baslangicta
    "N/A" gelebilir.
    """
    key, _, value = line.strip().partition("=")
    if key not in ("out_time_us", "out_time_ms"):
        return None
    try:
        micro = int(value)
    except ValueError:
        return None
    return max(0.0, micro / 1_000_000.0)


def run_with_progress(
    ffmpeg: Path,
    job: EncodeJob,
    *,
    duration: float | None,
    on_progress: Callable[[float], None],
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    cancel: CancelToken | None = None,
) -> Path:
    """Kodlar ve ilerlemeyi 0..1 olarak bildirir. Sure bilinmiyorsa bildirim yok.

    Iptal, satirlar arasinda yoklanir (`-progress` ~0.5 s'de bir yazar) ve
    surec agaci oldurulur; yarim kalan cikti dosyasi silinir.
    """
    job.output.parent.mkdir(parents=True, exist_ok=True)
    args = build_args(job, resample=resample, progress=True)
    proc = spawn(ffmpeg, args)
    collector = StderrCollector(proc.stderr)
    try:
        assert proc.stdout is not None
        buffer = b""
        while True:
            if cancel is not None and cancel.cancelled:
                kill_tree(proc)
                break
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for raw in lines:
                seconds = parse_progress(raw.decode("ascii", "replace"))
                if seconds is not None and duration:
                    on_progress(min(1.0, seconds / duration))
        proc.wait()
    except BaseException:
        kill_tree(proc)
        raise
    finally:
        collector.join()
    if cancel is not None and cancel.cancelled:
        job.output.unlink(missing_ok=True)
        cancel.raise_if_cancelled()
    if proc.returncode != 0:
        job.output.unlink(missing_ok=True)
        raise FFmpegFailedError((str(ffmpeg), *args), proc.returncode, collector.tail())
    on_progress(1.0)
    return job.output


# -- kodlayici sinirlari (denetim D14, D15) -----------------------------------------
#
# ffmpeg cogu kodlayici icin kabul ettigi hizlari ve kanal duzenlerini
# `-h encoder=X` ciktisinda bildirir. Bildirmediklerinde sinir yine vardir ve
# kodlama ham bir ffmpeg hatasiyla duser; bunlar asagida OLCULDU (bu makinedeki
# ffmpeg 8 build'i, Ekim 2026):
#   - wmav2: 48 kHz ustu "sample rate is too high", 2 kanaldan fazlasi
#     "too many channels".
#   - libvorbis: bit hizi modunda 88.2/96 kHz "encoder setup failed"; kalite
#     modu 192 kHz'e kadar calisiyor.
#   - libopus: "(side)"/"(wide)" duzenleri "Invalid channel layout"; ayni kanal
#     sayili duz duzenler (5.1, 7.1) calisiyor.
# Desteklenmeyen hiza ffmpeg'in VARSAYILAN resampler'i sessizce cevirir (96 kHz
# MP3/AC-3/MP2 -> 48 kHz); Opus icin ozellikle kacinilan tuzak buydu. Artik
# donusum her zaman acikca, analiz zincirinin soxr ayariyla yapilir ve not edilir.

_LAYOUT_CHANNELS = {
    "mono": 1,
    "stereo": 2,
    "2.1": 3,
    "3.0": 3,
    "3.0(back)": 3,
    "3.1": 4,
    "4.0": 4,
    "quad": 4,
    "quad(side)": 4,
    "4.1": 5,
    "5.0": 5,
    "5.0(side)": 5,
    "5.1": 6,
    "5.1(side)": 6,
    "6.0": 6,
    "6.1": 7,
    "6.1(back)": 7,
    "7.0": 7,
    "7.1": 8,
    "7.1(wide)": 8,
}
_LAYOUT_TOKEN = re.compile(r"(\d+) channels \([^)]*\)|(\S+)")

# Bildirilmeyen, olculmus sinirlar.
_MEASURED_RATES = {"wmav2": (48000, 44100, 32000, 22050, 16000, 11025, 8000)}
_MEASURED_MAX_CHANNELS = {"wmav2": 2}
_VORBIS_BITRATE_MAX_RATE = 48000
# Opus'un kabul ettigi duz karsiliklar; kanal sirasi ayni, yalnizca etiket farkli.
_OPUS_RELABEL = {"5.1(side)": "5.1", "5.0(side)": "5.0", "quad(side)": "quad"}
_OPUS_REJECTED = frozenset({"7.1(wide)", "6.1(back)"})


@dataclass(frozen=True)
class EncoderLimits:
    # Kabul edilen hizlar; None = her hiz.
    rates: tuple[int, ...] | None = None
    # En fazla kanal; None = sinir bilinmiyor.
    max_channels: int | None = None


class EncodeUnsupportedError(CodecDeltaError):
    """Secilen kodlayici bu kaynagi kodlayamaz; is baslatilmadan soylenir."""


def parse_limits(help_text: str) -> EncoderLimits:
    """`ffmpeg -h encoder=X` ciktisindan hiz ve kanal sinirini cikarir."""
    rates: tuple[int, ...] | None = None
    max_channels: int | None = None
    for line in help_text.splitlines():
        key, _, value = line.strip().partition(":")
        if key == "Supported sample rates":
            rates = tuple(int(v) for v in value.split() if v.isdigit()) or None
        elif key == "Supported channel layouts":
            counts = []
            for numbered, name in _LAYOUT_TOKEN.findall(value):
                if numbered:
                    counts.append(int(numbered))
                elif name in _LAYOUT_CHANNELS:
                    counts.append(_LAYOUT_CHANNELS[name])
            max_channels = max(counts) if counts else None
    return EncoderLimits(rates, max_channels)


@functools.cache
def encoder_limits(ffmpeg: Path, encoder: str) -> EncoderLimits:
    """Kodlayicinin bildirdigi sinirlar + olculmus sinirlar (onbellekli)."""
    try:
        declared = parse_limits(
            run_capture(
                ffmpeg, ["-hide_banner", "-h", f"encoder={encoder}"], check=False
            ).stdout.decode("utf-8", "replace")
        )
    except (CodecDeltaError, OSError):
        declared = EncoderLimits()
    return EncoderLimits(
        declared.rates or _MEASURED_RATES.get(encoder),
        declared.max_channels or _MEASURED_MAX_CHANNELS.get(encoder),
    )


def _target_rate(source: int, rates: tuple[int, ...]) -> int:
    """Kaynaga en yakin kabul edilen hiz: once kaynagin ustundeki en kucuk
    (44.1 -> Opus 48, bilgi kaybi yok), yoksa en buyuk (96 -> MP3 48)."""
    above = [r for r in rates if r >= source]
    return min(above) if above else max(rates)


def prepare(
    job: EncodeJob,
    limits: EncoderLimits,
    *,
    channels: int,
    channel_layout: str = "",
) -> tuple[EncodeJob, tuple[str, ...]]:
    """Isi kodlayicinin sinirlarina gore hazirlar: hedef hiz ve kanal etiketi.

    Dondurulen notlar kullaniciya gosterilir. Kodlayici kaynagi hic
    kodlayamiyorsa (kanal sayisi) `EncodeUnsupportedError`: plan geregi
    otomatik downmix YOK, olcumu kirletir.
    """
    notes: list[str] = []
    if limits.max_channels is not None and channels > limits.max_channels:
        raise EncodeUnsupportedError(
            Message(
                "encode.too_many_channels",
                "{encoder} encodes at most {limit} channels; the source has {channels}. "
                "Choose another codec.",
                encoder=job.codec,
                limit=limits.max_channels,
                channels=channels,
            )
        )
    filters: tuple[str, ...] = ()
    if job.codec == "libopus":
        if channel_layout in _OPUS_REJECTED:
            raise EncodeUnsupportedError(
                Message(
                    "encode.layout",
                    "{encoder} does not accept the {layout} channel layout. Choose another codec.",
                    encoder=job.codec,
                    layout=channel_layout,
                )
            )
        if channel_layout in _OPUS_RELABEL:
            plain = _OPUS_RELABEL[channel_layout]
            filters = (f"channelmap=channel_layout={plain}",)
            notes.append(
                Message(
                    "encode.relabelled",
                    "channel layout {layout} relabelled as {plain} for {encoder} (same "
                    "channels, side speakers named as back)",
                    layout=channel_layout,
                    plain=plain,
                    encoder=job.codec,
                )
            )
    rates = limits.rates
    vorbis_bitrate = job.codec == "libvorbis" and job.bitrate_kbps is not None
    if vorbis_bitrate and job.source_rate and job.source_rate > _VORBIS_BITRATE_MAX_RATE:
        rates = (_VORBIS_BITRATE_MAX_RATE,)
    target: int | None = None
    if rates and job.source_rate and job.source_rate not in rates:
        target = _target_rate(job.source_rate, rates)
        notes.append(
            Message(
                "encode.resampled",
                "{encoder} does not accept {source:g} kHz: converted to {target:g} kHz with soxr "
                "before encoding (the comparison resamples the same way)",
                encoder=job.codec,
                source=job.source_rate / 1000,
                target=target / 1000,
            )
        )
    return replace(job, target_rate=target, filters=filters), tuple(notes)
