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

from dataclasses import dataclass
from pathlib import Path

from app.core.ffmpeg_runner import CancelToken, run_capture
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, ResampleCfg

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


def build_args(job: EncodeJob, *, resample: ResampleCfg = DEFAULT_RESAMPLE) -> list[str]:
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
    if job.codec == "libopus" and job.source_rate is not None and job.source_rate not in OPUS_RATES:
        args += ["-af", resample.filter_expr(48000)]
    args += ["-c:a", job.codec]
    if job.bitrate_kbps is not None:
        args += ["-b:a", f"{job.bitrate_kbps}k"]
    args += list(job.extra)
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
