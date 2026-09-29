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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import FFmpegFailedError
from app.core.ffmpeg_runner import CancelToken, StderrCollector, kill_tree, run_capture, spawn
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
    if job.codec == "libopus" and job.source_rate is not None and job.source_rate not in OPUS_RATES:
        args += ["-af", resample.filter_expr(48000)]
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
