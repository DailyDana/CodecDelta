"""Bit hizi taramasi: bir kaynagi tek codec'te sabit bir merdivende kodlar ve olcer.

Plan kurali: tarama DAR kapsamli -- sabit merdiven (96/128/160/192/256), tek
codec. Genel bir codec x bitrate x ayar matrisi kurulmaz, kapsam orada patlar.
Sonuc baslangictan beri cogul modelde (`ComparisonSet`, eksen "bitrate").

Kodlanan dosyalar SAKLANIR: tarama sonucundan bir basamak ABX ile dinlenebilir
(eğri sinyal farkini olcer, duyulabilirligi degil; duzlesen egri seffaflik
demek degildir).

Qt IMPORT ETMEZ.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.compare import ladder
from app.compare.pipeline import Track, open_track
from app.compare.result import ComparisonResult, ComparisonSet
from app.core.ffmpeg_runner import CancelToken
from app.encode.matrix import CodecSpec

LADDER_KBPS: tuple[int, ...] = (96, 128, 160, 192, 256)
_MAX_POINTS = 5


@dataclass(frozen=True)
class SweepPoint:
    bitrate_kbps: int
    path: Path
    result: ComparisonResult


@dataclass(frozen=True)
class Sweep:
    codec: str
    points: tuple[SweepPoint, ...]

    def as_set(self) -> ComparisonSet:
        return ComparisonSet(items=tuple(p.result for p in self.points), sweep_axis="bitrate")


def bitrates_for(spec: CodecSpec) -> tuple[int, ...]:
    """Codec'in sundugu bitrate'lerle plandaki merdivenin kesisimi.

    Kesisim bossa (orn. AC-3 192..640) codec'in kendi listesinden en fazla bes
    deger, esit aralikla.
    """
    if not spec.bitrates:
        return ()
    common = tuple(k for k in LADDER_KBPS if k in spec.bitrates)
    if len(common) >= 3:
        return common
    values = spec.bitrates
    if len(values) <= _MAX_POINTS:
        return tuple(values)
    step = (len(values) - 1) / (_MAX_POINTS - 1)
    return tuple(values[round(i * step)] for i in range(_MAX_POINTS))


def run(
    ffmpeg: Path,
    ffprobe: Path,
    reference: Track,
    spec: CodecSpec,
    workdir: Path,
    *,
    bitrates: Sequence[int] | None = None,
    cancel: CancelToken | None = None,
    stage: Callable[[str], None] | None = None,
) -> Sweep:
    """Taramayi calistirir. Dosyalar `workdir`de kalir (ABX icin)."""
    rates = tuple(bitrates) if bitrates is not None else bitrates_for(spec)
    if not rates:
        raise ValueError(f"{spec.name} has no bitrate mode to sweep")
    workdir.mkdir(parents=True, exist_ok=True)
    rungs = ladder.build(
        ffmpeg,
        ffprobe,
        reference,
        workdir,
        codec=spec.encoder,
        rungs=rates,
        keep_files=True,
        extension=spec.extension,
        prefix="sweep",
        cancel=cancel,
        stage=stage,
    )
    points = tuple(
        SweepPoint(
            r.bitrate_kbps,
            workdir / f"sweep_{spec.encoder}_{r.bitrate_kbps}k.{spec.extension}",
            r.result,
        )
        for r in rungs
    )
    return Sweep(codec=spec.key, points=points)


def track_of(ffprobe: Path, point: SweepPoint) -> Track:
    """ABX icin basamagin dosyasini acar."""
    return open_track(ffprobe, point.path)
