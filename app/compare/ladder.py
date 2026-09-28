"""Capa merdiveni -- mansetteki hukmu ureten yontem.

Mutlak bir psikoakustik esik yerine, kullanicinin KENDI referans dosyasindan
bilinen bitrate'lerde bir merdiven kodlanir, her basamak ayni boru hattiyla
olculur ve test dosyasi bu merdivende konumlandirilir:

    "Olculen fark, ayni parcanin Opus ~112 kbps kodlamasina denk:
     128k'dan daha az bozulmus, 96k'dan daha fazla."

Neden bu yontem: mutlak esik kalibre edilemedi (bkz. DECISIONS, NMR: ayni
dosyada tonalite secimine gore 15 dB yayilim). Merdiven ise goreli: model
hatasi iki tarafa da ayni sekilde bulasir ve buyuk olcude sadelesir; klasik ile
metal arasindaki fark kullanicinin kendi materyalinde otomatik hesaba girer.

Iki eksende konumlandirilir: genis bant codec S/N (mansettaki sayi) ve MPEG
toplam NMR p95. Ikisi bir basamaktan fazla ayrisirsa bu soylenir; hangisinin
dogru oldugu iddia edilmez.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.compare.pipeline import Track, compare, open_track
from app.compare.result import ComparisonResult
from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, ResampleCfg
from app.encode.jobs import EncodeJob
from app.encode.jobs import run as run_encode

# Varsayilan basamaklar (kbps). Opus icin tipik "duyulur / sinirda / seffaf" araligi.
DEFAULT_RUNGS: tuple[int, ...] = (64, 96, 128, 192)
DEFAULT_CODEC = "libopus"
_EXTENSION = {"libopus": "opus", "libmp3lame": "mp3", "aac": "m4a", "libvorbis": "ogg"}

Position = Literal["below", "within", "above", "unknown"]


@dataclass(frozen=True)
class Rung:
    bitrate_kbps: int
    snr_db: float
    nmr_p95_db: float
    result: ComparisonResult


@dataclass(frozen=True)
class Placement:
    """Bir eksende merdivendeki konum."""

    position: Position
    # Log-bitrate uzerinde dogrusal ara degerleme; merdiven disindaysa None.
    equivalent_kbps: float | None
    # Kusatan basamaklar (kbps); disaridaysa yalnizca biri dolu.
    lower_kbps: int | None
    upper_kbps: int | None
    # Eksen bu parcada bitrate ile monoton mu? Degilse konum guvenilmez.
    monotonic: bool


@dataclass(frozen=True)
class LadderVerdict:
    codec: str
    rungs: tuple[Rung, ...]
    by_snr: Placement
    by_nmr: Placement
    notes: tuple[str, ...]

    @property
    def headline(self) -> str:
        """Rapor cumlesi. Iki eksen ayrisiyorsa ikisini de soyler."""
        name = _codec_label(self.codec)
        primary = _describe(self.by_snr, name)
        if self.by_nmr.position == "unknown" or _agree(self.by_snr, self.by_nmr):
            return primary
        return f"{primary} By masking (NMR) instead: {_describe(self.by_nmr, name, short=True)}"


def _codec_label(codec: str) -> str:
    return {"libopus": "Opus", "libmp3lame": "MP3", "aac": "AAC", "libvorbis": "Vorbis"}.get(
        codec, codec
    )


def _describe(placement: Placement, name: str, *, short: bool = False) -> str:
    if placement.position == "unknown":
        return "the measured difference could not be placed on the ladder."
    if placement.position == "above":
        return f"less distorted than {name} at {placement.lower_kbps} kbps, the top of the ladder."
    if placement.position == "below":
        return (
            f"more distorted than {name} at {placement.upper_kbps} kbps, the bottom of the ladder."
        )
    assert placement.equivalent_kbps is not None
    core = f"equivalent to {name} at roughly {placement.equivalent_kbps:.0f} kbps"
    if short:
        return core + "."
    text = (
        f"The measured difference is {core}: less distorted than {placement.upper_kbps} kbps, "
        f"more than {placement.lower_kbps} kbps."
    )
    if not placement.monotonic:
        text += " The ladder is not monotonic for this track, so treat this as approximate."
    return text


def _agree(a: Placement, b: Placement) -> bool:
    if a.position != b.position:
        return False
    if a.equivalent_kbps is None or b.equivalent_kbps is None:
        return True
    # Bir basamaktan az fark: log2 oraninda 0.5 (96 -> 128 arasi ~0.42)
    return abs(math.log2(a.equivalent_kbps / b.equivalent_kbps)) < 0.5


def place(values: Sequence[float], bitrates: Sequence[int], test_value: float) -> Placement:
    """`test_value`i, basamak degerleri `values` olan merdivende konumlandirir.

    Eksen "buyuk = daha iyi" olmali (S/N). NMR icin cagiran taraf isareti
    cevirir. Ara degerleme log2(bitrate) uzerinde dogrusal.
    """
    if math.isnan(test_value) or len(values) < 2 or any(math.isnan(v) for v in values):
        return Placement("unknown", None, None, None, False)
    order = np.argsort(bitrates)
    rates = np.asarray(bitrates, dtype=np.float64)[order]
    axis = np.asarray(values, dtype=np.float64)[order]
    monotonic = bool(np.all(np.diff(axis) > 0.0))
    if test_value >= axis.max():
        return Placement("above", None, int(rates[int(np.argmax(axis))]), None, monotonic)
    if test_value <= axis.min():
        return Placement("below", None, None, int(rates[int(np.argmin(axis))]), monotonic)
    # Monoton degilse, np.interp icin ekseni monotonlastir (kumulatif maksimum).
    usable = np.maximum.accumulate(axis)
    log_rate = float(np.interp(test_value, usable, np.log2(rates)))
    kbps = 2.0**log_rate
    lower = int(rates[np.searchsorted(rates, kbps, side="right") - 1])
    upper = int(rates[min(np.searchsorted(rates, kbps, side="right"), rates.size - 1)])
    return Placement("within", kbps, lower, upper, monotonic)


def build(
    ffmpeg: Path,
    ffprobe: Path,
    reference: Track,
    workdir: Path,
    *,
    codec: str = DEFAULT_CODEC,
    rungs: Sequence[int] = DEFAULT_RUNGS,
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    keep_files: bool = False,
    cancel: CancelToken | None = None,
) -> tuple[Rung, ...]:
    """Referansi her basamakta kodlar ve olcer. Dosyalar `workdir`e yazilir."""
    extension = _EXTENSION.get(codec, "mka")
    out: list[Rung] = []
    for bitrate in rungs:
        path = workdir / f"ladder_{codec}_{bitrate}k.{extension}"
        run_encode(
            ffmpeg,
            EncodeJob(
                source=reference.path,
                output=path,
                codec=codec,
                bitrate_kbps=bitrate,
                stream_index=reference.stream_index,
                source_rate=reference.stream.sample_rate,
            ),
            resample=resample,
            cancel=cancel,
        )
        try:
            result = compare(
                ffmpeg, reference, open_track(ffprobe, path), resample=resample, cancel=cancel
            )
        finally:
            if not keep_files:
                path.unlink(missing_ok=True)
        out.append(
            Rung(
                bitrate_kbps=bitrate,
                snr_db=result.headline_snr_db,
                nmr_p95_db=result.nmr.p95_db if result.nmr is not None else math.nan,
                result=result,
            )
        )
    return tuple(out)


def judge(
    rungs: Sequence[Rung], test: ComparisonResult, *, codec: str = DEFAULT_CODEC
) -> LadderVerdict:
    """Test sonucunu hazir bir merdivende konumlandirir."""
    notes: list[str] = []
    bitrates = [r.bitrate_kbps for r in rungs]
    by_snr = place([r.snr_db for r in rungs], bitrates, test.headline_snr_db)
    # NMR: dusuk = iyi; isaret cevrilir.
    test_nmr = test.nmr.p95_db if test.nmr is not None else math.nan
    by_nmr = place([-r.nmr_p95_db for r in rungs], bitrates, -test_nmr)
    if math.isnan(test.headline_snr_db):
        notes.append("headline SNR is not measurable (at or above the measurement floor)")
    if not by_snr.monotonic and by_snr.position != "unknown":
        notes.append("codec SNR is not monotonic in bitrate on this ladder")
    if not by_nmr.monotonic and by_nmr.position != "unknown":
        notes.append("NMR is not monotonic in bitrate on this ladder")
    return LadderVerdict(
        codec=codec, rungs=tuple(rungs), by_snr=by_snr, by_nmr=by_nmr, notes=tuple(notes)
    )
