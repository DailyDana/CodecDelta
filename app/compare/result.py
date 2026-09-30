"""Karsilastirma sonuc modeli.

Model gun birinden COGUL kurulur: `ComparisonSet` bir ya da daha fazla
`ComparisonResult` tasir ve tek bir karsilastirma basitce N=1'dir. Bitrate
taramasi (ayni kaynaktan 96/128/160/192 kbps) ayni modele sonradan eklenir;
tekil kurulmus bir model bunu yeniden yazim gerektirirdi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.align.plan import AlignmentPlan
from app.dsp.accum import BandStats
from app.psycho.nmr import NmrSummary

# Bir bandin S/N'i olcum tabaninin bu kadar yakinindaysa olculen sey codec
# degil zincirin kendisidir. Plan kurali.
FLOOR_MARGIN_DB = 3.0

Status = Literal["measured", "not_comparable", "not_measured"]


@dataclass(frozen=True)
class FileSummary:
    """Karsilastirilan dosyanin raporda gorunecek kimligi."""

    path: Path
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration: float | None


@dataclass(frozen=True)
class BandResult:
    lo_hz: float
    hi_hz: float
    mid: BandStats
    side: BandStats | None
    # Zincirin bu banttaki kendi S/N'i (dB). inf = zincir hic hata eklemiyor.
    floor_db: float

    @property
    def measurable(self) -> bool:
        """Olculen S/N tabandan ayirt edilebiliyor mu?

        Taban tanimsizsa (NaN) ya da S/N tabana `FLOOR_MARGIN_DB`'den yakinsa
        olculen deger codec'i degil zinciri anlatir; ozetten dislanir.
        """
        snr = self.mid.snr_db
        if math.isnan(self.floor_db) or math.isnan(snr):
            return False
        return snr < self.floor_db - FLOOR_MARGIN_DB


@dataclass(frozen=True)
class ComparisonResult:
    reference: FileSummary
    test: FileSummary
    plan: AlignmentPlan
    status: Status
    # Analizin yapildigi ornekleme hizi (iki dosyanin buyuk olani).
    analysis_rate: int
    bands: tuple[BandResult, ...] = ()
    broadband: BandResult | None = None
    # Genis bant skaler kazanc: test'in reference'e gore seviyesi (dB, isaretsiz)
    # ve polaritesi.
    gain_db: float = math.nan
    polarity: int = 1
    # STFT cercevesi ve ortusen ornek sayisi (analiz hizinda).
    frames: int = 0
    samples: int = 0
    # Hizasiz oldugu icin olcume KATILMAYAN sure (s): duzenlenmis dosya, kesik,
    # farkli kapanis. Sifirdan buyukse sonuc yalnizca kalan kisim icindir.
    excluded_s: float = 0.0
    # Gurultu/maske orani ozeti; olculmediyse None.
    nmr: NmrSummary | None = None
    notes: tuple[str, ...] = ()

    @property
    def headline_snr_db(self) -> float:
        """Mansettaki sayi: genis bant inkoherent S/N; olculemezse NaN."""
        if self.broadband is None or not self.broadband.measurable:
            return math.nan
        return self.broadband.mid.snr_db


@dataclass(frozen=True)
class ComparisonSet:
    """Bir ya da daha fazla karsilastirma. Tekil karsilastirma N=1'dir."""

    items: tuple[ComparisonResult, ...]
    # Tarama ekseni ("bitrate" gibi); tekil karsilastirmada None.
    sweep_axis: str | None = None

    @classmethod
    def single(cls, result: ComparisonResult) -> ComparisonSet:
        return cls(items=(result,), sweep_axis=None)
