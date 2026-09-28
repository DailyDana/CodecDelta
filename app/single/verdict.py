"""Referanssiz hukum -- "bu dosya kayipsiz bir kaynakla tutarli mi".

Tek bir "suphe skoru" YOK: agirliklar uydurma olurdu. Cikti, insan-okunur
kanit ve karsi-kanit listeleri ve dort kovadan biri:

    consistent_lossless   kayipli iz bulunamadi
    consistent_lossy      kayipli izler var, karsi-kanit yok
    undetermined          kanitlar celisiyor ya da yargilayacak icerik yok
    (not_applicable)      dosya zaten kayipli bir formatta

Dil kurali: asla "bu bir transcode" ya da "kayipsiz oldugu kanitlandi"
denmez. Arac koken kanitlayamaz, imza gosterir. Sinir olculdu: AAC 256 ve
MP3 V0'in bazi parcalari spektral olarak gercekle ayirt edilemez ve
"kayipsiz kaynakla tutarli" cikar (bkz. thresholds.py).

Her esik `thresholds.py`de, kaynagiyla. Buradaki mantik yalnizca kanitlarin
nasil birlestigidir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.bitstream import flac as flac_bitstream
from app.core.ffmpeg_runner import CancelToken
from app.core.probe import Probe
from app.single import spectral, thresholds
from app.single.spectral import SpectralEvidence

Bucket = Literal["consistent_lossless", "consistent_lossy", "undetermined", "not_applicable"]

# Tek dosya analizinde cozulecek kesit: ortadan bu kadar saniye. Ozellikler
# muzik boyunca duragan (kesim, taban), 30 s yeter; batch taramada daha kisa.
DEFAULT_EXCERPT_S = 30.0
# Bas ve son bu kesir kadar atlanir (fade, alkis, gizli parca).
_SKIP_EDGE_FRACTION = 0.05
# Bunun altindaki FLAC blok boyu olagan bir kodlayicidan gelmez (bkz. judge_container).
_UNUSUAL_BLOCKSIZE = 256


@dataclass(frozen=True)
class Verdict:
    bucket: Bucket
    # Kayipli kaynak lehine kanitlar.
    reasons: tuple[str, ...]
    # Kayipsiz kaynak lehine / kayipli kanitini zayiflatan gozlemler.
    counter_reasons: tuple[str, ...]
    # Hukme girmeyen ama raporda gorunen bilgi.
    notes: tuple[str, ...]
    spectral: SpectralEvidence | None = None
    extra: dict[str, float | str] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        return {
            "consistent_lossless": "Consistent with a lossless source.",
            "consistent_lossy": "Consistent with a lossy source.",
            "undetermined": "Not enough evidence to judge.",
            "not_applicable": "The file is in a lossy format; nothing to verify.",
        }[self.bucket]


def judge_spectral(evidence: SpectralEvidence) -> tuple[list[str], list[str], list[str]]:
    """Spektral kanitlari uc listeye ayirir: (kanit, karsi-kanit, not)."""
    reasons: list[str] = []
    counter: list[str] = []
    notes: list[str] = []
    if evidence.frames < thresholds.MIN_ACTIVE_FRAMES:
        notes.append(f"only {evidence.frames} non-silent frames; too little to judge")
        return reasons, counter, notes

    nyquist = evidence.nyquist_hz
    cutoff = evidence.cutoff_median_hz
    low_cutoff = (
        cutoff < thresholds.MAX_LOSSY_CUTOFF_HZ
        and cutoff < thresholds.MAX_LOSSY_CUTOFF_NYQUIST_FRACTION * nyquist
    )
    if low_cutoff:
        text = f"content stops at {cutoff / 1000:.1f} kHz (Nyquist {nyquist / 1000:.1f} kHz)"
        for known_khz, label in thresholds.KNOWN_CUTOFFS_KHZ:
            if abs(cutoff / 1000 - known_khz) <= thresholds.CUTOFF_SNAP_KHZ:
                text += f"; typical of {label}"
                break
        reasons.append(text)
    else:
        counter.append(f"content extends to {cutoff / 1000:.1f} kHz, near Nyquist")

    drop = evidence.knee_drop_db
    wall = not math.isnan(drop) and drop > thresholds.BRICKWALL_DROP_DB
    if wall:
        reasons.append(
            f"brickwall at {evidence.knee_hz / 1000:.1f} kHz: {drop:.0f} dB drop within 500 Hz "
            f"(natural roll-off measured at 5-10 dB)"
        )
    elif not math.isnan(drop) and drop < thresholds.GENTLE_KNEE_DROP_DB and low_cutoff:
        counter.append(
            f"the roll-off at {evidence.knee_hz / 1000:.1f} kHz is gentle "
            f"({drop:.0f} dB per 500 Hz), as in a naturally dark recording"
        )

    floor = evidence.floor_rel_db
    if not math.isnan(floor):
        if floor < thresholds.EMPTY_FLOOR_REL_DB:
            reasons.append(
                f"nothing above the knee: {floor:.0f} dB below the 1-4 kHz level "
                f"(tape hiss or room noise would sit around -30..-45 dB)"
            )
        elif floor > thresholds.CONTENT_FLOOR_REL_DB and not wall:
            counter.append(
                f"content continues above the knee at {floor:.0f} dB relative: not a brickwall"
            )
        elif floor > thresholds.CONTENT_FLOOR_REL_DB:
            # Duvarin ustunde "icerik": bir doga kaydinda 500 Hz'de 25+ dB dusus
            # olmaz; oradaki seviye AAC'nin gurultu ikamesi (PNS) olabilir
            # (olculen: ffmpeg aac 128k taban -43 dB, duvar 18-49 dB).
            notes.append(
                f"level above the wall is {floor:.0f} dB relative; AAC noise substitution "
                "can leave synthetic noise there"
            )
    return reasons, counter, notes


def judge_container(info: Probe, flac_info: flac_bitstream.FlacInfo | None) -> list[str]:
    """Kap ve bit akisi bilgisi: hukme girmez, rapora girer."""
    notes: list[str] = []
    if flac_info is not None:
        family = flac_info.encoder_family
        if family:
            notes.append(f"FLAC encoder: {family} ({flac_info.vendor})")
        if not flac_info.stream_info.md5_present:
            notes.append("STREAMINFO carries no MD5: the encoder did not sign the PCM")
        ratio = flac_info.compression_ratio
        blocksize = flac_info.stream_info.max_blocksize
        if blocksize and blocksize < _UNUSUAL_BLOCKSIZE:
            # Olculen: ffmpeg'in flac kodlayicisi, bir cozucuden dogrudan
            # beslendiginde blok boyunu cozucunun paket boyundan aliyor (MP3 -> 47,
            # Vorbis -> 128) ve dosya neredeyse sikismiyor (0.97). Gercek bir rip
            # 47'lik blok uretmez. Hukme GIRMEZ: blok boyu kodlayiciyi belirlemez
            # (DECISIONS), ama bu kadar kucugu bir kodlama yolunun izidir.
            notes.append(
                f"unusual FLAC block size {blocksize}: consistent with ffmpeg encoding straight "
                "from a decoder with small packets (compression ratio is not meaningful)"
            )
        elif ratio is not None:
            notes.append(f"FLAC compression ratio {ratio:.2f}")
    return notes


def combine(reasons: list[str], counter: list[str], notes: list[str]) -> Bucket:
    if not reasons and not counter:
        return "undetermined"
    if reasons and not counter:
        return "consistent_lossy"
    if counter and not reasons:
        return "consistent_lossless"
    # Ikisi de var: tek basina dusuk kesim, karsi-kanitla "koyu kayit" olabilir.
    return "undetermined"


def verify(
    ffmpeg: Path,
    info: Probe,
    *,
    stream_index: int = 0,
    excerpt_s: float = DEFAULT_EXCERPT_S,
    cancel: CancelToken | None = None,
) -> Verdict:
    """Tek dosyayi dogrular. Dosya kayipli formattaysa `not_applicable`."""
    stream = info.stream(stream_index)
    if not stream.is_lossless:
        return Verdict("not_applicable", (), (), (f"codec {stream.codec} is lossy by design",))

    start: float | None = None
    duration: float | None = None
    if info.duration is not None and info.duration > excerpt_s / (1.0 - 2 * _SKIP_EDGE_FRACTION):
        start = max(_SKIP_EDGE_FRACTION * info.duration, info.duration / 2.0 - excerpt_s / 2.0)
        duration = excerpt_s
    evidence = spectral.analyse(
        ffmpeg,
        info.path,
        sample_rate=stream.sample_rate,
        channels=stream.channels,
        stream_index=stream_index,
        start=start,
        duration=duration,
        cancel=cancel,
    )
    reasons, counter, notes = judge_spectral(evidence)
    flac_info = flac_bitstream.scan(info.path) if stream.codec == "flac" else None
    notes += judge_container(info, flac_info)
    bucket = combine(reasons, counter, notes)
    extra: dict[str, float | str] = {}
    if flac_info is not None and flac_info.compression_ratio is not None:
        extra["compression_ratio"] = flac_info.compression_ratio
    return Verdict(bucket, tuple(reasons), tuple(counter), tuple(notes), evidence, extra)
