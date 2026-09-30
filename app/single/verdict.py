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
from app.core.messages import Message
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
        notes.append(
            Message(
                "single.few_frames",
                "only {frames} non-silent frames; too little to judge",
                frames=evidence.frames,
            )
        )
        return reasons, counter, notes

    nyquist = evidence.nyquist_hz
    cutoff = evidence.cutoff_median_hz
    low_cutoff = (
        cutoff < thresholds.MAX_LOSSY_CUTOFF_HZ
        and cutoff < thresholds.MAX_LOSSY_CUTOFF_NYQUIST_FRACTION * nyquist
    )
    drop = evidence.knee_drop_db
    knee_hz = evidence.knee_hz
    wall = not math.isnan(drop) and drop > thresholds.BRICKWALL_DROP_DB
    # Nyquist'in hemen altindaki duvar kaydin/ornekleme hizi donusumunun
    # anti-alias filtresidir, codec degil. Olculen (D:/music): 24/48 remaster
    # 23.7 kHz'de 60-80 dB, gercek CD 21.1-21.2 kHz'de 17-24 dB. Codec dizleri
    # en fazla 20.9 kHz (Vorbis, 44.1'de 0.948 Nyquist).
    wall_near_nyquist = wall and knee_hz >= thresholds.MAX_LOSSY_CUTOFF_NYQUIST_FRACTION * nyquist

    if not low_cutoff:
        # Ana kapi: icerik Nyquist'e kadar. Olculen hicbir seffaf-olmayan codec
        # buraya ulasmadi; duvar ve taban burada yalnizca not.
        counter.append(
            Message(
                "single.full_band",
                "content extends to {khz:.1f} kHz, near Nyquist",
                khz=cutoff / 1000,
            )
        )
        if wall:
            notes.append(
                Message(
                    "single.antialias_note",
                    "steep filter at {khz:.1f} kHz ({drop:.0f} dB within 500 Hz, {pct:.0f}% of "
                    "Nyquist): typical of the recording's anti-alias or sample-rate "
                    "conversion filter",
                    khz=knee_hz / 1000,
                    drop=drop,
                    pct=100 * knee_hz / nyquist,
                )
            )
        return reasons, counter, notes

    typical = next(
        (
            label
            for known_khz, label in thresholds.KNOWN_CUTOFFS_KHZ
            if abs(cutoff / 1000 - known_khz) <= thresholds.CUTOFF_SNAP_KHZ
        ),
        None,
    )
    if typical is None:
        reasons.append(
            Message(
                "single.cutoff",
                "content stops at {khz:.1f} kHz (Nyquist {nyq:.1f} kHz)",
                khz=cutoff / 1000,
                nyq=nyquist / 1000,
            )
        )
    else:
        reasons.append(
            Message(
                "single.cutoff_typical",
                "content stops at {khz:.1f} kHz (Nyquist {nyq:.1f} kHz); typical of {label}",
                khz=cutoff / 1000,
                nyq=nyquist / 1000,
                label=typical,
            )
        )

    if wall and not wall_near_nyquist:
        reasons.append(
            Message(
                "single.brickwall",
                "brickwall at {khz:.1f} kHz: {drop:.0f} dB drop within 500 Hz "
                "(natural roll-off measured at 4-10 dB)",
                khz=knee_hz / 1000,
                drop=drop,
            )
        )
    elif wall_near_nyquist:
        notes.append(
            Message(
                "single.antialias_low",
                "steep filter at {khz:.1f} kHz is at {pct:.0f}% of Nyquist: an anti-alias "
                "filter, not evidence",
                khz=knee_hz / 1000,
                pct=100 * knee_hz / nyquist,
            )
        )
    elif not math.isnan(drop) and drop < thresholds.GENTLE_KNEE_DROP_DB:
        counter.append(
            Message(
                "single.gentle",
                "the roll-off at {khz:.1f} kHz is gentle ({drop:.0f} dB per 500 Hz), "
                "as in a naturally dark recording",
                khz=knee_hz / 1000,
                drop=drop,
            )
        )

    codec_wall = wall and not wall_near_nyquist
    floor = evidence.floor_rel_db
    if not math.isnan(floor):
        if floor < thresholds.EMPTY_FLOOR_REL_DB:
            reasons.append(
                Message(
                    "single.empty_floor",
                    "nothing above the knee: {floor:.0f} dB below the 1-4 kHz level "
                    "(tape hiss or room noise would sit around -30..-45 dB)",
                    floor=floor,
                )
            )
        elif floor > thresholds.CONTENT_FLOOR_REL_DB and not codec_wall:
            counter.append(
                Message(
                    "single.content_floor",
                    "content continues above the knee at {floor:.0f} dB relative: not a brickwall",
                    floor=floor,
                )
            )
        elif floor > thresholds.CONTENT_FLOOR_REL_DB:
            # Duvarin ustunde "icerik": bir doga kaydinda Nyquist'ten uzak
            # 500 Hz'de 18+ dB dusus olmaz; oradaki seviye AAC'nin gurultu
            # ikamesi (PNS) olabilir (olculen: ffmpeg aac 128k taban -43 dB).
            notes.append(
                Message(
                    "single.pns",
                    "level above the wall is {floor:.0f} dB relative; AAC noise substitution "
                    "can leave synthetic noise there",
                    floor=floor,
                )
            )
    return reasons, counter, notes


def judge_container(info: Probe, flac_info: flac_bitstream.FlacInfo | None) -> list[str]:
    """Kap ve bit akisi bilgisi: hukme girmez, rapora girer."""
    notes: list[str] = []
    if flac_info is not None:
        family = flac_info.encoder_family
        if flac_info.vendor_rewritten:
            notes.append(
                Message(
                    "single.vendor_rewritten",
                    "FLAC vendor string was rewritten by a tagging library ({vendor}); "
                    "the encoder is unknown",
                    vendor=flac_info.vendor,
                )
            )
        elif family:
            notes.append(
                Message(
                    "single.encoder",
                    "FLAC encoder: {family} ({vendor})",
                    family=family,
                    vendor=flac_info.vendor,
                )
            )
        if not flac_info.stream_info.md5_present:
            notes.append(
                Message(
                    "single.no_md5", "STREAMINFO carries no MD5: the encoder did not sign the PCM"
                )
            )
        ratio = flac_info.compression_ratio
        blocksize = flac_info.stream_info.max_blocksize
        if blocksize and blocksize < _UNUSUAL_BLOCKSIZE:
            # Olculen: ffmpeg'in flac kodlayicisi, bir cozucuden dogrudan
            # beslendiginde blok boyunu cozucunun paket boyundan aliyor (MP3 -> 47,
            # Vorbis -> 128) ve dosya neredeyse sikismiyor (0.97). Gercek bir rip
            # 47'lik blok uretmez. Hukme GIRMEZ: blok boyu kodlayiciyi belirlemez
            # (DECISIONS), ama bu kadar kucugu bir kodlama yolunun izidir.
            notes.append(
                Message(
                    "single.blocksize",
                    "unusual FLAC block size {blocksize}: consistent with ffmpeg encoding straight "
                    "from a decoder with small packets (compression ratio is not meaningful)",
                    blocksize=blocksize,
                )
            )
        elif ratio is not None:
            notes.append(Message("single.ratio", "FLAC compression ratio {ratio:.2f}", ratio=ratio))
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


def _rejudge_rate(evidence: SpectralEvidence) -> int | None:
    """Yuksek hizli dosyanin yargilanacagi standart hiz, ya da None.

    Duvar bir standart hizin Nyquist'indeyse dosya o hizdan buyutulmustur ve
    o hizda yargilanir. Degilse ve icerik 48 kHz'e sigiyorsa 48 kHz: 44.1'e
    indirmek 48 kHz kaynakli uc kayipli dosyayi "belirsiz"e cekti (olculen,
    D:/music 231 dosya).
    """
    # Esikler 44.1 ve 48 kHz'te olculdu; bu hizlarda dosya oldugu gibi yargilanir.
    widest = max(thresholds.REJUDGE_RATES)
    if evidence.frames < thresholds.MIN_ACTIVE_FRAMES or evidence.sample_rate <= widest:
        return None
    source = _upsampled_from(evidence)
    if source is not None:
        return source
    if evidence.cutoff_median_hz < thresholds.REJUDGE_NYQUIST_FRACTION * widest / 2:
        return widest
    return None


def _upsampled_from(evidence: SpectralEvidence) -> int | None:
    """Diz bir standart hizin Nyquist'indeki dik duvarsa o hiz."""
    drop = evidence.knee_drop_db
    if math.isnan(drop) or drop <= thresholds.BRICKWALL_DROP_DB:
        return None
    for rate in thresholds.REJUDGE_RATES:
        if rate < evidence.sample_rate and (
            abs(evidence.knee_hz / (rate / 2) - 1.0) <= thresholds.UPSAMPLED_KNEE_TOLERANCE
        ):
            return rate
    return None


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

    def analyse(rate: int | None = None) -> SpectralEvidence:
        return spectral.analyse(
            ffmpeg,
            info.path,
            sample_rate=stream.sample_rate,
            channels=stream.channels,
            stream_index=stream_index,
            start=start,
            duration=duration,
            rate=rate,
            cancel=cancel,
        )

    evidence = analyse()
    rate_notes: list[str] = []
    rejudge = _rejudge_rate(evidence)
    if rejudge is not None:
        source = _upsampled_from(evidence)
        if source is not None:
            rate_notes.append(
                Message(
                    "single.upsampled",
                    "steep wall at {khz:.1f} kHz, the Nyquist of {rate:g} kHz: the file appears "
                    "upsampled from {rate:g} kHz",
                    khz=evidence.knee_hz / 1000,
                    rate=source / 1000,
                )
            )
        rate_notes.append(
            Message(
                "single.rejudged",
                "no content above {khz:.1f} kHz in a {file:g} kHz file: judged at {rate:g} kHz",
                khz=evidence.cutoff_median_hz / 1000,
                file=stream.sample_rate / 1000,
                rate=rejudge / 1000,
            )
        )
        evidence = analyse(rejudge)
    reasons, counter, notes = judge_spectral(evidence)
    notes = rate_notes + notes
    flac_info = flac_bitstream.scan(info.path) if stream.codec == "flac" else None
    notes += judge_container(info, flac_info)
    bucket = combine(reasons, counter, notes)
    extra: dict[str, float | str] = {}
    if flac_info is not None and flac_info.compression_ratio is not None:
        extra["compression_ratio"] = flac_info.compression_ratio
    return Verdict(bucket, tuple(reasons), tuple(counter), tuple(notes), evidence, extra)
