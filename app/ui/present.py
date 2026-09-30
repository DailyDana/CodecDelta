"""Sonuclari ekrana hazir satirlara cevirir. Qt IMPORT ETMEZ.

Arayuzdeki her sayi buradan gecer; boylece bicimlendirme kurallari (NaN'i
"--" yazmak, olculemez bandi gizlemek yerine isaretlemek, polariteyi kelimeyle
soylemek) ekransiz test edilir. Widget'lar yalnizca bu satirlari cizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from app.compare.ladder import LadderVerdict, Placement
from app.compare.result import BandResult, ComparisonResult
from app.single.verdict import Verdict
from app.ui.i18n import localize, tr

Tone = Literal["good", "warn", "bad", "neutral"]

DASH = "--"


@dataclass(frozen=True)
class Headline:
    title: str
    detail: str
    tone: Tone


@dataclass(frozen=True)
class BandRow:
    label: str
    mid: str
    side: str
    linear: str
    floor: str
    measurable: bool
    # Grafik icin ham degerler (NaN olabilir).
    centre_khz: float
    mid_db: float
    side_db: float
    floor_db: float


def db_text(value: float, *, digits: int = 1, unit: str = " dB") -> str:
    if value is None or math.isnan(value):
        return DASH
    if math.isinf(value):
        return ("> 150" if value > 0 else "< -150") + unit
    return f"{value:.{digits}f}{unit}"


def comparison_headline(result: ComparisonResult) -> Headline:
    plan = result.plan
    if result.status == "measured":
        snr = result.headline_snr_db
        if math.isnan(snr):
            title = tr("headline.floor")
            tone: Tone = "good"
        else:
            title = tr("headline.snr", snr=snr)
            tone = "neutral"
        detail = tr(f"verdict.{plan.verdict}")
        if plan.verdict == "different_master":
            tone = "warn"
        if result.excluded_s > 0:
            # Sonuc dosyanin yalnizca bir kismi icin: okuyucu bunu basliktan gormeli.
            tone = "warn"
            detail = tr("headline.partial", excluded=result.excluded_s)
        return Headline(title, detail, tone)
    key = "verdict.not_measured" if result.status == "not_measured" else f"verdict.{plan.verdict}"
    # Olculmediyse asil sebep boru hattinin notudur (orn. dosyalar surekli
    # degil); plan gerekcesi ancak o yoksa gosterilir.
    if result.status == "not_measured" and result.notes:
        detail = localize(result.notes[-1])
    else:
        detail = localize(plan.reasons[-1]) if plan.reasons else ""
        if not detail and result.notes:
            detail = localize(result.notes[-1])
    tone = "bad" if plan.verdict in ("different_recording", "unaligned") else "warn"
    return Headline(tr(key), detail, tone)


def _delay_text(result: ComparisonResult) -> str:
    samples = result.plan.delay_samples
    if math.isnan(samples):
        return DASH
    if abs(samples) < 0.5:
        # Yarim ornegin altini "0.000 ms onde" diye yazmak sahte bir yon verir.
        return tr("value.in_sync", samples=abs(samples))
    ms = abs(samples) / result.plan.sample_rate * 1000.0
    key = "value.test_later" if samples >= 0 else "value.test_earlier"
    return tr(key, ms=ms, samples=abs(samples))


def summary_rows(result: ComparisonResult) -> list[tuple[str, str]]:
    plan = result.plan
    rows = [
        (tr("summary.status"), tr(f"verdict.{plan.verdict}")),
        (tr("summary.delay"), _delay_text(result)),
    ]
    gain = result.gain_db if result.status == "measured" else plan.gain_db
    rows.append((tr("summary.gain"), DASH if math.isnan(gain) else tr("value.gain", db=gain)))
    polarity = result.polarity if result.status == "measured" else plan.polarity
    rows.append(
        (tr("summary.polarity"), tr("value.inverted") if polarity < 0 else tr("value.normal"))
    )
    if plan.channel_map:
        swapped = plan.channel_map[:2] == (1, 0)
        rows.append((tr("summary.channels"), tr("value.swapped") if swapped else tr("value.as_is")))
    drift = plan.drift
    if drift is not None and drift.status == "drift":
        label = localize(drift.label) if drift.label else f"{drift.ppm:+.1f} ppm"
        rows.append((tr("summary.drift"), label))
    elif drift is not None and drift.status == "none":
        rows.append((tr("summary.drift"), tr("value.no_drift")))
    if result.broadband is not None:
        rows.append((tr("summary.snr"), db_text(result.broadband.mid.snr_db)))
        rows.append((tr("summary.plain_snr"), db_text(result.broadband.mid.scalar_snr_db)))
    if result.nmr is not None and result.nmr.frames:
        rows.append(
            (tr("summary.nmr"), f"{db_text(result.nmr.p50_db)} / {db_text(result.nmr.p95_db)}")
        )
        rows.append((tr("summary.nmr_frames"), f"{100.0 * result.nmr.fraction_above_0:.1f} %"))
    if result.samples:
        rows.append(
            (
                tr("summary.duration"),
                tr("value.seconds", seconds=result.samples / result.analysis_rate)
                if result.excluded_s <= 0
                else tr(
                    "value.seconds_partial",
                    seconds=result.samples / result.analysis_rate,
                    excluded=result.excluded_s,
                ),
            )
        )
    return rows


def frequency_text(hz: float) -> str:
    """1 kHz altinda Hz, ustunde kHz: "20 Hz", "1 kHz", "21.8 kHz"."""
    if hz < 1000.0:
        return f"{hz:.0f} Hz"
    return f"{hz / 1000:.1f}".rstrip("0").rstrip(".") + " kHz"


def _band_label(lo: float, hi: float) -> str:
    """ "20 Hz-1 kHz", "1-4 kHz": birim ayniysa bir kez yazilir."""
    lo_text, hi_text = frequency_text(lo), frequency_text(hi)
    if lo_text.endswith(" kHz") and hi_text.endswith(" kHz"):
        return f"{lo_text[:-4]}–{hi_text}"
    return f"{lo_text}–{hi_text}"


def band_row(band: BandResult) -> BandRow:
    linear = math.nan
    if band.mid.incoherent_power > 0 and band.mid.linear_power > 0:
        linear = 10.0 * math.log10(band.mid.linear_power / band.mid.incoherent_power)
    measurable = band.measurable
    mid = (
        db_text(band.mid.snr_db)
        if measurable
        else f"{db_text(band.mid.snr_db)} ({tr('bands.unmeasurable')})"
    )
    return BandRow(
        label=_band_label(band.lo_hz, band.hi_hz),
        mid=mid,
        side=db_text(band.side.snr_db) if band.side is not None else DASH,
        linear=db_text(linear),
        floor=db_text(band.floor_db, digits=0),
        measurable=measurable,
        centre_khz=(band.lo_hz + band.hi_hz) / 2000.0,
        mid_db=band.mid.snr_db,
        side_db=band.side.snr_db if band.side is not None else math.nan,
        floor_db=band.floor_db,
    )


def band_rows(result: ComparisonResult) -> list[BandRow]:
    return [band_row(b) for b in result.bands]


def verdict_headline(verdict: Verdict) -> Headline:
    tones: dict[str, Tone] = {
        "consistent_lossless": "good",
        "consistent_lossy": "bad",
        "undetermined": "warn",
        "not_applicable": "neutral",
    }
    tone = tones[verdict.bucket]
    first = verdict.reasons or verdict.counter_reasons
    detail = localize(first[0]) if first else ""
    return Headline(tr(f"single.{verdict.bucket}"), detail, tone)


def verdict_rows(verdict: Verdict) -> list[tuple[str, str]]:
    e = verdict.spectral
    if e is None or e.frames == 0:
        return []
    return [
        (
            tr("single.cutoff"),
            f"{e.cutoff_median_hz / 1000:.1f} kHz / {e.nyquist_hz / 1000:.1f} kHz",
        ),
        (
            tr("single.knee"),
            f"{e.knee_hz / 1000:.1f} kHz, {db_text(e.knee_drop_db, digits=0)} / 500 Hz",
        ),
        (tr("single.floor"), db_text(e.floor_rel_db, digits=0)),
    ]


_CODEC_NAMES = {"libopus": "Opus", "libmp3lame": "MP3", "aac": "AAC", "libvorbis": "Vorbis"}


def _placement_text(placement: Placement, codec: str) -> str:
    if placement.position == "within" and placement.equivalent_kbps is not None:
        text = tr(
            "ladder.within",
            codec=codec,
            kbps=placement.equivalent_kbps,
            upper=placement.upper_kbps,
            lower=placement.lower_kbps,
        )
        if not placement.monotonic:
            text += " " + tr("ladder.approximate")
        return text
    if placement.position == "above":
        return tr("ladder.above", codec=codec, lower=placement.lower_kbps)
    if placement.position == "below":
        return tr("ladder.below", codec=codec, upper=placement.upper_kbps)
    return tr("ladder.unknown")


def _same_rung(a: Placement, b: Placement) -> bool:
    if a.position != b.position:
        return False
    if a.equivalent_kbps is None or b.equivalent_kbps is None:
        return True
    return abs(math.log2(a.equivalent_kbps / b.equivalent_kbps)) < 0.5


def ladder_text(verdict: LadderVerdict) -> str:
    """Merdiven hukmu gecerli dilde. Motorun Ingilizce `headline`i ile ayni mantik."""
    codec = _CODEC_NAMES.get(verdict.codec, verdict.codec)
    text = _placement_text(verdict.by_snr, codec)
    nmr = verdict.by_nmr
    if nmr.position != "unknown" and not _same_rung(verdict.by_snr, nmr):
        text += " " + tr("ladder.nmr_disagrees", text=_placement_text(nmr, codec))
    return text
