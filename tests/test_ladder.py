"""Capa merdiveni ve kodlama isi testleri.

Kendi kendini dogrulayan sinav: referans 96 kbps Opus'a kodlanir, 64/96/128
merdiveninde konumlandirilir ve ~96 kbps'e oturmasi beklenir. Gercek bir
FLAC ile YouTube Opus ciftinde (9.5 dk, 5 basamak) olculen: S/N 19.6 -> 30.9 dB
ve NMR p95 13.8 -> -0.2 dB, ikisi de monoton; test ~128 (S/N) / ~134 (NMR).
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from app.compare import ladder
from app.compare.ladder import Placement, place
from app.compare.pipeline import compare, open_track
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_stream import ResampleCfg
from app.encode.jobs import EncodeJob, build_args

# -- kodlama isi -------------------------------------------------------------


def test_encode_args_never_touch_video() -> None:
    args = build_args(EncodeJob(Path("in.mkv"), Path("out.opus"), "libopus", 128, stream_index=2))
    assert "-vn" in args and "-sn" in args and "-dn" in args
    assert args[args.index("-map") + 1] == "0:a:2"
    assert args[args.index("-b:a") + 1] == "128k"
    assert args[-1] == "out.opus"


def test_opus_from_44k1_resamples_with_the_analysis_chain() -> None:
    """44.1 kHz -> Opus: ffmpeg'in sessiz varsayilan resampler'i degil, soxr + cutoff."""
    cfg = ResampleCfg(cutoff=0.97)
    args = build_args(
        EncodeJob(Path("a.flac"), Path("a.opus"), "libopus", 96, source_rate=44100), resample=cfg
    )
    chain = args[args.index("-af") + 1]
    assert "aresample=48000" in chain and "cutoff=0.97" in chain and "aformat" in chain
    # 48 kHz kaynak ve Opus disi kodlayicilar dokunulmaz
    assert "-af" not in build_args(
        EncodeJob(Path("a.flac"), Path("a.opus"), "libopus", 96, source_rate=48000)
    )
    assert "-af" not in build_args(
        EncodeJob(Path("a.flac"), Path("a.mp3"), "libmp3lame", 96, source_rate=44100)
    )


def test_lossless_job_has_no_bitrate() -> None:
    args = build_args(
        EncodeJob(Path("a.wav"), Path("a.flac"), "flac", extra=("-compression_level", "8"))
    )
    assert "-b:a" not in args
    assert args[args.index("-compression_level") + 1] == "8"


# -- konumlandirma (saf) -----------------------------------------------------

RATES = [64, 96, 128, 192]
SNR = [19.6, 23.2, 25.7, 28.7]  # gercek olcumden


def test_place_interpolates_on_log_bitrate() -> None:
    p = place(SNR, RATES, 24.4)
    assert p.position == "within" and p.monotonic
    assert p.lower_kbps == 96 and p.upper_kbps == 128
    assert p.equivalent_kbps is not None and 96 < p.equivalent_kbps < 128
    # tam basamak degeri -> tam bitrate
    assert place(SNR, RATES, 23.2).equivalent_kbps == pytest.approx(96.0, abs=0.5)


def test_place_reports_outside_the_ladder() -> None:
    above = place(SNR, RATES, 35.0)
    assert above.position == "above" and above.lower_kbps == 192 and above.equivalent_kbps is None
    below = place(SNR, RATES, 10.0)
    assert below.position == "below" and below.upper_kbps == 64


def test_place_flags_a_non_monotonic_ladder() -> None:
    p = place([19.6, 25.0, 23.0, 28.7], RATES, 24.0)
    assert not p.monotonic
    assert p.position == "within"


def test_place_is_unknown_when_the_value_is_not_measurable() -> None:
    assert place(SNR, RATES, math.nan).position == "unknown"
    assert place([19.6, math.nan, 25.7, 28.7], RATES, 24.0).position == "unknown"


def test_headline_mentions_disagreement() -> None:
    rungs = ()
    agree = ladder.LadderVerdict(
        "libopus",
        rungs,
        Placement("within", 110.0, 96, 128, True),
        Placement("within", 120.0, 96, 128, True),
        (),
    )
    assert "roughly 110 kbps" in agree.headline and "NMR" not in agree.headline
    disagree = ladder.LadderVerdict(
        "libopus",
        rungs,
        Placement("within", 110.0, 96, 128, True),
        Placement("above", None, 192, None, True),
        (),
    )
    assert "NMR" in disagree.headline and "192" in disagree.headline


# -- uctan uca ---------------------------------------------------------------


@pytest.fixture(scope="module")
def reference(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ladder")
    path = root / "ref.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=20:seed=11,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.needs_ffmpeg
def test_ladder_places_a_known_bitrate_on_itself(
    ffmpeg_tools: FFmpegTools, reference: Path, tmp_path: Path
) -> None:
    """96 kbps Opus, 64/96/128 merdiveninde ~96'ya oturmali; eksenler monoton olmali."""
    track = open_track(ffmpeg_tools.ffprobe, reference)
    rungs = ladder.build(
        ffmpeg_tools.ffmpeg, ffmpeg_tools.ffprobe, track, tmp_path, rungs=(64, 96, 128)
    )
    assert [r.bitrate_kbps for r in rungs] == [64, 96, 128]
    assert not list(tmp_path.glob("ladder_*")), "gecici kodlamalar silinmeli"
    snr = [r.snr_db for r in rungs]
    assert snr == sorted(snr)

    test_path = tmp_path / "test96.opus"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-i",
            str(reference),
            "-af",
            "aformat=sample_fmts=dbl,aresample=48000:resampler=soxr:precision=28:cutoff=0.99",
            "-c:a",
            "libopus",
            "-b:a",
            "96k",
            str(test_path),
        ],
        check=True,
    )
    result = compare(ffmpeg_tools.ffmpeg, track, open_track(ffmpeg_tools.ffprobe, test_path))
    verdict = ladder.judge(rungs, result)
    assert verdict.by_snr.position == "within"
    assert verdict.by_snr.equivalent_kbps == pytest.approx(96.0, rel=0.15)
    assert verdict.by_nmr.position in ("within", "above", "below")
    assert "kbps" in verdict.headline
