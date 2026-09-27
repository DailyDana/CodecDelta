"""Karsilastirma boru hatti -- uctan uca, gercek ffmpeg ile.

Her senaryo ffmpeg ile uretilmis dosyalarla, bilinen bir farkla kurulur.
Gercek bir FLAC/Opus ciftinde (9.5 dk) elle dogrulandi: gecikme -0.02 ornek,
hizali r 0.9983, duz S/N 24.33 dB -- planlama oturumundaki elle analizle
(lag 0, r ~0.998, S/N ~24 dB) ortusuyor.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest

from app.compare.pipeline import band_layout, compare, open_track
from app.compare.result import ComparisonResult, ComparisonSet
from app.core.ffmpeg_locate import FFmpegTools

SOURCE = "anoisesrc=color=pink:sample_rate=44100:duration=20:seed={seed},tremolo=f=1.1:d=0.85"


def _make(ffmpeg: Path, out: Path, *args: str, seed: int = 1, filters: str = "") -> Path:
    lavfi = SOURCE.format(seed=seed) + (f",{filters}" if filters else "")
    subprocess.run(
        [str(ffmpeg), "-y", "-v", "error", "-f", "lavfi", "-i", lavfi, "-ac", "2", *args, str(out)],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def files(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("pipeline")
    ff = ffmpeg_tools.ffmpeg
    reference = _make(ff, root / "ref.flac", "-c:a", "flac")
    out = {"ref": reference}
    copy = root / "copy.flac"
    shutil.copy(reference, copy)
    out["copy"] = copy
    for name, args in {
        "opus": ["-c:a", "libopus", "-b:a", "96k"],
        "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"],
    }.items():
        out[name] = root / f"test.{name}"
        subprocess.run(
            [str(ff), "-y", "-v", "error", "-i", str(reference), *args, str(out[name])], check=True
        )
    # 0.37 s sessizlikle geciktirilmis kopya (kayipsiz, gecikme tam bilinir)
    out["delayed"] = root / "delayed.flac"
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(reference),
            "-af",
            "adelay=370|370",
            "-c:a",
            "flac",
            str(out["delayed"]),
        ],
        check=True,
    )
    out["other"] = _make(ff, root / "other.flac", "-c:a", "flac", seed=99)
    # Saat kaymasi: asetrate hizi TAMSAYIYA yuvarlar, 44109/44100 = 204.08 ppm.
    out["drift"] = root / "drift.flac"
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(reference),
            "-af",
            "asetrate=44109,aformat=sample_fmts=dbl,"
            "aresample=44100:resampler=soxr:precision=28:cutoff=0.99",
            "-c:a",
            "flac",
            str(out["drift"]),
        ],
        check=True,
    )
    out["drift_opus"] = root / "drift.opus"
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(out["drift"]),
            "-c:a",
            "libopus",
            "-b:a",
            "96k",
            str(out["drift_opus"]),
        ],
        check=True,
    )
    out["pal"] = root / "pal.flac"
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(reference),
            "-af",
            "asetrate=44100*25/24,aresample=44100",
            "-c:a",
            "flac",
            str(out["pal"]),
        ],
        check=True,
    )
    return out


def _run(tools: FFmpegTools, ref: Path, test: Path) -> ComparisonResult:
    return compare(tools.ffmpeg, open_track(tools.ffprobe, ref), open_track(tools.ffprobe, test))


def test_band_layout_is_cut_at_the_smaller_nyquist() -> None:
    bands = band_layout(48000, 21830.0)
    assert bands[0] == (20.0, 1000.0)
    assert bands[-1] == (20000.0, 21830.0)
    assert all(hi <= 21830.0 for _, hi in bands)


@pytest.mark.needs_ffmpeg
def test_lossy_encode_is_measured_below_the_floor(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    result = _run(ffmpeg_tools, files["ref"], files["opus"])
    assert result.status == "measured", result.notes
    assert result.plan.verdict == "aligned"
    assert result.analysis_rate == 48000
    assert abs(result.plan.delay_samples) < 1.0
    assert result.broadband is not None and result.broadband.measurable
    assert 5.0 < result.headline_snr_db < 40.0
    # 44.1 kHz referans 48'e cikarildi: bantlar soxr kesiminde bitmeli
    assert result.bands[-1].hi_hz == pytest.approx(22050 * 0.99)
    assert all(b.floor_db > b.mid.snr_db for b in result.bands)


@pytest.mark.needs_ffmpeg
def test_known_delay_is_recovered_in_the_full_rate_timeline(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    """0.37 s = 16317 ornek @ 44.1 kHz; kayipsiz oldugu icin fark ~sifir olmali."""
    result = _run(ffmpeg_tools, files["ref"], files["delayed"])
    assert result.status == "measured", result.notes
    assert result.plan.delay_samples == pytest.approx(0.37 * 44100, abs=0.05)
    assert result.broadband is not None
    assert result.broadband.mid.snr_db > 100.0
    assert result.gain_db == pytest.approx(0.0, abs=0.01)


@pytest.mark.needs_ffmpeg
def test_mp3_encoder_delay_is_absorbed(ffmpeg_tools: FFmpegTools, files: dict[str, Path]) -> None:
    result = _run(ffmpeg_tools, files["ref"], files["mp3"])
    assert result.status == "measured", result.notes
    assert result.analysis_rate == 44100
    assert abs(result.plan.delay_samples) < 1.0
    assert 5.0 < result.headline_snr_db < 40.0


@pytest.mark.needs_ffmpeg
def test_identical_files_are_below_the_measurement_floor(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    """Birebir ayni dosya: fark zincirin tabaninin altinda, yani "olculemez".

    Bu dogru cevap ama henuz eksik: bit-ozdeslik ayrica ve daha ucuz yoldan
    (PCM ozeti) raporlanmali. Simdilik mansettaki sayi NaN.
    """
    result = _run(ffmpeg_tools, files["ref"], files["copy"])
    assert result.status == "measured"
    assert result.broadband is not None
    assert not result.broadband.measurable
    assert math.isnan(result.headline_snr_db)


@pytest.mark.needs_ffmpeg
def test_unrelated_files_are_not_measured(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    result = _run(ffmpeg_tools, files["ref"], files["other"])
    assert result.status == "not_comparable"
    assert result.plan.verdict == "different_recording"
    assert result.broadband is None


@pytest.mark.needs_ffmpeg
def test_pal_is_not_measured_as_a_codec_difference(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    result = _run(ffmpeg_tools, files["ref"], files["pal"])
    assert result.status == "not_comparable"
    assert result.plan.verdict == "speed_mismatch"
    assert math.isnan(result.headline_snr_db)


@pytest.mark.needs_ffmpeg
def test_result_set_is_plural_from_day_one(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    result = _run(ffmpeg_tools, files["ref"], files["mp3"])
    single = ComparisonSet.single(result)
    assert single.items == (result,)
    assert single.sweep_axis is None


@pytest.mark.needs_ffmpeg
def test_clock_drift_is_tracked_down_to_the_floor(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    """Kayipsiz, 204 ppm kaymali kopya. Olculen (90 s): izlemesiz -11.3 dB,
    cerceve basina hizalamayla 20.4, surekli warp + yinelemeli modelle 89.5 dB
    -- yani tabanda ("olculemez"), kayipsiz bir kopya icin dogru cevap.
    """
    result = _run(ffmpeg_tools, files["ref"], files["drift"])
    assert result.status == "measured", result.notes
    assert result.plan.drift is not None and result.plan.drift.needs_tracking
    assert any("clock drift tracked" in n for n in result.notes)
    assert result.broadband is not None
    assert result.broadband.mid.snr_db > 75.0


@pytest.mark.needs_ffmpeg
def test_drift_does_not_change_the_codec_measurement(
    ffmpeg_tools: FFmpegTools, files: dict[str, Path]
) -> None:
    """Ayni codec, ayni bitrate: kayma olsun ya da olmasin S/N ayni cikmali."""
    plain = _run(ffmpeg_tools, files["ref"], files["opus"])
    drifted = _run(ffmpeg_tools, files["ref"], files["drift_opus"])
    assert drifted.status == "measured", drifted.notes
    assert drifted.headline_snr_db == pytest.approx(plain.headline_snr_db, abs=1.0)
