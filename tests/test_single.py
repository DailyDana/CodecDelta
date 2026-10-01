"""Referanssiz dogrulama testleri.

Sentetik sinyaller uc durumu kurar: icerik Nyquist'e kadar (kayipsizla
tutarli), duvar gibi kesim + bos taban (kayipli), yumusak dogal roll-off +
hisir (koyu kayit -> belirsiz, karsi-kanitla). Esikler etiketli gercek setten
geldi (thresholds.py); burada sinanan sey kanitlarin dogru kovaya gitmesi.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.core.ffmpeg_locate import FFmpegTools
from app.core.probe import probe
from app.single import spectral, thresholds, verdict
from app.single.spectral import SpectralEvidence

RATE = 44100
SECONDS = 20


def pink(seed: int, seconds: int = SECONDS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = seconds * RATE
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0.0
    x = np.fft.irfft(spectrum, n)
    return 0.3 * x / np.max(np.abs(x))


def shape(x: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """Frekans alaninda kazanc egrisi uygular (`gain_db` rfft binleri boyunca)."""
    spectrum = np.fft.rfft(x)
    return np.fft.irfft(spectrum * 10 ** (gain_db / 20), x.size)


def freqs_of(x: np.ndarray) -> np.ndarray:
    return np.fft.rfftfreq(x.size, 1 / RATE)


def brickwall(x: np.ndarray, cutoff_hz: float, floor_db: float = -110.0) -> np.ndarray:
    f = freqs_of(x)
    gain = np.where(f < cutoff_hz, 0.0, floor_db)
    return shape(x, gain)


def dark_natural(x: np.ndarray, corner_hz: float = 5000.0, hiss_db: float = -55.0) -> np.ndarray:
    """Dik ama duvarsiz dogal roll-off (30 dB/okt) + genis bantli hisir: eski, koyu kayit.

    6 dB/okt yetmiyor: pembe gurultuyle Nyquist'te ancak -15 dB'e iner ve
    -55 dB'lik kesim olcutune hic ulasmaz. 78'lik ya da erken bant kayitlari
    8 kHz ustunde 30-40 dB kaybeder; 500 Hz'lik pencerede dusus yine 1-2 dB.
    """
    f = freqs_of(x)
    gain = np.where(f > corner_hz, -100 * np.log10(np.maximum(f, 1) / corner_hz), 0.0)
    rng = np.random.default_rng(99)
    hiss = rng.standard_normal(x.size)
    hiss *= np.sqrt(np.mean(x**2) / np.mean(hiss**2)) * 10 ** (hiss_db / 20)
    return shape(x, gain) + hiss


def analyse(x: np.ndarray) -> SpectralEvidence:
    return spectral.analyse_samples(x, x * 0.3, RATE)


# -- spektral ozellikler ------------------------------------------------------


def test_full_band_content_reaches_nyquist() -> None:
    e = analyse(pink(1))
    assert e.frames > thresholds.MIN_ACTIVE_FRAMES
    assert e.cutoff_median_hz > 0.95 * e.nyquist_hz
    assert e.knee_drop_db < thresholds.GENTLE_KNEE_DROP_DB


def test_brickwall_is_found_where_it_was_put() -> None:
    e = analyse(brickwall(pink(2), 16000.0))
    assert e.knee_hz == pytest.approx(16000.0, abs=400.0)
    assert e.cutoff_median_hz == pytest.approx(16000.0, abs=300.0)
    assert e.knee_drop_db > thresholds.BRICKWALL_DROP_DB
    assert e.floor_rel_db < thresholds.EMPTY_FLOOR_REL_DB


def test_dark_denoised_recording_has_a_gentle_knee() -> None:
    """Koyu ve sessiz tabanli kayit: kesim dusuk ama diz yumusak (karsi-kanit)."""
    e = analyse(dark_natural(pink(3), hiss_db=-75.0))
    assert e.cutoff_median_hz < thresholds.MAX_LOSSY_CUTOFF_HZ
    assert e.knee_drop_db < thresholds.GENTLE_KNEE_DROP_DB


def test_tape_hiss_counts_as_content_to_nyquist() -> None:
    """Bant hisiri (-45 dB) -55 dB olcutunun ustunde: koyu kayit kayipli sayilmaz."""
    e = analyse(dark_natural(pink(3), hiss_db=-45.0))
    assert e.cutoff_median_hz > 0.95 * e.nyquist_hz


def test_silence_yields_no_frames() -> None:
    e = spectral.analyse_samples(np.zeros(RATE * 5), None, RATE)
    assert e.frames == 0
    assert math.isnan(e.cutoff_median_hz)


def test_mono_has_no_side_measure() -> None:
    e = spectral.analyse_samples(pink(4), None, RATE)
    assert math.isnan(e.side_hf_rel_db)
    assert e.frames > 0


def test_smoothing_does_not_pad_the_edge_with_zero_db() -> None:
    """Kenar sifir dB ile doldurulursa her cercevede kesim Nyquist cikar (olculdu)."""
    rows = np.full((1, 100), -90.0)
    smoothed = spectral._smooth(rows, 5)
    assert smoothed.shape == (1, 96)
    assert np.all(smoothed == pytest.approx(-90.0))


# -- hukum ---------------------------------------------------------------------


def _evidence(**overrides: float) -> SpectralEvidence:
    base = {
        "sample_rate": RATE,
        "frames": 200,
        "reference_db": -50.0,
        "knee_hz": 16000.0,
        "knee_drop_db": 50.0,
        "floor_rel_db": -95.0,
        "cutoff_median_hz": 16000.0,
        "cutoff_iqr_hz": 100.0,
        "side_hf_rel_db": -3.0,
    }
    base.update(overrides)
    return SpectralEvidence(ltas_rel_db=np.zeros(0), freqs_hz=np.zeros(0), **base)  # type: ignore[arg-type]


def test_mp3_signature_is_lossy_with_reasons() -> None:
    reasons, counter, _ = verdict.judge_spectral(_evidence())
    assert verdict.combine(reasons, counter, []) == "consistent_lossy"
    assert any("16.0 kHz" in r and "128 kbps" in r for r in reasons)
    assert any("brickwall" in r for r in reasons)
    assert any("nothing above the knee" in r for r in reasons)
    assert not counter


def test_full_band_is_lossless_consistent() -> None:
    e = _evidence(cutoff_median_hz=21400.0, knee_hz=19500.0, knee_drop_db=6.0, floor_rel_db=-30.0)
    reasons, counter, _ = verdict.judge_spectral(e)
    assert not reasons
    assert verdict.combine(reasons, counter, []) == "consistent_lossless"


def test_dark_recording_is_undetermined_not_lossy() -> None:
    """Dusuk kesim ANCAK yumusak diz ve diz ustu icerik -> celisen kanit."""
    e = _evidence(cutoff_median_hz=14000.0, knee_hz=12000.0, knee_drop_db=6.0, floor_rel_db=-35.0)
    reasons, counter, _ = verdict.judge_spectral(e)
    assert reasons and counter
    assert verdict.combine(reasons, counter, []) == "undetermined"
    assert any("gentle" in c for c in counter)
    assert any("continues above the knee" in c for c in counter)


def test_low_sample_rate_file_is_not_called_lossy_for_its_own_nyquist() -> None:
    """32 kHz'lik dosyada 15.3 kHz kesim: Nyquist'in %96'si, kayip degil."""
    e = _evidence(sample_rate=32000, cutoff_median_hz=15300.0, knee_hz=15500.0)
    reasons, _, _ = verdict.judge_spectral(e)
    assert not any("stops at" in r for r in reasons)


def test_too_few_frames_is_undetermined() -> None:
    reasons, counter, notes = verdict.judge_spectral(_evidence(frames=10))
    assert not reasons and not counter
    assert notes and "too little" in notes[0]
    assert verdict.combine(reasons, counter, notes) == "undetermined"


def test_aac_pns_floor_is_not_counted_as_content_when_knee_is_a_wall() -> None:
    """AAC-PNS: taban -43 dB (icerik gibi) ama kesim 20.0 ve duvar 40 dB -> yine kayipli."""
    e = _evidence(cutoff_median_hz=19900.0, knee_hz=20000.0, knee_drop_db=40.0, floor_rel_db=-43.0)
    reasons, counter, _ = verdict.judge_spectral(e)
    assert verdict.combine(reasons, counter, []) == "consistent_lossy"


def test_headline_never_claims_proof() -> None:
    for bucket in ("consistent_lossless", "consistent_lossy", "undetermined", "not_applicable"):
        text = verdict.Verdict(bucket, (), (), ()).headline.lower()  # type: ignore[arg-type]
        assert "proven" not in text and "is a transcode" not in text


# -- uctan uca -----------------------------------------------------------------


@pytest.mark.needs_ffmpeg
def test_end_to_end_transcode_and_clean_file(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    ff = str(ffmpeg_tools.ffmpeg)
    clean = tmp_path / "clean.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=20:seed=5,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(clean),
        ],
        check=True,
    )
    lossy = tmp_path / "t.mp3"
    subprocess.run(
        [ff, "-v", "error", "-i", str(clean), "-c:a", "libmp3lame", "-b:a", "128k", str(lossy)],
        check=True,
    )
    transcode = tmp_path / "transcode.flac"
    subprocess.run(
        [ff, "-v", "error", "-i", str(lossy), "-c:a", "flac", str(transcode)], check=True
    )

    clean_verdict = verdict.verify(ffmpeg_tools.ffmpeg, probe(ffmpeg_tools.ffprobe, clean))
    assert clean_verdict.bucket == "consistent_lossless", clean_verdict
    assert any("FLAC encoder" in n for n in clean_verdict.notes)
    assert "compression_ratio" in clean_verdict.extra

    lossy_verdict = verdict.verify(ffmpeg_tools.ffmpeg, probe(ffmpeg_tools.ffprobe, transcode))
    assert lossy_verdict.bucket == "consistent_lossy", lossy_verdict
    assert any("stops at 16." in r for r in lossy_verdict.reasons)
    assert any("brickwall" in r for r in lossy_verdict.reasons)

    mp3_verdict = verdict.verify(ffmpeg_tools.ffmpeg, probe(ffmpeg_tools.ffprobe, lossy))
    assert mp3_verdict.bucket == "not_applicable"


def _cd_like(seconds: int = SECONDS) -> np.ndarray:
    """Gercek CD'ye benzer: 12 kHz'ten sonra yumusak dogal inis (~2.7 dB/500 Hz).

    Kare basina kesim bu yuzden 20.5 kHz civarinda kalir; gercek CD'lerde
    olculen 20.87-21.2 kHz'e yakin.
    """
    x = pink(7, seconds)
    f = freqs_of(x)
    return shape(x, np.where(f > 12_000, -5.4 * (f - 12_000) / 1000, 0.0))


def _flac(ffmpeg: Path, x: np.ndarray, out: Path, rate: int | None = None) -> Path:
    resample = (
        ["-af", f"aformat=sample_fmts=dbl,aresample={rate}:resampler=soxr:precision=28:cutoff=0.99"]
        if rate
        else []
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            "-i",
            "-",
            *resample,
            "-sample_fmt",
            "s32",
            "-c:a",
            "flac",
            str(out),
        ],
        input=x.astype("<f4").tobytes(),
        check=True,
    )
    return out


@pytest.mark.needs_ffmpeg
def test_an_upsampled_cd_is_judged_at_its_source_rate(
    ffmpeg_tools: FFmpegTools, tmp_path: Path
) -> None:
    """44.1'den 96'ya buyutulmus kayit: resampler duvari codec duvari degil (D3).

    Once 22.05 kHz'teki duvar "brickwall" ve ustundeki bosluk "bos taban"
    sayiliyordu; dosya dogal hizda ne diyorsa buyutulmus hali de onu demeli.
    """
    ff = ffmpeg_tools.ffmpeg
    x = _cd_like()
    native = _flac(ff, x, tmp_path / "native.flac")
    upsampled = _flac(ff, x, tmp_path / "up96.flac", rate=96_000)
    lossy = tmp_path / "lossy.mp3"
    subprocess.run(
        [
            str(ff),
            "-v",
            "error",
            "-i",
            str(native),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(lossy),
        ],
        check=True,
    )
    lossy_up = tmp_path / "lossy96.flac"
    subprocess.run(
        [
            str(ff),
            "-v",
            "error",
            "-i",
            str(lossy),
            "-af",
            "aformat=sample_fmts=dbl,aresample=96000:resampler=soxr:precision=28:cutoff=0.99",
            "-sample_fmt",
            "s32",
            "-c:a",
            "flac",
            str(lossy_up),
        ],
        check=True,
    )

    def judge(path: Path) -> verdict.Verdict:
        return verdict.verify(ff, probe(ffmpeg_tools.ffprobe, path))

    base, up = judge(native), judge(upsampled)
    assert base.bucket != "consistent_lossy", base
    assert up.bucket == base.bucket, up
    keys = [getattr(n, "key", "") for n in up.notes]
    assert "single.upsampled" in keys and "single.rejudged" in keys
    assert judge(lossy_up).bucket == "consistent_lossy"


def test_streamed_evidence_matches_one_block() -> None:
    """Akisli biriktirici, sinyal parca parca verilince ayni kaniti uretir (D5)."""
    x = brickwall(pink(3), 16_000)
    whole = spectral.analyse_samples(x, x * 0.3, RATE)
    accumulator = spectral._Accumulator(RATE, top_hz=None, stereo=True)
    for start in range(0, x.size, 10_007):
        part = x[start : start + 10_007]
        accumulator.push(part, part * 0.3)
    parts = accumulator.finish()
    assert parts.frames == whole.frames
    for field in ("cutoff_median_hz", "knee_hz", "knee_drop_db", "floor_rel_db", "side_hf_rel_db"):
        assert getattr(parts, field) == pytest.approx(getattr(whole, field), abs=1e-9), field


@pytest.mark.needs_ffmpeg
def test_verification_memory_does_not_grow_with_the_rate(
    ffmpeg_tools: FFmpegTools, tmp_path: Path
) -> None:
    """192 kHz'lik kesit 705 MB tutuyordu; akisli analizle sabit kalmali (D5)."""
    import tracemalloc

    path = tmp_path / "hires.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=192000:duration=40:seed=3",
            "-ac",
            "2",
            "-c:a",
            "flac",
            "-sample_fmt",
            "s32",
            str(path),
        ],
        check=True,
    )
    info = probe(ffmpeg_tools.ffprobe, path)
    tracemalloc.start()
    try:
        verdict.verify(ffmpeg_tools.ffmpeg, info)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64 * 2**20, peak


@pytest.mark.needs_ffmpeg
def test_a_file_without_a_length_is_judged_on_its_start(
    ffmpeg_tools: FFmpegTools, tmp_path: Path
) -> None:
    """Boruya yazilmis FLAC'ta uzunluk yok; tum dosya bellege aliniyordu (D5)."""
    path = tmp_path / "piped.flac"
    with path.open("wb") as out:
        subprocess.run(
            [
                str(ffmpeg_tools.ffmpeg),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=pink:sample_rate=44100:duration=20:seed=3",
                "-ac",
                "2",
                "-c:a",
                "flac",
                "-f",
                "flac",
                "-",
            ],
            stdout=out,
            check=True,
        )
    info = probe(ffmpeg_tools.ffprobe, path)
    assert info.duration is None
    result = verdict.verify(ffmpeg_tools.ffmpeg, info)
    assert any(getattr(n, "key", "") == "single.unknown_duration" for n in result.notes)
    assert result.bucket == "consistent_lossless"


@pytest.mark.needs_ffmpeg
def test_a_silent_middle_moves_the_excerpt(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Orta kesit sessizken hukum "belirsiz" cikiyordu (D13)."""
    from tests.conftest import silent_middle

    path = silent_middle(ffmpeg_tools.ffmpeg, tmp_path / "gap.flac")
    result = verdict.verify(ffmpeg_tools.ffmpeg, probe(ffmpeg_tools.ffprobe, path))
    assert result.bucket == "consistent_lossless", result
    assert any(getattr(n, "key", "") == "single.moved_excerpt" for n in result.notes)


@pytest.mark.parametrize("filter_hz", [21_000.0, 21_500.0])
def test_an_anti_alias_filter_does_not_hide_a_dark_recording(filter_hz: float) -> None:
    """Karanlik kayit + dik anti-alias filtresi "kayipli" cikiyordu (D11).

    Diz filtreye oturuyor ve yumusak dogal inis (karsi-kanit) gorunmuyordu.
    Filtreli hukum filtresizle ayni olmali; filtre yalnizca not.
    """
    x = pink(5)
    f = freqs_of(x)
    dark = shape(x, np.where(f > 9000, -5.4 * (f - 9000) / 1000, 0.0))

    def bucket(signal: np.ndarray) -> str:
        reasons, counter, notes = verdict.judge_spectral(analyse(signal))
        return verdict.combine(reasons, counter, notes)

    filtered = brickwall(dark, filter_hz, floor_db=-110)
    assert bucket(filtered) == bucket(dark) != "consistent_lossy"
    evidence = analyse(filtered)
    assert evidence.antialias_hz == pytest.approx(filter_hz, abs=400)
    assert evidence.knee_hz < filter_hz - 500


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize("kbps", [32, 48])
def test_a_low_bitrate_mp3_wall_below_8_khz_is_found(
    ffmpeg_tools: FFmpegTools, tmp_path: Path, kbps: int
) -> None:
    """Diz aramasi 8 kHz'ten basliyordu; 4-8 kHz duvari gorulmuyordu (D12)."""
    ff = str(ffmpeg_tools.ffmpeg)
    clean = tmp_path / "clean.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=20:seed=5,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(clean),
        ],
        check=True,
    )
    lossy = tmp_path / "low.mp3"
    subprocess.run(
        [ff, "-v", "error", "-i", str(clean), "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(lossy)],
        check=True,
    )
    transcode = tmp_path / "low.flac"
    subprocess.run(
        [ff, "-v", "error", "-i", str(lossy), "-c:a", "flac", str(transcode)], check=True
    )
    result = verdict.verify(ffmpeg_tools.ffmpeg, probe(ffmpeg_tools.ffprobe, transcode))
    assert result.bucket == "consistent_lossy", result
    assert result.spectral is not None and result.spectral.knee_hz < 8000
