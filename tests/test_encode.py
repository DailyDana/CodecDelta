"""Kodlama matrisi, adlandirma ve ilerleme testleri.

Matrisin asil sinavi: tablodaki HER kullanilabilir codec, her modda ve her
secenek degeriyle bu ffmpeg'de gercekten kodluyor mu, ve cikan dosya dogru
codec'i tasiyor mu. Tabloya yazilip hic denenmemis bir secenek, arayuzde
tiklayinca hata veren bir dugme olurdu.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.core.errors import CancelledError
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.core.probe import probe
from app.encode import jobs, matrix, naming
from app.encode.matrix import SPECS, EncodeSettings

# ffprobe'un codec adlari kodlayici adlarindan farkli.
PROBED_CODEC = {
    "libopus": "opus",
    "aac": "aac",
    "aac_mf": "aac",
    "libfdk_aac": "aac",
    "libmp3lame": "mp3",
    "libvorbis": "vorbis",
    "ac3": "ac3",
    "eac3": "eac3",
    "mp2": "mp2",
    "wmav2": "wmav2",
    "flac": "flac",
    "alac": "alac",
    "wavpack": "wavpack",
}


# -- saf ------------------------------------------------------------------------


def test_every_spec_is_consistent() -> None:
    keys = [s.key for s in SPECS]
    assert len(keys) == len(set(keys))
    for spec in SPECS:
        assert spec.modes, spec.key
        if spec.bitrates:
            assert spec.default_bitrate in spec.bitrates, spec.key
        if spec.quality is not None:
            assert spec.quality.low <= spec.quality.default <= spec.quality.high
        for option in spec.options:
            assert option.default in {c[0] for c in option.choices}, (spec.key, option.key)
        assert spec.encoder in PROBED_CODEC


def test_availability_reports_a_reason() -> None:
    fdk = matrix.by_key("fdk_aac")
    assert matrix.availability(fdk, frozenset({"aac"})) is not None
    assert "fdk" in matrix.availability(fdk, frozenset()).lower()  # type: ignore[union-attr]
    assert matrix.availability(matrix.by_key("opus"), frozenset({"libopus"})) is None
    reason = matrix.availability(matrix.by_key("wma"), frozenset())
    assert reason is not None and "wmav2" in reason


def test_extra_args_follow_the_choice() -> None:
    opus = EncodeSettings("opus", "bitrate", 96, options={"vbr": "off", "application": "voip"})
    args = matrix.extra_args(opus)
    assert args[args.index("-vbr") + 1] == "off"
    assert args[args.index("-application") + 1] == "voip"
    assert args[args.index("-frame_duration") + 1] == "20"  # varsayilan
    mp3 = EncodeSettings("mp3", "quality", quality=0)
    assert matrix.extra_args(mp3)[-2:] == ("-q:a", "0")
    assert matrix.describe(mp3) == "mp3V0"
    assert matrix.describe(opus) == "opus96k"
    assert matrix.describe(EncodeSettings("flac", "lossless")) == "flac"
    # AAC-MF 16 bit girdi ister; sabit arguman her zaman eklenir
    assert "-sample_fmt" in matrix.extra_args(EncodeSettings("aac_mf", "bitrate", 128))


def test_default_aac_has_noise_substitution_off_and_warns_when_on() -> None:
    off = EncodeSettings("aac", "bitrate", 192)
    args = matrix.extra_args(off)
    assert args[args.index("-aac_pns") + 1] == "0"
    assert matrix.warnings(off) == ()
    on = EncodeSettings("aac", "bitrate", 192, options={"aac_pns": "1"})
    assert matrix.warnings(on) and "SNR" in matrix.warnings(on)[0]


def test_invalid_choices_are_rejected() -> None:
    for bad in (
        EncodeSettings("opus", "quality", quality=5),
        EncodeSettings("vorbis", "quality", quality=11),
        EncodeSettings("opus", "bitrate", 128, options={"nope": "1"}),
        EncodeSettings("opus", "bitrate", 128, options={"vbr": "maybe"}),
        EncodeSettings("flac", "bitrate", 128),
    ):
        with pytest.raises(ValueError):
            matrix.extra_args(bad)


# -- adlandirma -------------------------------------------------------------------


def test_output_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "song.flac"
    source.write_bytes(b"x")
    first = naming.output_path(source, "opus128k", "opus")
    assert first.name == "song_enc_opus128k.opus"
    first.write_bytes(b"y")
    assert naming.output_path(source, "opus128k", "opus").name == "song_enc_opus128k_2.opus"


def test_session_reservations_prevent_two_jobs_taking_one_name(tmp_path: Path) -> None:
    """Iki is diske yazmadan ayni adi secemez (Aniflow'daki UsedOuts kurali)."""
    source = tmp_path / "song.flac"
    source.write_bytes(b"x")
    taken: set[Path] = set()
    a = naming.output_path(source, "mp3V2", "mp3", taken=taken)
    b = naming.output_path(source, "mp3V2", "mp3", taken=taken)
    assert a != b and b.name.endswith("_2.mp3")


def test_output_never_equals_the_source(tmp_path: Path) -> None:
    source = tmp_path / "song_enc_flac.flac"
    source.write_bytes(b"x")
    out = naming.output_path(tmp_path / "song.flac", "flac", "flac", directory=tmp_path)
    assert out != source


def test_output_directory_is_honoured(tmp_path: Path) -> None:
    source = tmp_path / "a" / "song.flac"
    target = tmp_path / "out"
    assert naming.output_path(source, "x", "opus", directory=target).parent == target


def test_progress_parser() -> None:
    assert jobs.parse_progress("out_time_us=1500000") == pytest.approx(1.5)
    assert jobs.parse_progress("out_time_ms=2000000") == pytest.approx(2.0)
    assert jobs.parse_progress("out_time_us=N/A") is None
    assert jobs.parse_progress("progress=continue") is None


# -- gercek ffmpeg ------------------------------------------------------------------


@pytest.fixture(scope="module")
def source(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("encode") / "src.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=3:seed=2",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(path),
        ],
        check=True,
    )
    return path


def _variants() -> list[tuple[str, EncodeSettings]]:
    """Her spec icin: varsayilan her mod + her secenegin varsayilan olmayan her degeri."""
    out = []
    for spec in SPECS:
        for mode in spec.modes:
            base = EncodeSettings(
                spec.key,
                mode,
                bitrate_kbps=spec.default_bitrate if mode == "bitrate" else None,
                quality=spec.quality.default if mode == "quality" and spec.quality else None,
            )
            out.append((f"{spec.key}-{mode}", base))
        for option in spec.options:
            for value, _ in option.choices:
                if value == option.default:
                    continue
                mode = spec.modes[0]
                out.append(
                    (
                        f"{spec.key}-{option.key}={value}",
                        EncodeSettings(
                            spec.key,
                            mode,
                            bitrate_kbps=spec.default_bitrate if mode == "bitrate" else None,
                            options={option.key: value},
                        ),
                    )
                )
    return out


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize(("label", "choice"), _variants(), ids=[v[0] for v in _variants()])
def test_every_variant_encodes_with_this_ffmpeg(
    ffmpeg_tools: FFmpegTools, source: Path, tmp_path: Path, label: str, choice: EncodeSettings
) -> None:
    spec = choice.spec
    if matrix.availability(spec, ffmpeg_tools.caps.encoders) is not None:
        pytest.skip(f"{spec.encoder} yok")
    output = tmp_path / f"out.{spec.extension}"
    job = jobs.EncodeJob(
        source=source,
        output=output,
        codec=spec.encoder,
        bitrate_kbps=choice.bitrate_kbps,
        source_rate=44100,
        extra=matrix.extra_args(choice),
    )
    jobs.run(ffmpeg_tools.ffmpeg, job)
    info = probe(ffmpeg_tools.ffprobe, output)
    assert info.audio[0].codec == PROBED_CODEC[spec.encoder]
    if spec.key == "opus":
        assert info.audio[0].sample_rate == 48000


@pytest.mark.needs_ffmpeg
def test_progress_rises_to_one(ffmpeg_tools: FFmpegTools, source: Path, tmp_path: Path) -> None:
    seen: list[float] = []
    job = jobs.EncodeJob(source, tmp_path / "p.mp3", "libmp3lame", 128, source_rate=44100)
    jobs.run_with_progress(ffmpeg_tools.ffmpeg, job, duration=3.0, on_progress=seen.append)
    assert seen and seen[-1] == 1.0
    assert seen == sorted(seen)
    assert (tmp_path / "p.mp3").exists()


@pytest.mark.needs_ffmpeg
def test_cancel_kills_and_removes_the_partial_file(
    ffmpeg_tools: FFmpegTools, tmp_path: Path
) -> None:
    long_source = tmp_path / "long.wav"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=48000:duration=900:seed=3",
            "-ac",
            "2",
            str(long_source),
        ],
        check=True,
    )
    token = CancelToken()
    output = tmp_path / "c.flac"

    def on_progress(fraction: float) -> None:
        if fraction > 0.0:
            token.cancel()

    job = jobs.EncodeJob(
        long_source, output, "flac", source_rate=48000, extra=("-compression_level", "12")
    )
    with pytest.raises(CancelledError):
        jobs.run_with_progress(
            ffmpeg_tools.ffmpeg, job, duration=900.0, on_progress=on_progress, cancel=token
        )
    assert not output.exists()


@pytest.mark.needs_ffmpeg
def test_failure_raises_and_leaves_no_file(
    ffmpeg_tools: FFmpegTools, source: Path, tmp_path: Path
) -> None:
    from app.core.errors import FFmpegFailedError

    output = tmp_path / "bad.mp3"
    job = jobs.EncodeJob(source, output, "libmp3lame", 128, extra=("-q:a", "banana"))
    with pytest.raises(FFmpegFailedError):
        jobs.run_with_progress(ffmpeg_tools.ffmpeg, job, duration=3.0, on_progress=lambda _: None)
    assert not output.exists()


# -- kodlayici sinirlari (denetim D14, D15) -------------------------------------------

_LAME_HELP = """Encoder libmp3lame [libmp3lame MP3 (MPEG audio layer 3)]:
    Supported sample rates: 44100 48000 32000 22050 24000 16000 11025 12000 8000
    Supported sample formats: s32p fltp s16p
    Supported channel layouts: mono stereo
"""
_AC3_HELP = """    Supported sample rates: 48000 44100 32000
    Supported channel layouts: mono stereo 2 channels (FC+LFE) 2.1 5.1(side) 5.1
"""


def test_encoder_limits_are_read_from_ffmpeg_help() -> None:
    lame = jobs.parse_limits(_LAME_HELP)
    assert lame.rates is not None and 44100 in lame.rates and 96000 not in lame.rates
    assert lame.max_channels == 2
    assert jobs.parse_limits(_AC3_HELP).max_channels == 6
    vorbis = "Encoder libvorbis\n    Supported sample formats: fltp\n"
    assert jobs.parse_limits(vorbis) == jobs.EncoderLimits()


def _job(codec: str, rate: int, bitrate: int | None = 192) -> jobs.EncodeJob:
    return jobs.EncodeJob(
        source=Path("in.flac"),
        output=Path("out.x"),
        codec=codec,
        bitrate_kbps=bitrate,
        source_rate=rate,
    )


def test_unsupported_rate_is_converted_explicitly_with_soxr() -> None:
    """96 kHz MP3 ffmpeg'in varsayilan resampler'iyla sessizce 48'e iniyordu (D15)."""
    job, notes = jobs.prepare(_job("libmp3lame", 96000), jobs.parse_limits(_LAME_HELP), channels=2)
    assert job.target_rate == 48000 and notes
    args = jobs.build_args(job)
    chain = args[args.index("-af") + 1]
    assert "resampler=soxr" in chain and "aresample=48000" in chain
    # Desteklenen hizda dokunulmaz
    same, quiet = jobs.prepare(_job("libmp3lame", 44100), jobs.parse_limits(_LAME_HELP), channels=2)
    assert same.target_rate is None and not quiet and "-af" not in jobs.build_args(same)


def test_rate_choice_prefers_the_next_rate_up() -> None:
    """44.1 -> Opus 48 (bilgi kaybi yok); 96 -> MP3 48 (en yuksek kabul edilen)."""
    opus = jobs.EncoderLimits(rates=(48000, 24000, 16000, 12000, 8000))
    assert jobs.prepare(_job("libopus", 44100), opus, channels=2)[0].target_rate == 48000
    assert jobs.prepare(_job("libopus", 96000), opus, channels=2)[0].target_rate == 48000


def test_vorbis_bitrate_mode_is_limited_to_48_khz() -> None:
    """libvorbis bit hizi modu 88.2/96 kHz'te "encoder setup failed" (D14, olculdu)."""
    none = jobs.EncoderLimits()
    assert jobs.prepare(_job("libvorbis", 96000), none, channels=2)[0].target_rate == 48000
    quality = _job("libvorbis", 96000, bitrate=None)
    assert jobs.prepare(quality, none, channels=2)[0].target_rate is None


def test_too_many_channels_stops_before_encoding() -> None:
    """WMA/MP3'e 5.1 kaynak ham ffmpeg hatasiyla dusuyordu; downmix YAPILMAZ (D14)."""
    with pytest.raises(jobs.EncodeUnsupportedError):
        jobs.prepare(_job("libmp3lame", 48000), jobs.parse_limits(_LAME_HELP), channels=6)
    wma = jobs.EncoderLimits(rates=None, max_channels=2)
    with pytest.raises(jobs.EncodeUnsupportedError):
        jobs.prepare(_job("wmav2", 48000), wma, channels=6)


def test_opus_side_layouts_are_relabelled() -> None:
    """libopus "5.1(side)" duzenini reddediyordu; kanallar ayni, yalnizca etiket (D14)."""
    job, notes = jobs.prepare(
        _job("libopus", 48000), jobs.EncoderLimits(), channels=6, channel_layout="5.1(side)"
    )
    assert job.filters == ("channelmap=channel_layout=5.1",) and notes
    with pytest.raises(jobs.EncodeUnsupportedError):
        jobs.prepare(
            _job("libopus", 48000), jobs.EncoderLimits(), channels=8, channel_layout="7.1(wide)"
        )


@pytest.mark.needs_ffmpeg
def test_wma_limits_come_from_measurement(ffmpeg_tools: FFmpegTools) -> None:
    """wmav2 sinirlarini bildirmiyor; olculmus sinirlar uygulanir."""
    limits = jobs.encoder_limits(ffmpeg_tools.ffmpeg, "wmav2")
    assert limits.max_channels == 2
    assert limits.rates is not None and max(limits.rates) == 48000
