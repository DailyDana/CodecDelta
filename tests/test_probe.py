"""ffprobe inceleme testleri.

ffprobe JSON'u sabit metin olarak beslenir, boylece testlerin cogu gercek bir
ffmpeg gerektirmez ve CI'da kosar. Ornek JSON'lar bu makinedeki gercek
ciktilardan kisaltildi.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.errors import ProbeError
from app.core.ffmpeg_locate import FFmpegTools
from app.core.probe import AudioStreamInfo, Probe, default_stream, probe

# Kapak resmi gomulu bir FLAC: ffprobe kapagi bir VIDEO izi olarak listeler.
FLAC_JSON = {
    "streams": [
        {
            "index": 0,
            "codec_type": "audio",
            "codec_name": "flac",
            "sample_rate": "44100",
            "channels": 2,
            "channel_layout": "stereo",
            "sample_fmt": "s16",
            "bits_per_raw_sample": "16",
            "disposition": {"default": 0, "attached_pic": 0},
        },
        {
            "index": 1,
            "codec_type": "video",
            "codec_name": "mjpeg",
            "disposition": {"attached_pic": 1},
        },
    ],
    "format": {
        "format_name": "flac",
        "size": "58275906",
        "duration": "568.940340",
        "bit_rate": "819430",
        "tags": {"TITLE": "Example Track", "ARTIST": "Example Artist"},
    },
}

# Cok izli bir film konteyneri: ilk ses izinin MUTLAK index'i 1 ama a:0'dir.
MKV_JSON = {
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "disposition": {}},
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "eac3",
            "sample_rate": "48000",
            "channels": 6,
            "channel_layout": "5.1(side)",
            "bit_rate": "640000",
            "tags": {"language": "rus", "title": "Movie Dubbing"},
            "disposition": {"default": 1},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "tags": {"language": "eng"},
            "disposition": {"default": 0},
        },
        {"index": 3, "codec_type": "subtitle", "codec_name": "subrip", "disposition": {}},
    ],
    "format": {"format_name": "matroska,webm", "size": "10630000000", "duration": "8054.4"},
}


def _probe_from(
    payload: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Probe:
    """probe()'u sahte bir ffprobe ciktisiyla calistirir."""
    media = tmp_path / "sample.bin"
    media.write_bytes(b"x")

    class _Run:
        def stdout_text(self) -> str:
            return json.dumps(payload)

    monkeypatch.setattr("app.core.probe.run_capture", lambda *a, **k: _Run())
    return probe(Path("ffprobe"), media)


def test_attached_picture_is_not_a_video_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gomulu kapak, dosyayi "video" yapmamali ama isaretlenmeli.

    Ayrim onemli: kapak video degildir (analiz akisi degismez) ama konteyner
    bitrate'ini sisirir, o yuzden bitrate cozumunde ayrica dikkate alinir.
    """
    p = _probe_from(FLAC_JSON, tmp_path, monkeypatch)
    assert p.has_video is False
    assert p.has_attached_pic is True
    assert len(p.audio) == 1


def test_audio_ordinal_differs_from_absolute_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-map 0:a:N` sira numarasini ister, konteyner index'ini degil.

    Bunlari karistirmak filmlerde sessizce YANLIS izi analiz etmek demektir.
    """
    p = _probe_from(MKV_JSON, tmp_path, monkeypatch)
    assert [s.audio_index for s in p.audio] == [0, 1]
    assert [s.index for s in p.audio] == [1, 2]


def test_subtitle_streams_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _probe_from(MKV_JSON, tmp_path, monkeypatch)
    assert len(p.audio) == 2


def test_video_container_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _probe_from(MKV_JSON, tmp_path, monkeypatch)
    assert p.has_video is True


def test_file_without_audio_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"streams": [{"index": 0, "codec_type": "video"}], "format": {}}
    with pytest.raises(ProbeError, match="no audio track"):
        _probe_from(payload, tmp_path, monkeypatch)


def test_missing_file_is_rejected() -> None:
    with pytest.raises(ProbeError, match="File not found"):
        probe(Path("ffprobe"), Path("does-not-exist-anywhere.flac"))


def test_lossless_detection() -> None:
    base = AudioStreamInfo(
        audio_index=0,
        index=0,
        codec="flac",
        profile=None,
        sample_rate=44100,
        channels=2,
        channel_layout="stereo",
        sample_fmt="s16",
        bits_per_raw_sample=16,
        bit_rate=None,
        duration=None,
        language=None,
        title=None,
        is_default=True,
    )
    assert base.is_lossless
    assert replace(base, codec="pcm_s24le").is_lossless
    assert replace(base, codec="wavpack").is_lossless
    assert not replace(base, codec="opus").is_lossless
    assert not replace(base, codec="aac").is_lossless


def test_default_stream_prefers_default_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _probe_from(MKV_JSON, tmp_path, monkeypatch)
    assert default_stream(p) == 0


def test_default_stream_falls_back_to_most_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hicbir iz default degilse en cok kanalli secilir.

    Bir konser diskinde 5.1 iz genellikle asil kariskimdir; stereo iz onun
    downmix'idir.
    """
    payload = json.loads(json.dumps(MKV_JSON))
    payload["streams"][1]["disposition"]["default"] = 0
    p = _probe_from(payload, tmp_path, monkeypatch)
    assert default_stream(p) == 0  # 6 kanal > 2 kanal


def test_stream_lookup_rejects_unknown_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _probe_from(MKV_JSON, tmp_path, monkeypatch)
    with pytest.raises(ProbeError, match="a:9"):
        p.stream(9)


def test_label_shows_double_dash_when_bitrate_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bitrate bilinmiyorsa uydurulmaz."""
    p = _probe_from(MKV_JSON, tmp_path, monkeypatch)
    assert "--" in p.stream(1).label()
    assert "640 kbps" in p.stream(0).label()


@pytest.mark.needs_ffmpeg
def test_probe_real_file_is_fast(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Gercek bir dosyayi incelemek konteyner basliklarini okumakla sinirli."""
    import subprocess

    media = tmp_path / "tone.flac"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=3",
            "-ac",
            "2",
            str(media),
        ],
        check=True,
        capture_output=True,
    )
    p = probe(ffmpeg_tools.ffprobe, media)
    assert p.audio[0].codec == "flac"
    assert p.audio[0].sample_rate == 44100
    assert p.audio[0].is_lossless
    assert p.duration is not None and 2.9 < p.duration < 3.1
