"""PCM akisi testleri.

En onemlisi `test_readinto_throughput_floor`: bu depoda ffmpeg borusundan
`read(n)` ile okumanin yasak olmasinin sebebi olculmus bir performans
patolojisi (16 MB'lik `read` cagrilari 57 kat yavas). Test o kurali kilitler.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, PcmStream, ResampleCfg, pcm_args

SRC = Path("dummy.flac")


def test_video_is_never_decoded() -> None:
    """20 GB'lik bir MKV verildiginde video paketine dokunulmamali."""
    args = pcm_args(SRC, stream_index=0)
    assert "-vn" in args
    assert "-sn" in args
    assert "-dn" in args
    assert "-map" in args
    assert args[args.index("-map") + 1] == "0:a:0"


def test_stream_index_is_honoured() -> None:
    args = pcm_args(SRC, stream_index=3)
    assert args[args.index("-map") + 1] == "0:a:3"


def test_native_rate_adds_no_resampler() -> None:
    """Dosya-ici olcumler resample edilmeden yapilmali.

    Kesim frekansi analizi resampler'in kendi duvarini codec'in kesimi
    sanabilir; bu yuzden rate=None yolunda hic aresample olmamali.
    """
    args = pcm_args(SRC, rate=None)
    assert "-af" not in args
    assert not any("aresample" in a for a in args)


def test_resampling_always_states_cutoff_explicitly() -> None:
    """soxr'in varsayilan cutoff'u tepe frekanslari sessizce kirpiyor.

    Olculen: 44.1 -> 48 kHz yukari orneklemede bile varsayilan ayar 21.5 kHz'de
    -30 dB bastirma uyguluyor. Cutoff acikca verilmezse arac temiz bir FLAC'i
    "20.9 kHz'de kesilmis" diye yanlis isaretler.
    """
    args = pcm_args(SRC, rate=48000)
    chain = args[args.index("-af") + 1]
    assert "aresample=48000" in chain
    assert "resampler=soxr" in chain
    assert "precision=28" in chain
    assert "cutoff=0.99" in chain


def test_input_seek_comes_before_input() -> None:
    """`-ss` girdiden once olmali; sonra verilirse tum dosya cozulur."""
    args = pcm_args(SRC, start=30.0, duration=15.0)
    assert args.index("-ss") < args.index("-i")
    assert args.index("-t") > args.index("-i")


def test_extra_filters_precede_resampler() -> None:
    """Kullanici zinciri once, yeniden ornekleme en sonda uygulanmali."""
    args = pcm_args(SRC, rate=48000, af_chain=["volume=0.5"])
    chain = args[args.index("-af") + 1]
    assert chain.index("volume=0.5") < chain.index("aresample")


def test_channel_count_is_explicit() -> None:
    args = pcm_args(SRC, channels=1)
    assert args[args.index("-ac") + 1] == "1"


def test_unknown_sample_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="desteklenmeyen"):
        pcm_args(SRC, fmt="u8")


def test_resample_cfg_expression() -> None:
    cfg = ResampleCfg(precision=20, cutoff=0.95)
    assert cfg.filter_expr(44100) == (
        "aformat=sample_fmts=dbl,aresample=44100:resampler=soxr:precision=20:cutoff=0.95"
    )
    assert DEFAULT_RESAMPLE.cutoff == 0.99


# -- gercek ffmpeg gerektiren testler ---------------------------------------


def _sine_args(seconds: float, rate: int, channels: int) -> list[str]:
    """lavfi ile uretilmis ses: fixture dosyasi gerektirmez."""
    layout = "stereo" if channels == 2 else "mono"
    return [
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate={rate}:duration={seconds}",
        "-af",
        f"aformat=channel_layouts={layout}",
        "-f",
        "f32le",
        "-",
    ]


@pytest.mark.needs_ffmpeg
def test_blocks_yield_exact_frame_count(ffmpeg_tools: FFmpegTools) -> None:
    seconds, rate, channels = 2.0, 44100, 2
    total = 0
    with PcmStream(
        ffmpeg_tools.ffmpeg,
        _sine_args(seconds, rate, channels),
        sample_rate=rate,
        channels=channels,
    ) as stream:
        for block in stream.blocks(4096):
            assert block.dtype == np.float32
            assert block.shape[1] == channels
            total += block.shape[0]
    assert total == int(seconds * rate)


@pytest.mark.needs_ffmpeg
def test_blocks_reuse_the_same_buffer(ffmpeg_tools: FFmpegTools) -> None:
    """Uretilen diziler kopyasizdir; saklamak isteyen taraf kopyalamali.

    Bu davranis bilincli (saniyede yuzlerce MB akarken her blok icin yeni dizi
    ayirmak GC'yi bosuna calistirir), ama sessiz kalirsa cagri yerlerinde cok
    kotu bir hataya yol acar. Test sozlesmeyi acikca yaziyor.
    """
    seen: list[np.ndarray] = []
    with PcmStream(
        ffmpeg_tools.ffmpeg,
        _sine_args(1.0, 44100, 2),
        sample_rate=44100,
        channels=2,
    ) as stream:
        for block in stream.blocks(4096):
            seen.append(block)
            if len(seen) == 2:
                break
    # numpy her blokta yeni bir view nesnesi uretir; ozdes olan sey ALTTAKI
    # bellek. Veri isaretcisini karsilastirmak sozlesmeyi tam olarak yazar.
    first = seen[0].__array_interface__["data"][0]
    second = seen[1].__array_interface__["data"][0]
    assert first == second, "bloklar ayni tamponu paylasmiyor"


@pytest.mark.needs_ffmpeg
@pytest.mark.slow
def test_readinto_throughput_floor(ffmpeg_tools: FFmpegTools) -> None:
    """Boru verimi regresyon testi.

    Olculen taban bu makinede ~104 MB/s (readinto). `read(n)` yoluna donulurse
    buyuk tamponlarda 4 MB/s'ye duser. Esik 30 MB/s: gercek patolojiyi
    yakalayacak kadar yuksek, yavas bir CI makinesinde yanlis alarm vermeyecek
    kadar dusuk.
    """
    seconds, rate, channels = 60.0, 44100, 2
    expected_bytes = int(seconds * rate) * channels * 4

    start = time.perf_counter()
    total_frames = 0
    with PcmStream(
        ffmpeg_tools.ffmpeg,
        _sine_args(seconds, rate, channels),
        sample_rate=rate,
        channels=channels,
    ) as stream:
        for block in stream.blocks(rate):  # 1 s'lik bloklar
            total_frames += block.shape[0]
    elapsed = time.perf_counter() - start

    assert total_frames == int(seconds * rate)
    mb_per_s = expected_bytes / (1 << 20) / max(elapsed, 1e-9)
    assert mb_per_s > 30.0, f"boru verimi cok dustu: {mb_per_s:.1f} MB/s"


@pytest.mark.needs_ffmpeg
def test_early_break_does_not_hang(ffmpeg_tools: FFmpegTools) -> None:
    """Tuketici erken cikarsa ffmpeg dolu boruda asili kalmamali.

    __exit__ EOF'a varilmadiysa surec agacini oldurur; bu test o yolun
    kilitlenmedigini dogrular.
    """
    start = time.perf_counter()
    with PcmStream(
        ffmpeg_tools.ffmpeg,
        _sine_args(600.0, 48000, 2),
        sample_rate=48000,
        channels=2,
    ) as stream:
        for _ in stream.blocks(1024):
            break
    assert time.perf_counter() - start < 20.0


# -- bozuk akislar (denetim D35, D39) ------------------------------------------------


def test_zero_channels_is_rejected_before_starting() -> None:
    """channels=0 ile 30 s bekleyip ham TimeoutExpired geliyordu (D39)."""
    with pytest.raises(ValueError):
        PcmStream(Path("ffmpeg"), [], sample_rate=44100, channels=0)


@pytest.mark.needs_ffmpeg
def test_a_partial_frame_is_an_error(ffmpeg_tools: FFmpegTools) -> None:
    """Akis cercevenin ortasinda biterse kalan baytlar sessizce atiliyordu (D39)."""
    from app.core.errors import UnsupportedInputError

    args = [
        *("-nostdin", "-v", "error", "-f", "lavfi"),
        *("-i", "sine=sample_rate=44100:duration=1", "-af", "atrim=end_sample=1001"),
        *("-ac", "1", "-f", "f32le", "-"),
    ]
    # 1001 mono ornek = 4004 bayt; stereo cerceve 8 bayt: son cerceve yarim.
    stream = PcmStream(ffmpeg_tools.ffmpeg, args, sample_rate=44100, channels=2)
    with pytest.raises(UnsupportedInputError), stream:
        for _ in stream.blocks(4096):
            pass


@pytest.mark.needs_ffmpeg
def test_non_finite_samples_become_silence(ffmpeg_tools: FFmpegTools, tmp_path: Path) -> None:
    """Float WAV'da tek NaN/inf ornek analizi ham ValueError ile dusuruyordu (D35)."""
    import subprocess

    from app.core.probe import probe
    from app.single import verdict

    samples = np.random.default_rng(1).standard_normal((44100 * 20, 2)).astype("<f4") * 0.1
    samples[1000, 0], samples[5000, 1] = np.nan, np.inf
    wav = tmp_path / "nan.wav"
    subprocess.run(
        [
            str(ffmpeg_tools.ffmpeg),
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-i",
            "-",
            "-c:a",
            "pcm_f32le",
            str(wav),
        ],
        input=samples.tobytes(),
        check=True,
    )
    result = verdict.verify(ffmpeg_tools.ffmpeg, probe(ffmpeg_tools.ffprobe, wav))
    notes = [n for n in result.notes if getattr(n, "key", "") == "single.nonfinite"]
    assert notes and notes[0].params["count"] == 2


def test_stderr_is_drained_in_chunks_not_bytes() -> None:
    """readline() tamponsuz boruda bayt bayt okuyordu: 12 MB 8.9 s (D38)."""
    import io

    from app.core.ffmpeg_runner import StderrCollector

    class Unbuffered(io.RawIOBase):
        """Tamponsuz boru gibi: her `read` en fazla istenen kadar, tek cagri."""

        def __init__(self, data: bytes) -> None:
            self._data = memoryview(data)
            self.calls = 0

        def readable(self) -> bool:
            return True

        def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
            self.calls += 1
            n = min(len(buffer), len(self._data))
            buffer[:n] = self._data[:n]
            self._data = self._data[n:]
            return n

    lines = b"".join(b"frame=%07d warning: something\r\n" % i for i in range(300_000))
    stream = Unbuffered(lines + b"last line without newline")
    collector = StderrCollector(stream)  # type: ignore[arg-type]
    collector.join(timeout=60)
    tail = collector.tail().splitlines()
    assert tail[-1] == "last line without newline"
    assert tail[-2] == "frame=0299999 warning: something"
    # 12 MB'lik akis parca parca okunmali; bayt bayt okuma milyonlarca cagri olurdu.
    assert stream.calls < 1000
