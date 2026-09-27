"""Tek gecisli karsilastirma boru hatti.

    probe -> zarflar (8 kHz, iki dosya eszamanli) -> hizalama plani
          -> ana gecis (iki ffmpeg akisi, ayni anda) -> capraz spektrum
          -> olcum tabani -> sonuc

Ana gecis iki dosyayi da BASTAN akitir ve hizlamayi ornek atlayarak uygular:
tamsayi gecikme kadar ornek geride olan akistan atilir, kesirli kisim
spektrumda faz rampasiyla. Kesir +-0.5'e yuvarlanir; faz rampasinin olculen
tabani orada -67 dB (bkz. `stft.phase_shift`).

Bellek dosya suresinden bagimsizdir: akislardan gelen bloklar STFT'ye verilip
birakilir, yalnizca bin basina uc toplam tutulur.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from app.align import envelope, plan
from app.align.plan import AlignmentPlan
from app.compare import calibration
from app.compare.reader import FFmpegWindowReader, Source, seek_exact
from app.compare.result import BandResult, ComparisonResult, FileSummary, Status
from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, PcmStream, ResampleCfg, open_pcm
from app.core.probe import AudioStreamInfo, Probe, default_stream, probe
from app.dsp.accum import CrossSpectrum, hz_to_bin
from app.dsp.stft import StreamingStft, phase_shift
from app.dsp.transforms import db

DEFAULT_FFT_SIZE = 4096

# Rapor bantlari (Hz). Ust sinir iki dosyanin kucuk Nyquist'ine kirpilir: bir
# dosyanin hic tasiyamayacagi bant olculemez, yalnizca "yok" olarak raporlanir.
BAND_EDGES_HZ: tuple[float, ...] = (
    20.0,
    1000.0,
    4000.0,
    8000.0,
    12000.0,
    16000.0,
    18000.0,
    20000.0,
    22000.0,
    24000.0,
)
BROADBAND_HZ = (20.0, 20000.0)

_BLOCK_FRAMES = 1 << 16

ProgressFn = Callable[[float], None]


@dataclass(frozen=True)
class Track:
    """Karsilastirmaya giren bir ses izi."""

    path: Path
    info: Probe
    stream_index: int

    @property
    def stream(self) -> AudioStreamInfo:
        return self.info.stream(self.stream_index)

    def summary(self) -> FileSummary:
        s = self.info.stream(self.stream_index)
        return FileSummary(
            path=self.path,
            container=self.info.container,
            codec=s.codec,
            sample_rate=s.sample_rate,
            channels=s.channels,
            duration=self.info.duration,
        )

    def source(self, target_rate: int) -> Source:
        s = self.info.stream(self.stream_index)
        return Source(
            path=self.path,
            stream_index=self.stream_index,
            channels=s.channels,
            source_rate=s.sample_rate,
            target_rate=target_rate if target_rate != s.sample_rate else None,
            exact_seek=seek_exact(self.info.container, s.codec),
        )


def open_track(ffprobe: Path, path: Path, stream_index: int | None = None) -> Track:
    info = probe(ffprobe, path)
    index = default_stream(info) if stream_index is None else stream_index
    info.stream(index)  # yoksa ProbeError
    return Track(path=path, info=info, stream_index=index)


def band_layout(analysis_rate: int, nyquist_hz: float) -> list[tuple[float, float]]:
    """Rapor bantlari, `nyquist_hz`'e (iki dosyanin kucuk Nyquist'i) kirpilmis."""
    bands = []
    for lo, hi in pairwise(BAND_EDGES_HZ):
        if lo >= nyquist_hz:
            break
        bands.append((lo, min(hi, nyquist_hz)))
    return bands


def _mid_side(
    block: np.ndarray, channel_map: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray | None]:
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float64), None
    left = block[:, channel_map[0] if channel_map else 0].astype(np.float64)
    right = block[:, channel_map[1] if channel_map else 1].astype(np.float64)
    return (left + right) * 0.5, (left - right) * 0.5


def _skipped(blocks: Iterator[np.ndarray], skip: int) -> Iterator[np.ndarray]:
    """Akisin ilk `skip` cercevesini atar. Bloklar kopyalanir (tampon yeniden kullanilir)."""
    for block in blocks:
        if skip >= block.shape[0]:
            skip -= block.shape[0]
            continue
        yield block[skip:].copy()
        skip = 0


class _Pairer:
    """Iki akisi ayni uzunlukta parcalar halinde eslestirir.

    Iki STFT'ye her zaman AYNI sayida ornek verilir; boylece cerceve `t` iki
    tarafta da ayni zamani temsil eder ve spektrumlar dogrudan eslesir.
    """

    def __init__(self, a: Iterator[np.ndarray], b: Iterator[np.ndarray]) -> None:
        self._a, self._b = a, b
        self._buf_a: list[np.ndarray] = []
        self._buf_b: list[np.ndarray] = []
        self._len_a = self._len_b = 0

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        while True:
            if self._len_a == 0 and not self._fill(self._a, self._buf_a, "a"):
                return
            if self._len_b == 0 and not self._fill(self._b, self._buf_b, "b"):
                return
            n = min(self._len_a, self._len_b)
            yield self._take(self._buf_a, n, "a"), self._take(self._buf_b, n, "b")

    def _fill(self, source: Iterator[np.ndarray], buf: list[np.ndarray], side: str) -> bool:
        block = next(source, None)
        if block is None:
            return False
        buf.append(block)
        if side == "a":
            self._len_a += block.shape[0]
        else:
            self._len_b += block.shape[0]
        return True

    def _take(self, buf: list[np.ndarray], n: int, side: str) -> np.ndarray:
        data = np.concatenate(buf) if len(buf) > 1 else buf[0]
        head, tail = data[:n], data[n:]
        buf.clear()
        if tail.shape[0]:
            buf.append(tail)
        if side == "a":
            self._len_a = tail.shape[0]
        else:
            self._len_b = tail.shape[0]
        return head


def _not_measured(
    reference: Track,
    test: Track,
    alignment: AlignmentPlan,
    rate: int,
    *,
    status: Status,
    notes: list[str],
) -> ComparisonResult:
    return ComparisonResult(
        reference=reference.summary(),
        test=test.summary(),
        plan=alignment,
        status=status,
        analysis_rate=rate,
        notes=tuple(notes),
    )


def compare(
    ffmpeg: Path,
    reference: Track,
    test: Track,
    *,
    fft_size: int = DEFAULT_FFT_SIZE,
    resample: ResampleCfg = DEFAULT_RESAMPLE,
    cancel: CancelToken | None = None,
) -> ComparisonResult:
    """Iki izi hizalar ve farklarini olcer."""
    ref_stream, test_stream = reference.stream, test.stream
    rate = max(ref_stream.sample_rate, test_stream.sample_rate)
    ref_source, test_source = reference.source(rate), test.source(rate)
    notes: list[str] = []

    # -- hizalama ----------------------------------------------------------
    with ThreadPoolExecutor(max_workers=2) as pool:
        env_futures = [
            pool.submit(envelope.build, ffmpeg, t.path, stream_index=t.stream_index, cancel=cancel)
            for t in (reference, test)
        ]
        ref_env, test_env = (f.result() for f in env_futures)

    alignment = plan.build(
        ref_env,
        test_env,
        FFmpegWindowReader(ffmpeg, ref_source, resample=resample, cancel=cancel),
        FFmpegWindowReader(ffmpeg, test_source, resample=resample, cancel=cancel),
        rate,
    )
    if alignment.verdict not in ("aligned", "different_master"):
        return _not_measured(reference, test, alignment, rate, status="not_comparable", notes=notes)
    if alignment.drift is not None and alignment.drift.needs_tracking:
        notes.append(
            f"clock drift of {alignment.drift.ppm:+.1f} ppm needs block-local delay tracking, "
            "which is not implemented yet; measuring without it would report the drift as noise"
        )
        return _not_measured(reference, test, alignment, rate, status="not_measured", notes=notes)
    if alignment.verdict == "different_master":
        notes.append("different master: the difference is not a codec difference")

    # -- ana gecis -----------------------------------------------------------
    whole = round(alignment.delay_samples)
    fraction = alignment.delay_samples - whole
    mid, side, samples = _main_pass(
        ffmpeg,
        ref_source,
        test_source,
        whole,
        fraction,
        alignment.channel_map,
        fft_size=fft_size,
        resample=resample,
        cancel=cancel,
    )
    if mid.frames < 2:
        notes.append("overlap too short to measure")
        return _not_measured(reference, test, alignment, rate, status="not_measured", notes=notes)

    # -- bantlar + taban -----------------------------------------------------
    nyquist = min(ref_stream.sample_rate, test_stream.sample_rate) / 2.0
    if ref_stream.sample_rate != test_stream.sample_rate:
        # Yeniden ornekleyen zincir kesimin ustunu TANIM GEREGI gecirmez; o bant
        # olculemez, "yok"tur. Ilk surum bandi Nyquist'e kadar raporluyordu ve
        # gercek bir FLAC/Opus ciftinde 22.00-22.05 kHz'i, tabani -11 dB iken,
        # "olculebilir" gosterdi.
        nyquist *= resample.cutoff
    layout = band_layout(rate, nyquist)
    broadband_hz = (BROADBAND_HZ[0], min(BROADBAND_HZ[1], nyquist))
    floors = calibration.measure_floor(
        ffmpeg,
        reference.path,
        stream_index=reference.stream_index,
        channels=ref_stream.channels,
        source_rate=ref_stream.sample_rate,
        other_rate=test_stream.sample_rate,
        fractional_delay=fraction,
        bands_hz=[*layout, broadband_hz],
        size=fft_size,
        start=_excerpt_start(reference.info.duration),
        resample=resample,
        cancel=cancel,
    )

    gain = mid.gain(
        hz_to_bin(broadband_hz[0], rate, fft_size), hz_to_bin(broadband_hz[1], rate, fft_size)
    )

    def band(lo_hz: float, hi_hz: float, floor_db: float) -> BandResult:
        lo, hi = hz_to_bin(lo_hz, rate, fft_size), hz_to_bin(hi_hz, rate, fft_size)
        hi = max(hi, lo + 1)
        return BandResult(
            lo_hz=lo_hz,
            hi_hz=hi_hz,
            mid=mid.band(lo, hi, gain=gain),
            side=side.band(lo, hi, gain=gain) if side is not None else None,
            floor_db=floor_db,
        )

    bands = tuple(band(lo, hi, f) for (lo, hi), f in zip(layout, floors[:-1], strict=True))
    return ComparisonResult(
        reference=reference.summary(),
        test=test.summary(),
        plan=alignment,
        status="measured",
        analysis_rate=rate,
        bands=bands,
        broadband=band(*broadband_hz, floors[-1]),
        gain_db=db(abs(gain)),
        polarity=-1 if gain < 0 else 1,
        frames=mid.frames,
        samples=samples,
        notes=tuple(notes),
    )


def _excerpt_start(duration: float | None) -> float | None:
    """Taban kesiti icin baslangic: kaydin ortasina yakin, sessiz giris/cikistan uzak."""
    if duration is None or duration <= calibration.DEFAULT_EXCERPT_S * 1.5:
        return None
    return max(0.0, duration / 2.0 - calibration.DEFAULT_EXCERPT_S / 2.0)


def _main_pass(
    ffmpeg: Path,
    ref_source: Source,
    test_source: Source,
    whole: int,
    fraction: float,
    channel_map: tuple[int, ...],
    *,
    fft_size: int,
    resample: ResampleCfg,
    cancel: CancelToken | None,
) -> tuple[CrossSpectrum, CrossSpectrum | None, int]:
    """Iki dosyayi bastan akitip capraz spektrumu biriktirir.

    test[T] = reference[T - d]: d > 0 ise test gerde, test'in basindan d
    ornek atilir; d < 0 ise reference'in basindan.
    """
    stereo = ref_source.channels >= 2
    stfts = {
        name: StreamingStft(fft_size) for name in ("ref_mid", "test_mid", "ref_side", "test_side")
    }
    mid = CrossSpectrum(stfts["ref_mid"].bins)
    side = CrossSpectrum(stfts["ref_side"].bins) if stereo else None
    total = 0

    def stream(source: Source) -> PcmStream:
        return open_pcm(
            ffmpeg,
            source.path,
            sample_rate=source.rate,
            channels=source.channels,
            stream_index=source.stream_index,
            rate=source.target_rate,
            resample=resample,
            cancel=cancel,
        )

    identity = tuple(range(ref_source.channels))
    with stream(ref_source) as ref_pcm, stream(test_source) as test_pcm:
        ref_blocks = _skipped(ref_pcm.blocks(_BLOCK_FRAMES), max(0, -whole))
        test_blocks = _skipped(test_pcm.blocks(_BLOCK_FRAMES), max(0, whole))
        for ref_block, test_block in _Pairer(ref_blocks, test_blocks):
            total += ref_block.shape[0]
            ref_mid, ref_side = _mid_side(ref_block, identity)
            test_mid, test_side = _mid_side(test_block, channel_map)
            # Kesirli gecikme referansa uygulanir: plan `fractional_shift(ref, frac)`in
            # test'le eslestigini olctu.
            mid.add(
                phase_shift(stfts["ref_mid"].push(ref_mid), fraction, fft_size),
                stfts["test_mid"].push(test_mid),
            )
            if side is not None and ref_side is not None and test_side is not None:
                side.add(
                    phase_shift(stfts["ref_side"].push(ref_side), fraction, fft_size),
                    stfts["test_side"].push(test_side),
                )
    return mid, side, total
