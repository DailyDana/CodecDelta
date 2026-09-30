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
from typing import Protocol

import numpy as np

from app.align import envelope, plan
from app.align.plan import AlignmentPlan
from app.compare import calibration, tracking
from app.compare.reader import FFmpegWindowReader, Source, seek_exact
from app.compare.result import BandResult, ComparisonResult, FileSummary, Status
from app.compare.tracking import DelayModel
from app.core.ffmpeg_runner import CancelToken
from app.core.ffmpeg_stream import DEFAULT_RESAMPLE, PcmStream, ResampleCfg, open_pcm
from app.core.messages import Message
from app.core.probe import AudioStreamInfo, Probe, default_stream, probe
from app.dsp import warp
from app.dsp.accum import CrossSpectrum, hz_to_bin
from app.dsp.stft import StreamingStft, hann
from app.dsp.transforms import db
from app.psycho.nmr import NmrAccumulator

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

# Hizalama denetimi blok uzunlugu (s) ve PHAT tepesinin sifirdan en fazla
# uzakligi (ornek). Cerceveler zaten hizali oldugu icin tepe sifirdadir;
# +-2 kesirli artiga ve pencere etkisine pay birakir.
_GATE_BLOCK_S = 1.0
_GATE_MAX_LAG = 2
# Olcume katilan blok orani bunun altindaysa sonuc olculmemis sayilir: plan
# gecikmesi dosyanin cogunluguna ait degil.
_GATE_MIN_KEPT = 0.5

# Raporlanan en ust frekans, Nyquist'in kesri olarak (bkz. `compare`).
NYQUIST_FRACTION = 0.99

_BLOCK_FRAMES = 1 << 16

# Asama bildirimi: kisa, sabit bir anahtar ("align", "measure", "floor").
# Motor metin uretmez; arayuz anahtari kendi diline cevirir.
StageFn = Callable[[str], None]


def _noop(_: str) -> None:
    return None


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


class _Sink(Protocol):
    """Gecislerin spektrum ciftlerini verdigi hedef."""

    def add(self, a: np.ndarray, b: np.ndarray) -> None: ...


class _Fanout:
    """Ayni spektrum ciftini birden fazla biriktiriciye dagitir.

    Capraz spektrum ve NMR ayni gecisten beslenir; ikinci bir gecis yok.
    """

    def __init__(self, *sinks: CrossSpectrum | NmrAccumulator) -> None:
        self.sinks = sinks

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        for sink in self.sinks:
            sink.add(a, b)


class _BlockGate:
    """Ana gecisin spektrumlarini ~1 s'lik bloklar halinde denetleyip iletir.

    Hizalama plani gecikmeyi birkac pencereden olcer; dosyanin geri kalaninin
    AYNI gecikmeyle hizali oldugu varsayilir. Duzenlenmis bir dosyada (ortadan
    kesilmis, sonu degistirilmis) bu varsayim bozulur ve hizasiz bolumler
    codec gurultusu diye olculur: gercek 21.7 dB, kesik dosyada 0.7-3.0 dB,
    cogu zaman uyarisiz (denetim D1).

    Her blokta mid capraz spektrumunun PHAT tepesi bulunur. Hizali bir blokta
    tepe sifir gecikmededir (cerceveler zaten hizali); hizasiz bir blokta
    rastgele bir yere duser. Korelasyon DEGIL tepe konumu kullaniliyor: codec
    gurultusu yuksek ama hizali bir blok (sessiz pasaj, dusuk bitrate) dusuk
    korelasyon verir ve atilirsa S/N oldugundan iyi gorunurdu; PHAT tepesinin
    YERI ise dusuk S/N'de de dogru kalir.

    Hizasiz bloklar olcume katilmaz ve sayilir; side ayni karari paylasir.

    Sinir bloklari: kesimin tam ustune dusen blok yari hizalidir ve PHAT tepesi
    yine sifirda cikar. Birkac saniyelik ilgisiz ses bile S/N'i cok bozar
    (olculen: son 13 s'si degistirilmis dosyada 2 s'lik bloklarla 12.05 yerine
    6.94 dB). Bu yuzden hizasiz her bolgenin iki yanindaki birer blok da
    atilir (tutulan kumenin bir blok asinmasi): blok i ancak i-1, i ve i+1
    hizaliysa olcume girer. Bedeli kesim basina ~2 s iyi ses.
    """

    def __init__(
        self,
        mid: _Fanout,
        side: _Fanout | None,
        *,
        fft_size: int,
        frames_per_block: int,
    ) -> None:
        self._mid, self._side = mid, side
        self._fft_size = fft_size
        self._block = frames_per_block
        self._mid_buf: list[tuple[np.ndarray, np.ndarray]] = []
        self._side_buf: list[tuple[np.ndarray, np.ndarray]] = []
        self.kept_frames = 0
        self.dropped_frames = 0
        # Bir blok gecikmeli karar: (mid, side, kendi karari, oncekinin karari)
        self._held: (
            tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None, bool, bool]
            | None
        ) = None
        self._last_decision = True
        self.mid = _GateInput(self, is_side=False)
        self.side = _GateInput(self, is_side=True) if side is not None else None

    @staticmethod
    def _count(buf: list[tuple[np.ndarray, np.ndarray]]) -> int:
        return sum(a.shape[0] for a, _ in buf)

    @staticmethod
    def _take(buf: list[tuple[np.ndarray, np.ndarray]], n: int) -> tuple[np.ndarray, np.ndarray]:
        a = np.concatenate([x for x, _ in buf])
        b = np.concatenate([y for _, y in buf])
        buf.clear()
        if a.shape[0] > n:
            buf.append((a[n:], b[n:]))
        return a[:n], b[:n]

    def push(self, a: np.ndarray, b: np.ndarray, *, is_side: bool) -> None:
        (self._side_buf if is_side else self._mid_buf).append((a, b))
        self._flush(self._block)

    def _flush(self, need: int) -> None:
        while self._count(self._mid_buf) >= need and (
            self._side is None or self._count(self._side_buf) >= need
        ):
            mid_block = self._take(self._mid_buf, need)
            side_block = self._take(self._side_buf, need) if self._side is not None else None
            decision = self._aligned(*mid_block)
            self._release(next_decision=decision)
            self._held = (mid_block, side_block, decision, self._last_decision)
            self._last_decision = decision

    def _release(self, *, next_decision: bool) -> None:
        """Bekleyen blogu, iki komsusunun karari da bilindiginde iletir ya da atar."""
        if self._held is None:
            return
        (mid_a, mid_b), side_block, own, previous = self._held
        self._held = None
        frames = mid_a.shape[0]
        if own and previous and next_decision:
            self._mid.add(mid_a, mid_b)
            if self._side is not None and side_block is not None:
                self._side.add(*side_block)
            self.kept_frames += frames
        else:
            self.dropped_frames += frames

    def finish(self) -> None:
        """Son eksik blogu da karara baglar."""
        remaining = self._count(self._mid_buf)
        if self._side is not None:
            remaining = min(remaining, self._count(self._side_buf))
        if remaining:
            self._flush(remaining)
        # Dosya sonu bir kesim degildir: son blok yalnizca kendi ve oncekinin
        # kararina gore degerlendirilir.
        self._release(next_decision=True)

    def _aligned(self, a: np.ndarray, b: np.ndarray) -> bool:
        cross = np.sum(np.conj(a) * b, axis=0)
        magnitude = np.abs(cross)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        power = float(np.sum(np.abs(a) ** 2)) * float(np.sum(np.abs(b) ** 2))
        if peak <= 0.0 or power <= 0.0:
            # Sessizlik ya da tek tarafli sessizlik (dropout): hizalama
            # yargilanamaz, blok tutulur -- dropout gercek bir farktir.
            return True
        whitened = cross / np.maximum(magnitude, 1e-12 * peak)
        correlation = np.fft.irfft(whitened, self._fft_size)
        index = int(np.argmax(correlation))
        lag = index if index <= self._fft_size // 2 else index - self._fft_size
        return abs(lag) <= _GATE_MAX_LAG


class _GateInput:
    """Gecislerin bekledigi `add(a, b)` arayuzunu kapiya baglar."""

    def __init__(self, gate: _BlockGate, *, is_side: bool) -> None:
        self._gate, self._is_side = gate, is_side

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        if a.shape[0]:
            self._gate.push(a, b, is_side=self._is_side)


class _Buffer:
    """Akan referansin kayan penceresi; mutlak ornek indeksiyle erisilir."""

    def __init__(self, blocks: Iterator[np.ndarray], stereo: bool) -> None:
        self._blocks = blocks
        self._stereo = stereo
        self.base = 0
        self.mid = np.zeros(0)
        self.side = np.zeros(0)
        self.eof = False

    @property
    def end(self) -> int:
        return self.base + self.mid.size

    def cover(self, start: int, stop: int) -> None:
        """[start, stop) araligini tampona getirir; `start` oncesini birakir."""
        self._drop_before(start)
        while self.end < stop and not self.eof:
            block = next(self._blocks, None)
            if block is None:
                self.eof = True
                break
            mid, side = _mid_side(block, tuple(range(block.shape[1])))
            if self.end + mid.size <= start:
                # Tamamen gerekli araligin oncesinde: tutmadan gec (uzun atlamalar).
                self.base += mid.size
                continue
            self.mid = np.concatenate([self.mid, mid])
            if self._stereo and side is not None:
                self.side = np.concatenate([self.side, side])
            self._drop_before(start)

    def _drop_before(self, start: int) -> None:
        cut = min(max(0, start - self.base), self.mid.size)
        if cut:
            self.mid = self.mid[cut:]
            if self._stereo:
                self.side = self.side[cut:]
            self.base += cut


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
    stage: StageFn = _noop,
) -> ComparisonResult:
    """Iki izi hizalar ve farklarini olcer. `stage` asama degisimlerini bildirir."""
    ref_stream, test_stream = reference.stream, test.stream
    rate = max(ref_stream.sample_rate, test_stream.sample_rate)
    ref_source, test_source = reference.source(rate), test.source(rate)
    notes: list[str] = []

    # -- hizalama ----------------------------------------------------------
    stage("align")
    with ThreadPoolExecutor(max_workers=2) as pool:
        env_futures = [
            pool.submit(envelope.build, ffmpeg, t.path, stream_index=t.stream_index, cancel=cancel)
            for t in (reference, test)
        ]
        ref_env, test_env = (f.result() for f in env_futures)

    read_reference = FFmpegWindowReader(ffmpeg, ref_source, resample=resample, cancel=cancel)
    read_test = FFmpegWindowReader(ffmpeg, test_source, resample=resample, cancel=cancel)
    alignment = plan.build(ref_env, test_env, read_reference, read_test, rate)
    if alignment.verdict not in ("aligned", "different_master"):
        return _not_measured(reference, test, alignment, rate, status="not_comparable", notes=notes)
    if alignment.verdict == "different_master":
        notes.append(
            Message(
                "compare.different_master",
                "different master: the difference is not a codec difference",
            )
        )

    # -- ana gecis -----------------------------------------------------------
    stage("measure")
    model = tracking.from_plan(alignment, rate)
    if model is None:
        notes.append(
            Message("compare.untracked", "clock drift could not be tracked: too few aligned points")
        )
        return _not_measured(reference, test, alignment, rate, status="not_measured", notes=notes)
    if model.slope:
        model = tracking.refine_model(
            model,
            read_reference,
            read_test,
            rate,
            _overlap(model, ref_env.duration_s * rate, test_env.duration_s * rate),
        )
        notes.append(
            Message(
                "compare.tracked",
                "clock drift tracked: {ppm:+.2f} ppm, largest deviation from the fit "
                "{residual:.3f} samples",
                ppm=model.slope * 1e6,
                residual=model.max_residual,
            )
        )
    fraction = alignment.delay_samples - round(alignment.delay_samples)
    measure = _tracked_pass if model.slope else _main_pass
    bins = fft_size // 2 + 1
    stereo = ref_stream.channels >= 2
    mid = CrossSpectrum(bins)
    side = CrossSpectrum(bins) if stereo else None
    # NMR icin kazanc plandan: test ~= gain * reference. Isaret polariteden.
    plan_gain = alignment.polarity * 10.0 ** (alignment.gain_db / 20.0)
    nmr = NmrAccumulator(
        rate,
        fft_size,
        gain=plan_gain,
        expected_frames=int(test_env.duration_s * rate) // (fft_size // 2),
    )
    gate = _BlockGate(
        _Fanout(mid, nmr),
        _Fanout(side) if side is not None else None,
        fft_size=fft_size,
        frames_per_block=max(8, round(_GATE_BLOCK_S * rate / (fft_size // 2))),
    )
    measure(
        ffmpeg,
        ref_source,
        test_source,
        model,
        alignment.channel_map,
        gate.mid,
        gate.side,
        fft_size=fft_size,
        resample=resample,
        cancel=cancel,
    )
    gate.finish()
    hop = fft_size // 2
    samples = gate.kept_frames * hop
    excluded_s = gate.dropped_frames * hop / rate
    total_frames = gate.kept_frames + gate.dropped_frames
    if total_frames and gate.kept_frames < _GATE_MIN_KEPT * total_frames:
        notes.append(
            Message(
                "compare.mostly_misaligned",
                "only {kept:.0f} s of {total:.0f} s line up with the reference at the planned "
                "delay: the files are not a continuous copy of each other",
                kept=gate.kept_frames * hop / rate,
                total=total_frames * hop / rate,
            )
        )
        return _not_measured(reference, test, alignment, rate, status="not_measured", notes=notes)
    if excluded_s > 0:
        notes.append(
            Message(
                "compare.excluded",
                "{excluded:.1f} s of {total:.1f} s did not line up with the reference (an edit, a "
                "cut or a different ending) and were left out of the measurement",
                excluded=excluded_s,
                total=total_frames * hop / rate,
            )
        )
    if mid.frames < 2:
        notes.append(Message("compare.short_overlap", "overlap too short to measure"))
        return _not_measured(reference, test, alignment, rate, status="not_measured", notes=notes)

    # -- bantlar + taban -----------------------------------------------------
    stage("floor")
    nyquist = min(ref_stream.sample_rate, test_stream.sample_rate) / 2.0
    # En ust %1 hicbir zaman raporlanmaz: kesirli gecikme orada tanimsiz
    # (olculen, bkz. `stft.phase_shift`: Nyquist bini tam kayip) ve bant
    # duyulabilir aralikta degil. Tek basina taban kurali bunu yakalamiyor:
    # 44.1/44.1 bir ciftte 22.00-22.05 kHz bandi taban 3 dB iken "olculebilir"
    # cikti. Hizlar farkliysa soxr kesimi de ayni sinirdir; ustunu tanim
    # geregi gecirmez.
    nyquist *= min(resample.cutoff, NYQUIST_FRACTION)
    layout = band_layout(rate, nyquist)
    broadband_hz = (BROADBAND_HZ[0], min(BROADBAND_HZ[1], nyquist))
    floors = calibration.measure_floor(
        ffmpeg,
        reference.path,
        stream_index=reference.stream_index,
        channels=ref_stream.channels,
        source_rate=ref_stream.sample_rate,
        other_rate=test_stream.sample_rate,
        fractional_delay=0.0 if model.slope else fraction,
        drift_slope=model.slope,
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
        excluded_s=excluded_s,
        nmr=nmr.summary(),
        notes=tuple(notes),
    )


def _overlap(model: DelayModel, ref_length: float, test_length: float) -> tuple[int, int]:
    """Test icinde, referansin karsiligi olan ornek araligi (kenarlardan %2 pay)."""
    lo = max(0.0, model.intercept)
    hi = min(test_length, (ref_length + model.intercept) / (1.0 - model.slope))
    pad = 0.02 * (hi - lo)
    return int(lo + pad), int(hi - pad)


def _excerpt_start(duration: float | None) -> float | None:
    """Taban kesiti icin baslangic: kaydin ortasina yakin, sessiz giris/cikistan uzak."""
    if duration is None or duration <= calibration.DEFAULT_EXCERPT_S * 1.5:
        return None
    return max(0.0, duration / 2.0 - calibration.DEFAULT_EXCERPT_S / 2.0)


def _main_pass(
    ffmpeg: Path,
    ref_source: Source,
    test_source: Source,
    model: DelayModel,
    channel_map: tuple[int, ...],
    mid: _Sink,
    side: _Sink | None,
    *,
    fft_size: int,
    resample: ResampleCfg,
    cancel: CancelToken | None,
) -> int:
    """Iki dosyayi bastan akitip capraz spektrumu biriktirir.

    Test tarafi sabit adimli STFT'dir. Her test cercevesi icin referans
    cercevesi, o cercevenin ortasindaki gecikmeye gore tampondan ayri ayri
    alinir: `test[T] = reference[T - d(T)]`. Tamsayi kisim cerceve baslangicini,
    +-0.5'lik kesir faz rampasini belirler. Gecikme sabitse bu, bir kere ornek
    atlayip sabit kaydirmakla ayni; kayma varsa her cerceve kendi aninda
    hizali olur ve ornek dusurme/tekrarlama (tik) hic olmaz.
    """
    hop = fft_size // 2
    window = hann(fft_size)
    freqs = np.arange(fft_size // 2 + 1) / fft_size
    stereo = ref_source.channels >= 2
    test_mid_stft, test_side_stft = StreamingStft(fft_size, hop), StreamingStft(fft_size, hop)
    offsets = np.arange(fft_size)
    frame_index = 0
    fed = 0

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

    with stream(ref_source) as ref_pcm, stream(test_source) as test_pcm:
        buffer = _Buffer(ref_pcm.blocks(_BLOCK_FRAMES), stereo)
        for block in test_pcm.blocks(_BLOCK_FRAMES):
            test_mid, test_side = _mid_side(block, channel_map)
            spectra_mid = test_mid_stft.push(test_mid)
            spectra_side = test_side_stft.push(test_side) if test_side is not None else None
            count = spectra_mid.shape[0]
            if count == 0:
                continue
            starts = (frame_index + np.arange(count)) * hop
            frame_index += count

            ref_start = starts - model.at(starts + fft_size / 2.0)
            whole = np.round(ref_start).astype(np.int64)
            # ref[whole + eps + n] gerekiyor (eps = ref_start - whole), yani
            # tampondan alinan cerceve -eps kadar GECIKTIRILIR.
            delay = whole - ref_start
            usable = whole >= 0
            if not usable.any():
                continue
            buffer.cover(int(whole[usable].min()), int(whole[usable].max()) + fft_size)
            usable &= (whole >= buffer.base) & (whole + fft_size <= buffer.end)
            if usable.any():
                fed += int(usable.sum())
                index = (whole[usable] - buffer.base)[:, None] + offsets
                ramp = np.exp(-2j * np.pi * np.outer(delay[usable], freqs))
                mid.add(np.fft.rfft(buffer.mid[index] * window, axis=1) * ramp, spectra_mid[usable])
                if side is not None and spectra_side is not None:
                    side.add(
                        np.fft.rfft(buffer.side[index] * window, axis=1) * ramp,
                        spectra_side[usable],
                    )
            if buffer.eof and whole[-1] + fft_size > buffer.end:
                break
    return fed * hop


def _tracked_pass(
    ffmpeg: Path,
    ref_source: Source,
    test_source: Source,
    model: DelayModel,
    channel_map: tuple[int, ...],
    mid: _Sink,
    side: _Sink | None,
    *,
    fft_size: int,
    resample: ResampleCfg,
    cancel: CancelToken | None,
) -> int:
    """Saat kaymasi varken: referans her ornek icin kendi konumundan orneklenir.

    Test ornegi T icin referans konumu `P = T - d(T)`; `warp.sample` bunu
    pencereli sinc ile uretir. Iki STFT'ye kilit adimda AYNI sayida ornek
    verilir, cerceveler birebir eslesir. Cerceve basina sabit gecikmenin
    (hizli yol) cerceve ici kaymasi burada yok.
    """
    hop = fft_size // 2
    stereo = ref_source.channels >= 2
    stfts = [StreamingStft(fft_size, hop) for _ in range(4)]
    ref_mid_stft, ref_side_stft, test_mid_stft, test_side_stft = stfts
    position = 0
    started = False
    fed = 0

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

    with stream(ref_source) as ref_pcm, stream(test_source) as test_pcm:
        buffer = _Buffer(ref_pcm.blocks(_BLOCK_FRAMES), stereo)
        for block in test_pcm.blocks(_BLOCK_FRAMES):
            test_mid, test_side = _mid_side(block, channel_map)
            times = position + np.arange(test_mid.size, dtype=np.float64)
            position += test_mid.size
            wanted = times - model.at(times)

            keep = np.ones(wanted.size, dtype=bool)
            if not started:
                keep = wanted >= warp.HALF_TAPS
                if not keep.any():
                    continue
                started = True
            first = int(np.floor(wanted[keep].min())) - warp.HALF_TAPS + 1
            last = int(np.floor(wanted[keep].max())) + warp.HALF_TAPS + 1
            buffer.cover(first, last)
            done = False
            if buffer.end < last:
                keep &= np.floor(wanted) + warp.HALF_TAPS < buffer.end
                done = True
            if keep.any():
                ref_mid = warp.sample(buffer.mid, wanted[keep], base=buffer.base)
                mid.add(ref_mid_stft.push(ref_mid), test_mid_stft.push(test_mid[keep]))
                if side is not None and test_side is not None:
                    ref_side = warp.sample(buffer.side, wanted[keep], base=buffer.base)
                    side.add(ref_side_stft.push(ref_side), test_side_stft.push(test_side[keep]))
                fed += int(keep.sum())
            if done:
                break
    return fed
