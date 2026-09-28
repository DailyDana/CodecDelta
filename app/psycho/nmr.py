"""Gurultu/maske orani (NMR) -- fark duyulabilir mi sorusunun ilk katmani.

Her cercevede referans spektrumundan maskeleme esigi, farktan bant basina
gurultu enerjisi hesaplanir:

    NMR_b = 10 log10( gurultu_b / esik_b )

`NMR = 0 dB`in fiziksel anlami acik: gurultu tam maskeleme esiginde. Bu capa
saglamdir. Yuzde esikleri (kac cercevede sifiri asmali) literatur dayanakli
DEGILDIR ve hukum icin kullanilmaz; hukmu capa merdiveni verir. NMR burada
merdivenin ikinci ekseni ve rapordaki NMR spektrogrami icin.

Cerceve basina iki istatistik tutulur:

- **toplam NMR** (MPEG): bant oranlarinin ortalamasi, `10 log10(mean(N_b/T_b))`.
  Literaturun tanimladigi buyukluk; mansette bu.
- **tepe NMR**: en kotu bant. Daha muhafazakar ve dar bantlarda tek-cerceve
  gurultu kestiriminin varyansiyla yukari yanli; spektrogram bunu gosterir.

Olculen (gercek FLAC/Opus ~141 kbps): toplam p50 -9.2 dB, tepe p50 -2.8 dB.

Gurultu, `test - kazanc * referans` farkidir; kazanc plandan gelir ve cerceve
basina degismez. Yani seviye/EQ farki da gurultuya girer -- bilincli: kulak
icin EQ farki da farktir. Mansettaki S/N ise dogrusal kismi ayiklar; ikisi
farkli soruya cevap verir.

Biriktirici `dsp.accum.CrossSpectrum` ile ayni `add(ref, test)` sozlesmesini
saglar ve ana geciste onunla yan yana calisir; ikinci gecis yoktur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.dsp.stft import hann
from app.psycho import mask
from app.psycho.erb import BandLayout, erb_layout
from app.psycho.mask import MaskingModel

# Cerceve-tepe NMR histogrami: -100..+60 dB, 0.5 dB adim.
_HIST_LO_DB = -100.0
_HIST_HI_DB = 60.0
_HIST_STEP_DB = 0.5
# Spektrogram icin zaman ekseninde en fazla bu kadar sutun tutulur.
_GRID_COLUMNS = 1000
# Hem referans hem gurultu bunun altindaysa cerceve degerlendirilmez: SAYISAL
# sessizlik, yargilanacak bir sey yok. Esik cok dusuk tutuluyor (-200 dB):
# -110 dBFS'lik bir gurultu ATH'nin altindadir ama "duyulmaz" hukmu icin
# degerlendirilmesi gerekir; ilk surumdeki -100 dB kapisi onu da atliyordu.
_SILENCE_POWER = 1e-20


def power_scale(fft_size: int) -> float:
    """|X|^2'yi, tam olcekli sinusun BANT ENERJISI 1.0 olacak sekilde olcekler.

    Tek tarafli spektrumda sinusun gucu `N * sum(w^2) / 4`e dagilir (ana lob
    boyunca). Tepe binine gore olcekleme +1.76 dB fazla veriyordu (olculdu);
    ATH kalibrasyonu ("tam olcek = 96 dB SPL") bant enerjisine dayandigi icin
    fark dogrudan esige giriyordu.
    """
    window = hann(fft_size)
    return 4.0 / (fft_size * float(np.sum(window * window)))


@dataclass(frozen=True)
class NmrSummary:
    frames: int
    # MPEG toplam NMR yuzdelikleri (dB): bant oranlarinin ortalamasi.
    p50_db: float
    p95_db: float
    max_db: float
    # Toplam NMR'i 0 dB'i asan cerceve orani.
    fraction_above_0: float
    # En kotu bant (cerceve-tepe) icin ayni istatistikler.
    peak_p50_db: float
    peak_p95_db: float
    peak_max_db: float
    peak_fraction_above_0: float
    # Spektrogram: (bant, zaman sutunu), her hucre o araliktaki en yuksek NMR (dB).
    grid_db: np.ndarray
    centre_hz: np.ndarray
    seconds_per_column: float

    @property
    def audible_fraction_label(self) -> str:
        return f"{100.0 * self.fraction_above_0:.1f}% of frames above the masking threshold"


class _Histogram:
    """dB degerlerinin sabit adimli histogrami; yuzdelik, tepe ve 0 dB ustu orani."""

    def __init__(self, edges: np.ndarray) -> None:
        self._edges = edges
        self._counts = np.zeros(edges.size - 1, dtype=np.int64)
        self._above = 0
        self._max = -math.inf
        self.total = 0

    def add(self, values_db: np.ndarray) -> None:
        self.total += values_db.size
        self._above += int((values_db > 0.0).sum())
        self._max = max(self._max, float(values_db.max()))
        clipped = np.clip(values_db, _HIST_LO_DB, _HIST_HI_DB - 1e-9)
        self._counts += np.histogram(clipped, bins=self._edges)[0]

    def percentile(self, q: float) -> float:
        if self.total == 0:
            return math.nan
        cumulative = np.cumsum(self._counts)
        index = int(np.searchsorted(cumulative, q * self.total, side="left"))
        index = min(index, self._counts.size - 1)
        return float(self._edges[index] + _HIST_STEP_DB / 2.0)

    @property
    def max(self) -> float:
        return self._max if self.total else math.nan

    @property
    def fraction_above_0(self) -> float:
        return (self._above / self.total) if self.total else math.nan


class NmrAccumulator:
    """Ana geciste cerceve cerceve NMR biriktirir. Sozlesme: `add(ref, test)`."""

    def __init__(
        self,
        sample_rate: int,
        fft_size: int,
        *,
        gain: float,
        expected_frames: int | None = None,
        model: MaskingModel | None = None,
    ) -> None:
        self.layout: BandLayout = erb_layout(sample_rate, fft_size)
        self.model = model if model is not None else mask.build(self.layout)
        self.gain = gain
        self.scale = power_scale(fft_size)
        self.hop = fft_size // 2
        self.sample_rate = sample_rate
        self.frames = 0
        self.judged = 0
        self._edges = np.arange(_HIST_LO_DB, _HIST_HI_DB + _HIST_STEP_DB, _HIST_STEP_DB)
        self._total = _Histogram(self._edges)
        self._peak = _Histogram(self._edges)
        # Zaman izgarasi: cerceve basina sutun, beklenen uzunluga gore.
        frames_per_column = 1
        if expected_frames is not None and expected_frames > _GRID_COLUMNS:
            frames_per_column = math.ceil(expected_frames / _GRID_COLUMNS)
        self._frames_per_column = frames_per_column
        self._grid: list[np.ndarray] = []
        self._column = np.full(self.layout.count, -np.inf)
        self._column_fill = 0

    def add(self, ref: np.ndarray, test: np.ndarray) -> None:
        if ref.shape != test.shape:
            raise ValueError(f"spektrum sekilleri farkli: {ref.shape} vs {test.shape}")
        if ref.shape[0] == 0:
            return
        ref_power = (ref.real**2 + ref.imag**2) * self.scale
        residual = test - self.gain * ref
        noise_power = (residual.real**2 + residual.imag**2) * self.scale

        thresholds = mask.threshold(self.model, ref_power)
        ratio = np.maximum(self.layout.energies(noise_power), 1e-30) / thresholds
        nmr_db = 10.0 * np.log10(ratio)

        judged = (ref_power.sum(axis=1) > _SILENCE_POWER) | (
            noise_power.sum(axis=1) > _SILENCE_POWER
        )
        self.frames += ref.shape[0]
        self._push_grid(nmr_db)
        if not judged.any():
            return
        self.judged += int(judged.sum())
        self._total.add(10.0 * np.log10(ratio[judged].mean(axis=1)))
        self._peak.add(nmr_db[judged].max(axis=1))

    def _push_grid(self, nmr_db: np.ndarray) -> None:
        for row in nmr_db:
            self._column = np.maximum(self._column, row)
            self._column_fill += 1
            if self._column_fill >= self._frames_per_column:
                self._grid.append(self._column)
                self._column = np.full(self.layout.count, -np.inf)
                self._column_fill = 0

    def summary(self) -> NmrSummary:
        columns = list(self._grid)
        if self._column_fill:
            columns.append(self._column)
        grid = np.stack(columns, axis=1) if columns else np.zeros((self.layout.count, 0))
        return NmrSummary(
            frames=self.judged,
            p50_db=self._total.percentile(0.50),
            p95_db=self._total.percentile(0.95),
            max_db=self._total.max,
            fraction_above_0=self._total.fraction_above_0,
            peak_p50_db=self._peak.percentile(0.50),
            peak_p95_db=self._peak.percentile(0.95),
            peak_max_db=self._peak.max,
            peak_fraction_above_0=self._peak.fraction_above_0,
            grid_db=grid.astype(np.float32),
            centre_hz=self.layout.centre_hz,
            seconds_per_column=self._frames_per_column * self.hop / self.sample_rate,
        )
