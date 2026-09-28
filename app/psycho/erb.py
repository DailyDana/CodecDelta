"""Isitsel frekans olcekleri ve STFT binlerinin bantlara bolunmesi.

Iki olcek, iki ayri is icin:

- **ERB** (Glasberg & Moore 1990): bantlarin genisligini belirler. Kulagin
  frekans cozunurlugu dusuk frekansta ince, yuksekte kabadir; 1 ERB genisligindeki
  bantlar 20 Hz-20 kHz arasini ~40 parcaya boler ve her parca kulak icin esit
  "genisliktedir".
- **Bark** (Zwicker): maskelemenin yayilma mesafesi icin. Yayilma fonksiyonu
  literaturde Bark cinsinden verilir (Schroeder 1979); onu ERB'ye cevirmek
  yerine bant merkezlerinin Bark degeri kullaniliyor. Boylece fonksiyon
  kaynaktaki haliyle kaliyor.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


def erb_bandwidth(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    """Glasberg-Moore: ERB(f) = 24.7 * (4.37 f/kHz + 1) Hz."""
    return 24.7 * (4.37 * np.asarray(frequency_hz) / 1000.0 + 1.0)


def hz_to_erb(frequency_hz: np.ndarray | float) -> np.ndarray:
    """ERB-sayisi olcegi: E(f) = 21.4 log10(4.37 f/kHz + 1)."""
    result: np.ndarray = 21.4 * np.log10(
        4.37 * np.asarray(frequency_hz, dtype=np.float64) / 1000.0 + 1.0
    )
    return result


def erb_to_hz(erb: np.ndarray | float) -> np.ndarray:
    result: np.ndarray = (10.0 ** (np.asarray(erb, dtype=np.float64) / 21.4) - 1.0) * 1000.0 / 4.37
    return result


def hz_to_bark(frequency_hz: np.ndarray | float) -> np.ndarray:
    """Zwicker: z = 13 atan(0.00076 f) + 3.5 atan((f/7500)^2)."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    result: np.ndarray = 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
    return result


@dataclass(frozen=True)
class BandLayout:
    """STFT binlerinin isitsel bantlara eslenmesi."""

    sample_rate: int
    fft_size: int
    # Her bandin [alt, ust) bin indeksi.
    edges: np.ndarray
    centre_hz: np.ndarray
    centre_bark: np.ndarray

    @property
    def count(self) -> int:
        return int(self.centre_hz.size)

    def energies(self, power: np.ndarray) -> np.ndarray:
        """(cerceve, bin) guc spektrumunu (cerceve, bant) enerjisine indirger."""
        cumulative = np.concatenate(
            [np.zeros((power.shape[0], 1)), np.cumsum(power, axis=1)], axis=1
        )
        result: np.ndarray = cumulative[:, self.edges[:, 1]] - cumulative[:, self.edges[:, 0]]
        return result


def erb_layout(
    sample_rate: int, fft_size: int, *, lo_hz: float = 20.0, hi_hz: float = 20000.0
) -> BandLayout:
    """1 ERB genisliginde bantlar; en az bir bin iceren bantlar tutulur.

    Dusuk frekansta 1 ERB (20 Hz'de ~27 Hz) tek bir binden dar olabilir
    (48 kHz / 4096 = 11.7 Hz): bos kalan bantlar atilir, komsu bant genisler.
    """
    if sample_rate <= 0 or fft_size <= 0:
        raise ValueError("sample_rate ve fft_size pozitif olmali")
    hi_hz = min(hi_hz, sample_rate / 2.0)
    bin_hz = sample_rate / fft_size
    erb_edges = np.arange(float(hz_to_erb(lo_hz)), float(hz_to_erb(hi_hz)) + 1e-9, 1.0)
    hz_edges = np.append(erb_to_hz(erb_edges), hi_hz)
    bin_edges = np.round(hz_edges / bin_hz).astype(np.int64)
    bin_edges = np.clip(bin_edges, 0, fft_size // 2 + 1)

    edges = []
    for lo, hi in pairwise(bin_edges):
        if hi > lo:
            edges.append((int(lo), int(hi)))
    if not edges:
        raise ValueError("hicbir bant bin icermiyor")
    edge_array = np.array(edges, dtype=np.int64)
    centre = (edge_array[:, 0] + edge_array[:, 1] - 1) / 2.0 * bin_hz
    return BandLayout(
        sample_rate=sample_rate,
        fft_size=fft_size,
        edges=edge_array,
        centre_hz=centre,
        centre_bark=hz_to_bark(centre),
    )
