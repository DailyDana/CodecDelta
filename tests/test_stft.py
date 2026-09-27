"""Akan STFT testleri."""

from __future__ import annotations

import numpy as np
import pytest

from app.dsp.stft import StreamingStft, hann, phase_shift
from app.dsp.transforms import fractional_shift


def _whole(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Referans: tum sinyali tek seferde, dogrudan cercevele."""
    window = hann(size)
    starts = range(0, x.size - size + 1, hop)
    return np.array([np.fft.rfft(x[s : s + size] * window) for s in starts])


@pytest.mark.parametrize("seed", range(5))
def test_arbitrary_blocks_give_the_same_frames_as_one_pass(seed: int) -> None:
    """Sozlesme: blok sinirlari cikti uzerinde HICBIR iz birakmamali."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(50_000)
    stft = StreamingStft(1024, 256)
    pieces = []
    position = 0
    while position < x.size:
        step = int(rng.integers(1, 5000))
        pieces.append(stft.push(x[position : position + step]))
        position += step
    streamed = np.concatenate(pieces)
    expected = _whole(x, 1024, 256)
    assert streamed.shape == expected.shape
    assert np.array_equal(streamed, expected)
    assert stft.frames_emitted == expected.shape[0]


def test_short_pushes_emit_nothing_until_a_frame_completes() -> None:
    stft = StreamingStft(1024)
    assert stft.push(np.ones(1000)).shape == (0, 513)
    assert stft.push(np.ones(24)).shape == (1, 513)


def test_hann_with_half_hop_is_cola() -> None:
    """Yarim adimli periyodik Hann ust uste toplandiginda sabit: yeniden sentez icin sart."""
    size = 1024
    window = hann(size)
    total = np.zeros(size * 8)
    for start in range(0, total.size - size + 1, size // 2):
        total[start : start + size] += window
    core = total[size:-size]
    assert np.allclose(core, 1.0)


def test_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="pozitif"):
        StreamingStft(0)
    with pytest.raises(ValueError, match="hop"):
        StreamingStft(1024, 2048)
    with pytest.raises(ValueError, match="tek boyutlu"):
        StreamingStft(1024).push(np.zeros((10, 2)))


def _band_error_db(delay: float, lo: float, hi: float) -> float:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300_000)
    size = 4096
    a = StreamingStft(size).push(x)
    b = StreamingStft(size).push(fractional_shift(x, delay))
    shifted = phase_shift(a, delay, size)
    band = slice(int(lo * size / 2), int(hi * size / 2))
    err = np.sum(np.abs(shifted[2:-2, band] - b[2:-2, band]) ** 2)
    return float(10 * np.log10(err / np.sum(np.abs(b[2:-2, band]) ** 2)))


def test_phase_shift_floor_below_99_percent_of_nyquist() -> None:
    """Olculen taban: 0.5 ornekte -67 dB, 0.25'te -73 dB (bkz. docstring).

    Bu, kesirli gecikme telafi edildiginde olculebilecek en yuksek S/N'yi
    belirler; kotulesirse sessizce codec gurultusu gibi gorunur.
    """
    assert _band_error_db(0.5, 0.0, 0.99) < -65.0
    assert _band_error_db(0.25, 0.0, 0.99) < -71.0


def test_phase_shift_breaks_down_only_near_nyquist() -> None:
    """Ust %1: kesirli kayma tanimsiz. Bilinen sinir, test onu belgeliyor.

    Siniri kaydiran bir degisiklik (daha iyi ya da daha kotu) docstring'deki
    olculen tabloyu gecersiz kilar; bu test o durumda dusmeli.
    """
    assert _band_error_db(0.5, 0.99, 1.0) > -25.0
    assert _band_error_db(0.5, 0.95, 0.99) < -65.0


def test_zero_shift_is_identity() -> None:
    spectra = np.ones((3, 5), dtype=np.complex128)
    assert phase_shift(spectra, 0.0, 8) is spectra
