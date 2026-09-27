"""Capraz spektrum biriktirme testleri.

Bilinen bir EQ ve bilinen seviyede bagimsiz gurultu enjekte edilir; ayrismanin
ikisini dogru bilesenlere koymasi beklenir.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.dsp.accum import CrossSpectrum, hz_to_bin
from app.dsp.stft import StreamingStft

SIZE = 1024


def tilt(x: np.ndarray, top_db: float = -12.0) -> np.ndarray:
    """Bilinen EQ: frekansla dogrusal egim, Nyquist'te `top_db`."""
    spectrum = np.fft.rfft(x)
    spectrum *= 10 ** (top_db * np.linspace(0, 1, spectrum.size) / 20)
    return np.fft.irfft(spectrum, x.size)


def scenario(n: int, snr_db: float, seed: int, hop: int = SIZE // 2) -> tuple[CrossSpectrum, float]:
    """EQ'lu kopya + bagimsiz gurultu. (biriktirici, gercek gurultu gucu) dondurur."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    clean = tilt(a)
    noise = rng.standard_normal(n)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2)) * 10 ** (-snr_db / 20)
    spectra_a = StreamingStft(SIZE, hop).push(a)
    spectra_b = StreamingStft(SIZE, hop).push(clean + noise)
    spectra_n = StreamingStft(SIZE, hop).push(noise)
    cs = CrossSpectrum(spectra_a.shape[1])
    cs.add(spectra_a, spectra_b)
    return cs, float(np.sum(np.abs(spectra_n[:, 1:512]) ** 2))


@pytest.mark.parametrize("snr_db", [40.0, 20.0, 6.0])
def test_codec_snr_is_recovered_despite_eq(snr_db: float) -> None:
    """Olculen: 39.89 / 19.99 / 5.99 dB. EQ, S/N'e karismamali."""
    cs, true_noise = scenario(1_000_000, snr_db, seed=0)
    stats = cs.band(1, 512)
    assert stats.snr_db == pytest.approx(snr_db, abs=0.2)
    assert stats.incoherent_power == pytest.approx(true_noise, rel=0.03)


def test_eq_lands_in_the_linear_component_not_the_noise() -> None:
    """Ayrismanin varlik sebebi: 12 dB'lik egim duz S/N'i 8.8 dB'e cekiyor.

    Duz (skaler kazancli) fark EQ'yu gurultu sayar; inkoherent S/N saymaz.
    """
    cs, _ = scenario(1_000_000, 40.0, seed=1)
    stats = cs.band(1, 512)
    assert stats.scalar_snr_db < 10.0
    assert stats.snr_db > 39.0
    assert stats.linear_power > 100 * stats.incoherent_power


def test_identical_signals_have_infinite_snr_and_no_linear_part() -> None:
    x = np.random.default_rng(2).standard_normal(100_000)
    spectra = StreamingStft(SIZE).push(x)
    cs = CrossSpectrum(spectra.shape[1])
    cs.add(spectra, spectra)
    stats = cs.band(1, 512)
    assert cs.gain() == pytest.approx(1.0)
    assert stats.incoherent_power == pytest.approx(0.0, abs=1e-9 * stats.test_power)
    assert stats.snr_db > 100.0
    assert stats.coherence == pytest.approx(1.0)


def test_inverted_polarity_gives_negative_gain() -> None:
    """Kazanc ISARETLI; abs() alinirsa ters polarite "cok bozuk" gorunur."""
    x = np.random.default_rng(3).standard_normal(50_000)
    spectra = StreamingStft(SIZE).push(x)
    cs = CrossSpectrum(spectra.shape[1])
    cs.add(spectra, -0.5 * spectra)
    assert cs.gain() == pytest.approx(-0.5)
    assert cs.band(1, 512, gain=cs.gain()).scalar_residual_power == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("hop", "frames", "floor"),
    [
        (SIZE, 2, 0.97),
        (SIZE, 8, 0.97),
        (SIZE // 2, 2, 0.95),
        (SIZE // 2, 8, 0.97),
    ],
)
def test_small_sample_bias_is_corrected(hop: int, frames: int, floor: float) -> None:
    """Az cercevede inkoherent guc oldugundan kucuk cikar; k/(k-1) duzeltir.

    Olculen (200 deneme ortalamasi, olculen/gercek): duzeltmesiz k=2'de 0.50;
    duzeltilmis ortusmesiz 1.001, %50 ortusmeli 0.981 (k=2) .. 0.993 (k=8).
    """
    n = SIZE + (frames - 1) * hop
    ratios = []
    for seed in range(200):
        cs, true_noise = scenario(n, 20.0, seed, hop)
        assert cs.frames == frames
        ratios.append(cs.band(1, 512).incoherent_power / true_noise)
    mean = float(np.mean(ratios))
    assert floor < mean < 1.03


def test_single_frame_does_not_claim_infinite_snr() -> None:
    """Tek cercevede her bin mukemmel "aciklanir"; bu sonsuz S/N DEGIL, bilinmez."""
    x = np.random.default_rng(4).standard_normal(SIZE)
    y = x + np.random.default_rng(5).standard_normal(SIZE)
    a = StreamingStft(SIZE).push(x)
    cs = CrossSpectrum(a.shape[1])
    cs.add(a, StreamingStft(SIZE).push(y))
    stats = cs.band(1, 512)
    assert math.isnan(stats.incoherent_power)
    assert math.isnan(stats.snr_db)


def test_merge_equals_single_accumulation() -> None:
    rng = np.random.default_rng(6)
    a = rng.standard_normal((40, 513)) + 1j * rng.standard_normal((40, 513))
    b = rng.standard_normal((40, 513)) + 1j * rng.standard_normal((40, 513))
    whole = CrossSpectrum(513)
    whole.add(a, b)
    first, second = CrossSpectrum(513), CrossSpectrum(513)
    first.add(a[:15], b[:15])
    second.add(a[15:], b[15:])
    first.merge(second)
    assert first.frames == whole.frames
    assert np.allclose(first.sab, whole.sab)
    assert np.allclose(first.saa, whole.saa)


def test_rejects_malformed_input() -> None:
    cs = CrossSpectrum(8)
    with pytest.raises(ValueError, match="sekilleri farkli"):
        cs.add(np.zeros((2, 8)), np.zeros((3, 8)))
    with pytest.raises(ValueError, match="beklenen"):
        cs.add(np.zeros((2, 9)), np.zeros((2, 9)))
    with pytest.raises(ValueError, match="gecersiz bant"):
        cs.band(5, 5)
    with pytest.raises(ValueError, match="pozitif"):
        CrossSpectrum(0)


def test_hz_to_bin() -> None:
    assert hz_to_bin(0.0, 48000, 4096) == 0
    assert hz_to_bin(24000.0, 48000, 4096) == 2048
    assert hz_to_bin(1000.0, 48000, 4096) == round(1000 * 4096 / 48000)
    assert hz_to_bin(99999.0, 48000, 4096) == 2048
