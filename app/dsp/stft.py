"""Akan STFT -- bloklar halinde gelen sinyalden kesintisiz cerceveler.

ffmpeg sesi sabit boyutlu bloklarla verir ama blok sinirlari analiz
cercevelerine denk gelmez. Bu modul arta kalan ornekleri tasiyarak, sinyal
parca parca gelse de TUM sinyal tek seferde islenmis gibi ayni cerceveleri
uretir. Sozlesme testi bunu rastgele blok boyutlariyla `array_equal` olarak
sinar.

Pencere periyodik Hann, varsayilan adim yarim pencere. Bu cift COLA kosulunu
saglar (pencereler toplami sabit); fark sinyalini disa aktarirken yeniden
sentez icin gerekli. Capraz spektrum oranlarinda (kazanc, koherans, S/N)
pencere olcegi sadelesir.
"""

from __future__ import annotations

import numpy as np

from app.dsp.transforms import ensure_signal


def hann(size: int) -> np.ndarray:
    """Periyodik Hann penceresi (simetrik degil; COLA icin periyodik olmali)."""
    if size <= 0:
        raise ValueError("pencere boyu pozitif olmali")
    window: np.ndarray = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)
    return window


def phase_shift(spectra: np.ndarray, delay: float, size: int) -> np.ndarray:
    """Cercevelerin spektrumunu `delay` ornek geciktirir: B(f) * exp(-j2*pi*f*delay).

    Kesirli gecikmeyi zaman alaninda interpolasyon yerine spektrumda uygular;
    blok-yerel gecikme takibinin mekanizmasi budur.

    YAKLASIKTIR ve hatasi bir olcum TABANI olusturur. Iki kaynak, OLCULEN
    (4096'lik cerceve, gercek kesirli kaydirilmis sinyalin STFT'sine gore):

    - Pencere kaymaz, yalnizca icerik kayar. Nyquist'in %99'una kadar DUZ:
      0.25 ornekte -73 dB, 0.5 ornekte -67 dB.
    - Nyquist'e yakin binlerde kesirli kayma tanimsiz (gercek bir sinyalin
      Nyquist bileseni gercek olmali). Bozulma yalnizca en ust %1'de (son ~20
      bin; 44.1 kHz'de 21.83 kHz ustu): orada -20 dB, son 4 binde -13 dB,
      Nyquist bininin kendisinde +2.9 dB (tam kayip). Beyaz gurultude toplam
      hatayi bu birkac bin belirliyor (0.5'te -29 dB); gercek ses orada enerji
      tasimaz ve o bant yeniden orneklemenin kendi tabani yuzunden zaten
      "olculemez" isaretlenir.

    Sonuc: kesirli gecikme telafi edildiginde ~64 dB ustundeki S/N olculemez.
    Kalibrasyon referansi ayni yoldan gecirdigi icin bu taban raporda gorunur.
    """
    if delay == 0.0:
        return spectra
    freqs = np.arange(spectra.shape[-1]) / size
    ramp = np.exp(-2j * np.pi * freqs * delay)
    shifted: np.ndarray = spectra * ramp
    return shifted


class StreamingStft:
    """Tek kanalli akan STFT.

    `push` her cagrida o ana kadar tamamlanan cercevelerin spektrumlarini
    (cerceve, bin) seklinde dondurur. Cerceve `t`, sinyalin `t * hop`
    orneginden baslar. Sonda tamamlanmayan cerceve uretilmez: kismi bir
    cerceveyi sifirla doldurmak spektruma sahte bir kenar ekler.
    """

    def __init__(self, size: int = 4096, hop: int | None = None) -> None:
        if size <= 0:
            raise ValueError("pencere boyu pozitif olmali")
        self.size = size
        self.hop = hop if hop is not None else size // 2
        if not 0 < self.hop <= size:
            raise ValueError("hop 1 ile pencere boyu arasinda olmali")
        self.window = hann(size)
        self._pending = np.zeros(0, dtype=np.float64)
        self.frames_emitted = 0

    @property
    def bins(self) -> int:
        return self.size // 2 + 1

    def push(self, block: np.ndarray) -> np.ndarray:
        ensure_signal(block, "block")
        data = np.concatenate([self._pending, block.astype(np.float64, copy=False)])
        if data.size < self.size:
            self._pending = data
            return np.zeros((0, self.bins), dtype=np.complex128)

        count = 1 + (data.size - self.size) // self.hop
        frames = np.lib.stride_tricks.sliding_window_view(data, self.size)[:: self.hop][:count]
        spectra: np.ndarray = np.fft.rfft(frames * self.window, axis=1)
        # Bir sonraki cercevenin baslangicindan itibaren tut.
        self._pending = data[count * self.hop :].copy()
        self.frames_emitted += count
        return spectra
