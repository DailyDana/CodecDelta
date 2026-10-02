"""ABX icin capraz gecisli karistirici. Saf numpy; Qt IMPORT ETMEZ.

Neden capraz gecis ZORUNLU: ham kesmede tikin genligi |A(t) - B(t)| ile
orantilidir, yani tik CEVABIN KENDISINI sizdirir. Gecis 8 ms ve DOGRUSAL:
A ve B neredeyse tam korelasyonlu oldugu icin esit guc gecisi ortada +3 dB'lik
bir tumsek yapar ve o da duyulur.

Kurallar:
- Her kaynagin kazanci hedefine `fade` uzunlugunda dogrusal ilerler. Gecis
  yarida yeni bir secimle kesilirse rampa o anki kazanctan devam eder; sicrama
  olmaz.
- A -> A dahil HER secim ayni makineden gecer (kisa devre yok): zamanlama
  farki da bir ipucu olurdu.
- Gecis oynatma konumunu degistirmez; iki kaynak ayni ani calar.
- Kesit dongude calar. Kuyruk ile bas ayni 8 ms'lik dogrusal gecisle
  birlestirilir; dongu noktasinda tik yok.
- Baslatma ve durdurma da sessizlikten / sessizlige ayni rampayla.
"""

from __future__ import annotations

import threading
from typing import Literal

import numpy as np

Source = Literal["A", "B"]
DEFAULT_FADE_MS = 8.0


class CrossfadeMixer:
    def __init__(
        self, a: np.ndarray, b: np.ndarray, rate: int, *, fade_ms: float = DEFAULT_FADE_MS
    ) -> None:
        if a.shape != b.shape or a.ndim != 2:
            raise ValueError("A ve B ayni (cerceve, kanal) seklinde olmali")
        self.rate = rate
        self.fade = max(1, round(fade_ms * rate / 1000.0))
        if a.shape[0] < 4 * self.fade:
            raise ValueError("kesit capraz gecis icin cok kisa")
        self._a = a.astype(np.float32, copy=False)
        self._b = b.astype(np.float32, copy=False)
        self.channels = a.shape[1]
        # Dongu boyu: kuyrugun son `fade` cercevesi basla birlestirilir.
        self._period = a.shape[0] - self.fade
        self._q = self.fade  # ilk calma basin gecis bolgesinden SONRA baslar
        # Kazanclar: [A, B, ana]. Her biri hedefine dogrusal ilerler.
        self._gain = np.array([1.0, 0.0, 0.0])
        self._target = self._gain.copy()
        self._step = np.zeros(3)
        self._lock = threading.Lock()
        self.source: Source = "A"

    # -- kontrol -----------------------------------------------------------

    def _ramp_to(self, target: np.ndarray) -> None:
        self._target = target.astype(float)
        self._step = (self._target - self._gain) / self.fade

    def select(self, source: Source) -> None:
        """Kaynagi degistirir (ayni kaynak dahil) -- her zaman rampayla."""
        with self._lock:
            self.source = source
            target = self._target.copy()
            target[0], target[1] = (1.0, 0.0) if source == "A" else (0.0, 1.0)
            self._ramp_to(target)

    def start(self) -> None:
        with self._lock:
            target = self._target.copy()
            target[2] = 1.0
            self._ramp_to(target)

    def stop(self) -> None:
        with self._lock:
            target = self._target.copy()
            target[2] = 0.0
            self._ramp_to(target)

    def rewind(self) -> None:
        """Kesitin basina doner (yalnizca sessizken anlamli)."""
        with self._lock:
            self._q = self.fade

    @property
    def silent(self) -> bool:
        """Durduruldu ve cikis sessizlige indi."""
        return bool(self._gain[2] == 0.0 and self._target[2] == 0.0)

    @property
    def position_s(self) -> float:
        return float(self._q) / self.rate

    # -- uretim ------------------------------------------------------------

    def _loop(self, signal: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Dongu ekseni q (0..period) icin sinyal; bas bolgesi kuyrukla birlesir."""
        out: np.ndarray = signal[q]
        head = q < self.fade
        if head.any():
            w = ((q[head] + 0.5) / self.fade)[:, None]
            out = out.copy()
            out[head] = (1.0 - w) * signal[q[head] + self._period] + w * signal[q[head]]
        return out

    def render(self, frames: int) -> np.ndarray:
        """Sonraki `frames` cerceveyi (cerceve, kanal) float32 olarak uretir."""
        with self._lock:
            idx = np.arange(frames)
            gains = np.empty((frames, 3))
            for k in range(3):
                remaining = (
                    int(np.ceil(abs(self._target[k] - self._gain[k]) / abs(self._step[k])))
                    if self._step[k] != 0.0
                    else 0
                )
                ramp = self._gain[k] + self._step[k] * (idx + 1)
                gains[:, k] = np.where(idx < remaining, ramp, self._target[k])
            self._gain = gains[-1].copy() if frames else self._gain
            done = self._gain == self._target
            self._step[done] = 0.0
            q = (self._q + idx) % self._period
            self._q = int((self._q + frames) % self._period)
        a = self._loop(self._a, q)
        b = self._loop(self._b, q)
        out = (gains[:, 0:1] * a + gains[:, 1:2] * b) * gains[:, 2:3]
        result: np.ndarray = out.astype(np.float32)
        return result
