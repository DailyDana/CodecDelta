"""Zamanla degisen kesirli gecikme -- surekli yeniden ornekleme.

Saat kaymasi olan iki kayitta gecikme zamanla dogrusal degisir. Cerceve basina
sabit bir gecikme yetmez: 4096'lik bir cercevede gecikme 50 ppm'de 0.2, 200
ppm'de 0.8 ornek degisir ve cercevenin kenarlari hizasiz kalir. OLCULEN
(kayipsiz, bilinen kaymali kopya): cerceve basina hizalamayla 45 ppm'de codec
S/N'i 33.4 dB, 204 ppm'de 20.4 dB -- ve frekansla dusuyor, cerceve ici
kaymanin tipik izi. Kayipsiz bir kopyada tabana yakin olmaliydi.

Burada referans her ornek icin ayri konumdan ornekleniyor: Kaiser pencereli
sinc (64 tap, beta 10), onceden hesaplanmis 8192 fazli tablo. OLCULEN
(0.9 Nyquist'e kadar 400 sinuzoidli analitik sinyal, 200 ppm):

    faz sayisi   interpolasyon   hata (bant)          maliyet (10 dk mono 48 kHz)
    1024         yuvarla         -66 dB               11.9 s
    8192         yuvarla         -82 .. -87 dB        13.2 s
    1024         dogrusal        -103 .. -116 dB      27.5 s

1024 fazda sinir tap sayisindan bagimsizdi (32/64/128 tap ayni -66 dB) --
yani sinirlayan faz cozunurluguydu. 8192 faz ayni maliyette 16-20 dB
kazandiriyor ve diger tabanlarin altinda kaliyor.

Bu yol yalnizca izleme gerektiginde kullanilir; sabit gecikmede cerceve
basina faz rampasi hem daha hizli hem yeterli (-67 dB).
"""

from __future__ import annotations

import numpy as np

HALF_TAPS = 32
PHASES = 8192
_BETA = 10.0
_CHUNK = 16384


def _kaiser(x: np.ndarray, beta: float) -> np.ndarray:
    """[-1, 1] araliginda Kaiser penceresi."""
    inside = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    window: np.ndarray = np.i0(beta * inside) / np.i0(beta)
    return window


def _table() -> np.ndarray:
    k = np.arange(-HALF_TAPS + 1, HALF_TAPS + 1)
    fraction = np.arange(PHASES + 1) / PHASES
    x = k[None, :] - fraction[:, None]
    table: np.ndarray = np.sinc(x) * _kaiser(x / HALF_TAPS, _BETA)
    return table


_TABLE = _table()
_OFFSETS = np.arange(-HALF_TAPS + 1, HALF_TAPS + 1)


def sample(signal: np.ndarray, positions: np.ndarray, *, base: int = 0) -> np.ndarray:
    """`signal`i kesirli `positions` noktalarinda ornekler.

    `positions` mutlak indekslerdir; `signal[0]` mutlak `base` indeksine
    karsilik gelir. Her konumun iki yaninda `HALF_TAPS` ornek bulunmali;
    bulunmayanlar `ValueError` firlatir (sessizce sifirla doldurmak kenarda
    sahte bir hata uretir).
    """
    whole = np.floor(positions).astype(np.int64)
    local = whole - base
    if positions.size and (
        local.min() - HALF_TAPS + 1 < 0 or local.max() + HALF_TAPS >= signal.size
    ):
        raise ValueError("konumlar icin tampon yetersiz: iki yanda HALF_TAPS ornek gerekli")
    phase = np.round((positions - whole) * PHASES).astype(np.int64)
    out = np.empty(positions.size, dtype=np.float64)
    for start in range(0, positions.size, _CHUNK):
        part = slice(start, start + _CHUNK)
        taps = signal[local[part, None] + _OFFSETS]
        out[part] = np.einsum("ij,ij->i", taps, _TABLE[phase[part]])
    return out
