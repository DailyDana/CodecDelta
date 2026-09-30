"""Bagimliliksiz PNG kodlayici ve NMR renk olcegi.

Plan spektrogramlar icin WebP (ffmpeg libwebp) ongoruyordu, ama rapordaki tek
goruntu NMR izgarasi: ~41 bant x en fazla 1000 zaman sutunu. Bu boyutta
`zlib` ile yazilan bir PNG birkac KB tutuyor, ffmpeg ya da Qt gerektirmiyor ve
tarayici onu CSS ile olceklendiriyor. WebP'ye gecmek icin bir sebep kalmadi.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

# NMR renk olcegi (-RANGE..+RANGE dB). Arayuzdeki olcekle ayni duraklar; 0 dB
# (gurultu = maskeleme esigi) ortadaki gri.
NMR_RANGE_DB = 20.0
NMR_STOPS: tuple[tuple[float, str], ...] = (
    (0.0, "#11111b"),
    (0.35, "#45475a"),
    (0.5, "#7f849c"),
    (0.65, "#fab387"),
    (1.0, "#f38ba8"),
)


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_rgb(image: np.ndarray) -> bytes:
    """(yukseklik, genislik, 3) uint8 diziyi PNG'ye kodlar."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("(h, w, 3) uint8 bekleniyor")
    height, width, _ = image.shape
    # Her satirin basina filtre turu 0 (None).
    raw = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), image.reshape(height, width * 3)], axis=1
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
        + _chunk(b"IEND", b"")
    )


def _hex(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def colormap(
    values: np.ndarray, *, low: float, high: float, stops: tuple[tuple[float, str], ...]
) -> np.ndarray:
    """Degerleri [low, high] araliginda duraklar arasi dogrusal renge cevirir."""
    t = np.clip(
        (np.nan_to_num(values, nan=low, neginf=low, posinf=high) - low) / (high - low), 0.0, 1.0
    )
    positions = np.array([p for p, _ in stops])
    rgb = np.array([_hex(c) for _, c in stops], dtype=np.float64)
    out = np.empty((*t.shape, 3), dtype=np.float64)
    for channel in range(3):
        out[..., channel] = np.interp(t, positions, rgb[:, channel])
    result: np.ndarray = np.round(out).astype(np.uint8)
    return result


def nmr_png(grid_db: np.ndarray) -> bytes:
    """(bant, zaman) NMR izgarasi -> PNG. Dusuk frekans altta."""
    image = colormap(grid_db[::-1], low=-NMR_RANGE_DB, high=NMR_RANGE_DB, stops=NMR_STOPS)
    return encode_rgb(image)
