"""Uygulama genelinde ayni anda calisan ffmpeg sureci siniri.

Toplu tarama ve kodlama ayni semaforu paylasir: tarama surerken baslatilan bir
kodlama, 12 ffmpeg'in birbirini bogmasina yol acmasin. Sinir mantikli bir
cekirdek sayisi; tek bir agir is (karsilastirma) bunu kullanmaz, kendi iki
cozucusunu zaten acar.

Qt IMPORT ETMEZ.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

MAX_FFMPEG = max(2, min(6, (os.cpu_count() or 4)))
_SLOTS = threading.BoundedSemaphore(MAX_FFMPEG)


@contextmanager
def ffmpeg_slot() -> Iterator[None]:
    """Bir ffmpeg agir isi icin yer ayirir; is bitince birakir."""
    _SLOTS.acquire()
    try:
        yield
    finally:
        _SLOTS.release()
