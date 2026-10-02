"""Toplu tarama onbellegi: (yol, boyut, degistirilme zamani) -> kayit.

Amac surdurulebilirlik: 1000 dosyanin 600'unde iptal edilen tarama kaldigi
yerden devam eder; degismeyen dosya yeniden cozulmez. Kayit, onu ureten
kurallarin surumunu tasir (`rules`); esikler degisince eski hukumler
kullanilmaz. Dosya JSON; yazma atomik (gecici dosya + yer degistirme), yarim
kalan yazma onceki onbellegi bozmaz. Okunamayan onbellek bos sayilir.

Qt IMPORT ETMEZ.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from app.batch.model import ScanEntry

CACHE_FORMAT = 1


class ScanCache:
    def __init__(self, path: Path, rules: str) -> None:
        self.path = path
        self.rules = rules
        self._entries: dict[str, ScanEntry] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, RecursionError):
            return
        if not isinstance(data, dict) or data.get("format") != CACHE_FORMAT:
            return
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return
        for key, value in entries.items():
            try:
                entry = ScanEntry.from_json(Path(key), value)
            except (KeyError, TypeError, ValueError):
                continue
            if entry.rules == self.rules:
                self._entries[key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, path: Path, size: int, mtime_ns: int) -> ScanEntry | None:
        """Dosya degismediyse ve kurallar ayniysa kayit; aksi halde None."""
        with self._lock:
            entry = self._entries.get(str(path))
        if entry is None or entry.size != size or entry.mtime_ns != mtime_ns:
            return None
        return entry

    def put(self, entry: ScanEntry) -> None:
        with self._lock:
            self._entries[str(entry.path)] = entry
            self._dirty = True

    def save(self) -> bool:
        """Degisiklik varsa atomik olarak yazar. Basariliysa True."""
        with self._lock:
            if not self._dirty:
                return True
            payload = {
                "format": CACHE_FORMAT,
                "entries": {key: e.to_json() for key, e in self._entries.items()},
            }
            self._dirty = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            with self._lock:
                self._dirty = True
            return False
        return True
