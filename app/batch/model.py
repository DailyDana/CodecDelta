"""Toplu tarama kaydi. Qt IMPORT ETMEZ."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.messages import Message, _rebuild

# Kova: tek dosya dogrulamasinin kovalari + tarama ozel iki durum.
SUSPICIOUS = "consistent_lossy"
BUCKETS = (
    "consistent_lossy",
    "undetermined",
    "consistent_lossless",
    "not_applicable",
    "error",
)


def _dump(items: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        key = getattr(item, "key", None)
        params = getattr(item, "params", {})
        out.append({"key": key, "text": str(item), "params": params})
    return out


def _load(items: list[dict[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    for item in items:
        if item.get("key"):
            out.append(_rebuild(item["key"], item["text"], dict(item.get("params", {}))))
        else:
            out.append(str(item.get("text", "")))
    return tuple(out)


@dataclass(frozen=True)
class ScanEntry:
    path: Path
    size: int
    mtime_ns: int
    bucket: str
    # 0: icerige bakilmadi (kayipli bicim ya da hata), 1: tek kesit, 2: uc kesit.
    stage: int
    codec: str = ""
    sample_rate: int = 0
    duration: float | None = None
    cutoff_hz: float | None = None
    reasons: tuple[str, ...] = ()
    counter: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    error: str | None = None
    seconds: float = 0.0
    # Bu kaydi ureten kurallarin surumu; degisince kayit bayattir.
    rules: str = field(default="")

    @property
    def suspicious(self) -> bool:
        return self.bucket == SUSPICIOUS

    @property
    def headline(self) -> str:
        """Tabloda gosterilecek en onemli gerekce (cevrilebilir)."""
        if self.error:
            return self.error
        for group in (self.reasons, self.counter, self.notes):
            if group:
                return group[0]
        return ""

    def to_json(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "bucket": self.bucket,
            "stage": self.stage,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "cutoff_hz": self.cutoff_hz,
            "reasons": _dump(self.reasons),
            "counter": _dump(self.counter),
            "notes": _dump(self.notes),
            "error": _dump((self.error,))[0] if self.error else None,
            "seconds": self.seconds,
            "rules": self.rules,
        }

    @staticmethod
    def from_json(path: Path, data: dict[str, Any]) -> ScanEntry:
        error = _load([data["error"]])[0] if data.get("error") else None
        return ScanEntry(
            path=path,
            size=int(data["size"]),
            mtime_ns=int(data["mtime_ns"]),
            bucket=str(data["bucket"]),
            stage=int(data["stage"]),
            codec=str(data.get("codec", "")),
            sample_rate=int(data.get("sample_rate", 0)),
            duration=data.get("duration"),
            cutoff_hz=data.get("cutoff_hz"),
            reasons=_load(data.get("reasons", [])),
            counter=_load(data.get("counter", [])),
            notes=_load(data.get("notes", [])),
            error=error,
            seconds=float(data.get("seconds", 0.0)),
            rules=str(data.get("rules", "")),
        )


def error_message(text: str) -> Message:
    return Message("batch.error", "{text}", text=text)
