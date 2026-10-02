"""CodecDelta hata turleri.

Kural: kullaniciya gosterilecek her hata bu agactan turer ve `user_message()`
ile Ingilizce, eyleme donuk bir cumle verir. Ileti bir `Message`dir (anahtar +
parametre), arayuz onu `localize` ile kendi diline cevirir; once sabit
Ingilizce dizelerdi ve Turkce arayuzde Ingilizce gorunuyordu (denetim D28).
Ham ffmpeg stderr'i mesaja gomulmez; ayri bir alanda tasinir ki arayuz onu
"Details" altinda gosterebilsin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.messages import Message


def _rebuild(cls: type[CodecDeltaError], args: tuple[Any, ...], state: dict[str, Any]) -> Any:
    error = cls.__new__(cls)
    Exception.__init__(error, *args)
    error.__dict__.update(state)
    return error


class CodecDeltaError(Exception):
    """Tum CodecDelta hatalarinin koku."""

    def user_message(self) -> str:
        return str(self)

    def __reduce__(self) -> tuple[Any, ...]:
        # Alt siniflarin `__init__` imzasi farkli: varsayilan pickle/copy
        # `cls(*self.args)` cagirip ya TypeError veriyor ya da iletiyi ikinci
        # kez sariyordu (denetim D31). Durum oldugu gibi geri kurulur.
        return (_rebuild, (type(self), self.args, dict(self.__dict__)))


class CancelledError(CodecDeltaError):
    """Kullanici isi iptal etti. Hata degil, akis kontrolu."""

    def __init__(self, what: str = "operation") -> None:
        self.what = what
        # `what` bir alan olarak kalir; iletiye konmaz (Ingilizce bir kelime,
        # Turkce cumlede karisik dururdu).
        super().__init__(Message("error.cancelled", f"The {what} was cancelled."))


class FFmpegNotFoundError(CodecDeltaError):
    """Calisan bir ffmpeg + ffprobe cifti bulunamadi."""

    def __init__(self, searched: list[str]) -> None:
        self.searched = searched
        super().__init__(
            Message(
                "error.ffmpeg_not_found",
                "No usable ffmpeg installation was found. CodecDelta needs ffmpeg and "
                "ffprobe in the same folder.",
            )
        )


class FFmpegCapabilityError(CodecDeltaError):
    """Bulunan ffmpeg zorunlu bir yetenegi tasimiyor."""

    def __init__(self, path: str, missing: list[str]) -> None:
        self.path = path
        self.missing = missing
        super().__init__(
            Message(
                "error.ffmpeg_missing",
                "This ffmpeg build is missing required features: {missing}.",
                missing=", ".join(missing),
            )
        )


class FFmpegFailedError(CodecDeltaError):
    """ffmpeg/ffprobe sifirdan farkli donus kodu verdi."""

    def __init__(self, command: tuple[str, ...], returncode: int, stderr: str) -> None:
        # `args` DEGIL: Exception.args'i eziyordu (D31).
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr
        # Hangi aracin dustugu soylenir; ffprobe hatasi "ffmpeg exited" diye
        # gorunuyordu (D28).
        tool = Path(command[0]).stem if command else "ffmpeg"
        super().__init__(
            Message(
                "error.ffmpeg_failed",
                "{tool} exited with code {code}.",
                tool=tool,
                code=returncode,
            )
        )


class ProbeError(CodecDeltaError):
    """Dosya okunamadi veya icinde kullanilabilir bir ses izi yok."""


class UnsupportedInputError(CodecDeltaError):
    """Dosya acildi ama analiz icin uygun degil (sure yok, kanal yok, vb.)."""
