"""CodecDelta hata turleri.

Kural: kullaniciya gosterilecek her hata bu agactan turer ve `user_message()`
ile Ingilizce, eyleme donuk bir cumle verir. Ham ffmpeg stderr'i mesaja
gomulmez; ayri bir alanda tasinir ki arayuz onu "Details" altinda gosterebilsin.
"""

from __future__ import annotations


class CodecDeltaError(Exception):
    """Tum CodecDelta hatalarinin koku."""

    def user_message(self) -> str:
        return str(self)


class CancelledError(CodecDeltaError):
    """Kullanici isi iptal etti. Hata degil, akis kontrolu."""

    def __init__(self, what: str = "operation") -> None:
        super().__init__(f"The {what} was cancelled.")


class FFmpegNotFoundError(CodecDeltaError):
    """Calisan bir ffmpeg + ffprobe cifti bulunamadi."""

    def __init__(self, searched: list[str]) -> None:
        self.searched = searched
        super().__init__(
            "No usable ffmpeg installation was found. CodecDelta needs ffmpeg and "
            "ffprobe in the same folder."
        )


class FFmpegCapabilityError(CodecDeltaError):
    """Bulunan ffmpeg zorunlu bir yetenegi tasimiyor."""

    def __init__(self, path: str, missing: list[str]) -> None:
        self.path = path
        self.missing = missing
        super().__init__(f"This ffmpeg build is missing required features: {', '.join(missing)}.")


class FFmpegFailedError(CodecDeltaError):
    """ffmpeg/ffprobe sifirdan farkli donus kodu verdi."""

    def __init__(self, args: tuple[str, ...], returncode: int, stderr: str) -> None:
        self.args = args
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"ffmpeg exited with code {returncode}.")


class ProbeError(CodecDeltaError):
    """Dosya okunamadi veya icinde kullanilabilir bir ses izi yok."""


class UnsupportedInputError(CodecDeltaError):
    """Dosya acildi ama analiz icin uygun degil (sure yok, kanal yok, vb.)."""
