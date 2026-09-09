"""ffmpeg/ffprobe alt sureclerini calistirmanin TEK yeri.

Uc kural bu modulde kilitlenir:

1. `shell=True` asla kullanilmaz ve cmd.exe asla araya girmez. subprocess'e
   liste verildiginde Windows argumanlari CreateProcessW'ye UTF-16 olarak
   gecirir; Turkce karakter, bosluk, `&`, `%`, `!` ve emoji sorunsuz calisir.
   (Aniflow bunu PowerShell'de elle tirnaklamak ve `%VAR%` hilesi yazmak
   zorunda kalmisti; Python'da o kodun tamami gereksiz.)

2. stderr HER ZAMAN bosaltilir. ffmpeg ilerleme ve uyarilari stderr'e yazar;
   boru dolarsa ffmpeg yazma sirasinda blokede kalir ve is asla bitmez. Ayri
   bir daemon thread stderr'i sinirli bir tampona ceker.

3. Iptal, sureci degil surec AGACINI oldurur. `Popen.kill()` yalnizca dogrudan
   cocugu oldurur ve alt surecleri yetim birakir; `taskkill /T /F` agaci indirir.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from app.core.errors import CancelledError, FFmpegFailedError

# Konsol penceresinin yanip sonmesini engeller. subprocess.CREATE_NO_WINDOW
# yalnizca Windows'ta tanimli; modul baska platformda da import edilebilsin
# diye getattr ile aliniyor.
_CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# stderr tamponunun ust siniri. ffmpeg uzun islerde on binlerce uyari satiri
# yazabilir; hata mesaji icin son birkac yuz satir fazlasiyla yeterli.
_STDERR_TAIL_LINES = 400

# Iptal yoklama araligi. 50 ms, arayuzde aninda hissedilir ve bos dongu maliyeti
# olculemeyecek kadar kucuktur.
_POLL_INTERVAL_S = 0.05


class CancelToken:
    """Is parcaciklari arasinda paylasilan iptal bayragi.

    threading.Event uzerine ince bir sarmalayici: cagri yerlerinde
    `token.cancelled` okumak `event.is_set()`ten daha okunur, ve
    `raise_if_cancelled()` iptali tek satirda akis kontrolune cevirir.
    """

    __slots__ = ("_event", "_what")

    def __init__(self, what: str = "operation") -> None:
        self._event = threading.Event()
        self._what = what

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError(self._what)

    def wait(self, timeout: float) -> bool:
        """Iptal edilene kadar veya sure dolana kadar bekler."""
        return self._event.wait(timeout)


@dataclass(frozen=True)
class CompletedRun:
    """Kisa bir ffmpeg/ffprobe cagrisinin sonucu."""

    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")


class StderrCollector:
    """Bir borudan stderr'i arka planda ceken sinirli tampon.

    Bosaltmanin kendisi amac: tampon dolarsa ffmpeg yazarken blokede kalir.
    Sakladigi son satirlar hata mesajinda kullanilir.
    """

    def __init__(self, stream: IO[bytes] | None, max_lines: int = _STDERR_TAIL_LINES) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._pump, args=(stream,), name="ffmpeg-stderr", daemon=True
        )
        self._thread.start()

    def _pump(self, stream: IO[bytes] | None) -> None:
        if stream is None:
            return
        while True:
            try:
                raw = stream.readline()
            except (ValueError, OSError):
                # Surec oldurulunce boru kapanir; bu beklenen bir son.
                return
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            with self._lock:
                self._lines.append(line)

    def tail(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout)


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Sureci ve tum alt agacini oldurur.

    Once `taskkill /T /F` denenir (Windows'ta agaci indiren tek guvenilir yol);
    bulunamazsa veya basarisiz olursa dogrudan `kill()`e duser.
    """
    if proc.poll() is not None:
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(  # noqa: S603
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],  # noqa: S607
            capture_output=True,
            check=False,
            creationflags=_CREATE_NO_WINDOW,
            timeout=10,
        )
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.kill()


def spawn(
    exe: Path,
    args: Sequence[str],
    *,
    stdout: int = subprocess.PIPE,
    cwd: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Bir ffmpeg/ffprobe sureci baslatir ve Popen nesnesini dondurur.

    Cagiran taraf stderr'i bosaltmaktan sorumludur (bkz. `StderrCollector`).

    `bufsize=0`: stdout ham bir FileIO olarak kalir. PcmStream'in `readinto()`
    yolu icin sart; tamponlu sarmalayici araya girerse kopyasiz okuma bozulur.
    """
    return subprocess.Popen(  # noqa: S603
        [str(exe), *args],
        stdout=stdout,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        cwd=str(cwd) if cwd else None,
        creationflags=_CREATE_NO_WINDOW,
        bufsize=0,
    )


def run_capture(
    exe: Path,
    args: Sequence[str],
    *,
    timeout: float = 60.0,
    cancel: CancelToken | None = None,
    check: bool = True,
) -> CompletedRun:
    """Kisa bir cagriyi calistirir ve ciktisini tumuyle toplar.

    ffprobe, `-version`, `-encoders` gibi milisaniyelik isler icindir. Uzun
    coz me/kodlama islerinde KULLANILMAZ: cikti tumuyle bellege alinir.
    """
    proc = spawn(exe, args)
    collector = StderrCollector(proc.stderr)
    try:
        out = _read_until_exit(proc, timeout=timeout, cancel=cancel)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        collector.join()
        raise FFmpegFailedError(
            (str(exe), *args),
            -1,
            f"Timed out after {timeout:.0f} s.\n{collector.tail()}",
        ) from None
    except BaseException:
        kill_tree(proc)
        raise
    finally:
        collector.join()

    result = CompletedRun(
        args=(str(exe), *args),
        returncode=proc.returncode,
        stdout=out,
        stderr=collector.tail(),
    )
    if check and not result.ok:
        raise FFmpegFailedError(result.args, result.returncode, result.stderr)
    return result


def _read_until_exit(
    proc: subprocess.Popen[bytes], *, timeout: float, cancel: CancelToken | None
) -> bytes:
    """stdout'u toplarken hem zaman asimini hem iptali yoklar.

    `communicate()` iptal edilemez, bu yuzden stdout ayri bir thread'de okunur
    ve ana thread kisa araliklarla sureci ve iptal bayragini yoklar.
    """
    chunks: list[bytes] = []

    def _read() -> None:
        stream = proc.stdout
        if stream is None:
            return
        while True:
            try:
                chunk = stream.read(65536)
            except (ValueError, OSError):
                return
            if not chunk:
                return
            chunks.append(chunk)

    reader = threading.Thread(target=_read, name="ffmpeg-stdout", daemon=True)
    reader.start()

    # Iptal varsa onun Event'i uzerinde bekleriz: iptal aninda uyanir, bos
    # bekleme yapmayiz. Yoksa hicbir zaman set edilmeyen bir Event sleep gorevi
    # gorur -- time.sleep ile ayni ama tek kod yolu.
    idle = threading.Event()
    waited = 0.0
    while proc.poll() is None:
        if cancel is not None and cancel.cancelled:
            kill_tree(proc)
            raise CancelledError()
        if waited >= timeout:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        if cancel is not None:
            cancel.wait(_POLL_INTERVAL_S)
        else:
            idle.wait(_POLL_INTERVAL_S)
        waited += _POLL_INTERVAL_S

    # Surec bitti ama okuyucu thread borudaki son parcalari henuz almamis olabilir.
    reader.join(timeout=5.0)
    return b"".join(chunks)
