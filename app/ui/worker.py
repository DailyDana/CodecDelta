"""Uzun islerin arayuzu dondurmadan calismasi.

Motor fonksiyonu ayri bir QThread'de kosar; asama bildirimleri ve sonuc
sinyallerle ana is parcacigina doner (Qt kuyruklu baglanti). Iptal, motorun
zaten kullandigi `CancelToken` ile: token isaretlenir, ffmpeg surec agaci
motor tarafinda olduruluyor.

Iptal sonrasi motor `CancelledError` yerine baska bir hata da firlatabilir
(oldurulen ffmpeg'in "basarisiz" cikisi). Token isaretliyse sonuc her zaman
"iptal edildi" sayilir; kullaniciya sahte bir hata gosterilmez.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot

from app.core.errors import CancelledError, CodecDeltaError
from app.core.ffmpeg_runner import CancelToken

JobFn = Callable[[CancelToken, Callable[[str], None]], object]


class _Job(QObject):
    stage = pyqtSignal(str)
    succeeded = pyqtSignal(object, float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, fn: JobFn, token: CancelToken) -> None:
        super().__init__()
        self._fn = fn
        self._token = token

    @pyqtSlot()
    def run(self) -> None:
        started = time.perf_counter()
        try:
            result = self._fn(self._token, self.stage.emit)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            if self._token.cancelled:
                self.cancelled.emit()
            elif isinstance(exc, CodecDeltaError):
                self.failed.emit(exc.user_message())
            else:
                self.failed.emit("".join(traceback.format_exception_only(exc)).strip())
        else:
            if self._token.cancelled:
                self.cancelled.emit()
            else:
                self.succeeded.emit(result, time.perf_counter() - started)


class Runner(QObject):
    """Tek seferde tek is calistirir; yeni is eskisi bitmeden baslatilamaz."""

    stage = pyqtSignal(str)
    succeeded = pyqtSignal(object, float)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._job: _Job | None = None
        self._token: CancelToken | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def start(self, fn: JobFn) -> None:
        if self.busy:
            raise RuntimeError("bir is zaten calisiyor")
        self._token = CancelToken("analysis")
        self._thread = QThread()
        self._job = _Job(fn, self._token)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.stage.connect(self.stage)
        self._job.succeeded.connect(self.succeeded)
        self._job.failed.connect(self.failed)
        self._job.cancelled.connect(self.cancelled)
        # DOGRUDAN baglanti: `quit` is parcaciginda cagrilir (QThread.quit
        # is parcacigi guvenli). Kuyruklu baglantida quit ana is parcacigina
        # gidiyordu; kapanista ana is parcacigi `wait()` icinde bloke oldugu
        # icin hic calismiyor ve pencere 10 s donuyordu (denetim D16).
        for signal in (self._job.succeeded, self._job.failed, self._job.cancelled):
            # PyQt6 stub'u baglanti turu argumanini tanimlamiyor; calisma zamani destekler.
            signal.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)  # type: ignore[call-arg]
        self._thread.finished.connect(self._cleanup)
        self.busy_changed.emit(True)
        self._thread.start()

    def cancel(self) -> None:
        if self._token is not None:
            self._token.cancel()

    def wait(self, timeout_ms: int = 30000) -> bool:
        """Testler ve kapanis icin: is bitene kadar bekler."""
        return self._thread.wait(timeout_ms) if self._thread is not None else True

    def _cleanup(self) -> None:
        if self._job is not None:
            self._job.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._job = None
        self._thread = None
        self._token = None
        self.busy_changed.emit(False)
