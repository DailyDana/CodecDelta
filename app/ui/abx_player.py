"""ABX oynatici: capraz gecis karistiricisini QAudioSink'e besler.

QMediaPlayer KULLANILMAZ: Media Foundation'a bagli, Ogg/Opus destegi Windows
surumune gore degisiyor ve gecikme denetimi yok. Burada cihaz, karistiricidan
cektigi (pull) float32 ornekleri calar; tampon 1024 cerceve (planda olculen:
~23 ms, sifir underrun).

Karistirici tum gecisleri, baslatmayi ve durdurmayi rampayla yapar; bu sinif
yalnizca cihaz yasam dongusunu yonetir. Durdurmada cihaz, rampa sessizlige
indikten SONRA kapatilir.
"""

from __future__ import annotations

from PyQt6.QtCore import QIODevice, QObject, QTimer
from PyQt6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

from app.abx.excerpts import ExcerptAudio
from app.abx.mixer import CrossfadeMixer, Source

BUFFER_FRAMES = 1024
_BYTES_PER_SAMPLE = 4


class MixerDevice(QIODevice):
    """Karistiricinin ciktisini cihazin istedigi kadar ureten sirali aygit."""

    def __init__(self, mixer: CrossfadeMixer) -> None:
        super().__init__()
        self._mixer = mixer
        self._frame_bytes = _BYTES_PER_SAMPLE * mixer.channels

    def readData(self, maxlen: int) -> bytes:
        frames = maxlen // self._frame_bytes
        return self._mixer.render(frames).tobytes() if frames > 0 else b""

    def writeData(self, data: object) -> int:
        return -1

    def isSequential(self) -> bool:
        return True

    def bytesAvailable(self) -> int:
        # Akis sonsuz (dongu): cihaz her zaman okuyabilir.
        return (1 << 20) + super().bytesAvailable()


def playback_rate() -> int:
    """Varsayilan cikis cihazinin tercih ettigi ornekleme hizi (yoksa 48 kHz)."""
    device = QMediaDevices.defaultAudioOutput()
    if device.isNull():
        return 48000
    rate = device.preferredFormat().sampleRate()
    return rate if rate > 0 else 48000


class AbxPlayer(QObject):
    """Bir kesit ciftini calar; kaynaklar arasinda rampayla gecer."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mixer: CrossfadeMixer | None = None
        self._device: MixerDevice | None = None
        self._sink: QAudioSink | None = None
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._close_if_silent)

    @property
    def loaded(self) -> bool:
        return self._mixer is not None

    @property
    def playing(self) -> bool:
        return self._sink is not None and self._sink.state() == QAudio.State.ActiveState

    def load(self, audio: ExcerptAudio) -> None:
        """Kesit ciftini yukler. Cihaz bu bicimi desteklemiyorsa RuntimeError."""
        self.release()
        fmt = QAudioFormat()
        fmt.setSampleRate(audio.rate)
        fmt.setChannelCount(audio.a.shape[1])
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        output = QMediaDevices.defaultAudioOutput()
        if output.isNull():
            raise RuntimeError("no audio output device")
        if not output.isFormatSupported(fmt):
            raise RuntimeError(
                f"the audio device does not accept {audio.rate} Hz, "
                f"{audio.a.shape[1]} channel float output"
            )
        self._mixer = CrossfadeMixer(audio.a, audio.b, audio.rate)
        self._device = MixerDevice(self._mixer)
        self._device.open(QIODevice.OpenModeFlag.ReadOnly)
        self._sink = QAudioSink(output, fmt, self)
        self._sink.setBufferSize(BUFFER_FRAMES * _BYTES_PER_SAMPLE * audio.a.shape[1])

    def play(self, source: Source) -> None:
        """`source`u calar. Calmiyorsa sessizlikten baslar; caliyorsa rampayla gecer."""
        if self._mixer is None or self._sink is None or self._device is None:
            return
        self._stop_timer.stop()
        self._mixer.select(source)
        self._mixer.start()
        if self._sink.state() != QAudio.State.ActiveState:
            self._sink.start(self._device)

    def stop(self) -> None:
        """Rampayla sessizlige iner, sonra cihazi durdurur."""
        if self._mixer is None:
            return
        self._mixer.stop()
        self._stop_timer.start(60)

    def rewind(self) -> None:
        if self._mixer is not None:
            self._mixer.rewind()

    def _close_if_silent(self) -> None:
        if self._mixer is not None and self._sink is not None and self._mixer.silent:
            self._sink.stop()

    def release(self) -> None:
        self._stop_timer.stop()
        if self._sink is not None:
            self._sink.stop()
            self._sink.deleteLater()
        if self._device is not None:
            self._device.close()
        self._sink = self._device = self._mixer = None
