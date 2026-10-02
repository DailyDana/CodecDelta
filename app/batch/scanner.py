"""Toplu kutuphane taramasi: kayipsiz gorunen dosyalarda kayipli kaynak izi.

Dosya basina butce, kademeli ornekleme ve erken cikis (plan):

- Asama 0: ffprobe. Kayipli bicimdeki dosya (MP3, Opus...) icerige bakilmadan
  "uygulanamaz" olarak gecer.
- Asama 1: dosyanin ortasindan 20 s. Hukum acik (kayipsiz ya da kayipli
  kaynakla tutarli) ise durulur.
- Asama 2: belirsizse %15 ve %80'den 20'ser s daha eklenir; kanit uc kesitin
  toplamindan cikar.

Bas ve son %5 her zaman atlanir (giris, alkis, gizli parca); bu ve sessiz kesit
yedekleri tek dosya dogrulamasindan (`verdict.verify`) gelir, burada tekrar
edilmez.

Is Python-CPU degil ffmpeg-alt-sureci agirlikli (subprocess ve numpy GIL'i
birakir): `ThreadPoolExecutor`. `ProcessPoolExecutor` Windows'ta her isciye
PyQt6'yi yeniden yukler. Isci sayisi kodlama ile paylasilan ffmpeg semaforuyla
da sinirli (`core.tasks`).

Qt IMPORT ETMEZ.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ThreadPoolExecutor, wait
from pathlib import Path

from app import __version__
from app.batch.cache import ScanCache
from app.batch.model import ScanEntry, error_message
from app.core.errors import CancelledError, CodecDeltaError
from app.core.ffmpeg_runner import CancelToken
from app.core.probe import default_stream, probe
from app.core.tasks import MAX_FFMPEG, ffmpeg_slot
from app.single import verdict

# Kayipsiz olabilecek kapsayicilar. Kayipli uzantilar (mp3, opus, ogg, aac)
# taranmaz: soru "bu kayipsiz dosya gercekten kayipsiz mi".
AUDIO_EXTENSIONS = frozenset(
    {
        ".flac",
        ".wav",
        ".wave",
        ".w64",
        ".aif",
        ".aiff",
        ".aifc",
        ".m4a",
        ".alac",
        ".ape",
        ".wv",
        ".tta",
        ".tak",
        ".caf",
        ".mka",
    }
)
STAGE_EXCERPT_S = 20.0
STAGE2_AT = (0.15, 0.80)
# Onbellek bu kadar sonuctan bir yazilir; iptal ya da cokmede kayip sinirli kalir.
SAVE_EVERY = 25
# Kurallarin surumu: esikler degisince onbellekteki eski hukumler kullanilmaz.
RULES = __version__


def default_workers() -> int:
    return max(1, min(MAX_FFMPEG, (os.cpu_count() or 4)))


def discover(root: Path) -> list[Path]:
    """`root` altindaki aday dosyalar, sirali. Gizli klasorler atlanir."""
    found: list[Path] = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                found.append(Path(folder) / name)
    return found


def scan_file(
    ffmpeg: Path, ffprobe: Path, path: Path, *, cancel: CancelToken | None = None
) -> ScanEntry:
    """Tek dosyayi kademeli olarak tarar. Hata kayda yazilir, firlatilmaz."""
    started = time.perf_counter()
    stat = path.stat()

    def done(**fields: object) -> ScanEntry:
        return ScanEntry(
            path=path,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            seconds=time.perf_counter() - started,
            rules=RULES,
            **fields,  # type: ignore[arg-type]
        )

    try:
        info = probe(ffprobe, path, cancel=cancel)
        index = default_stream(info)
        stream = info.stream(index)
        common = {
            "codec": stream.codec,
            "sample_rate": stream.sample_rate,
            "duration": info.duration,
        }
        if not stream.is_lossless:
            return done(bucket="not_applicable", stage=0, **common)
        with ffmpeg_slot():
            result = verdict.verify(
                ffmpeg, info, stream_index=index, excerpt_s=STAGE_EXCERPT_S, cancel=cancel
            )
            stage = 1
            if result.bucket == "undetermined" and info.duration:
                result = verdict.verify(
                    ffmpeg,
                    info,
                    stream_index=index,
                    excerpt_s=STAGE_EXCERPT_S,
                    extra_at=STAGE2_AT,
                    cancel=cancel,
                )
                stage = 2
    except CancelledError:
        raise
    except CodecDeltaError as exc:
        text = exc.args[0] if exc.args else exc.user_message()
        return done(bucket="error", stage=0, error=text)
    except (OSError, ValueError) as exc:
        return done(bucket="error", stage=0, error=error_message(str(exc)))
    evidence = result.spectral
    cutoff = evidence.cutoff_median_hz if evidence is not None and evidence.frames else None
    return done(
        bucket=result.bucket,
        stage=stage,
        cutoff_hz=cutoff,
        reasons=result.reasons,
        counter=result.counter_reasons,
        notes=result.notes,
        **common,
    )


class Scanner:
    """Bir dosya listesini paralel tarar; onbellekteki dosyalar aninda doner."""

    def __init__(
        self,
        ffmpeg: Path,
        ffprobe: Path,
        cache: ScanCache | None = None,
        *,
        workers: int | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
        scan: Callable[..., ScanEntry] = scan_file,
    ) -> None:
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe
        self.cache = cache
        self.workers = workers or default_workers()
        self._executor_factory = executor_factory or (lambda n: ThreadPoolExecutor(n))
        self._scan = scan

    def run(
        self,
        paths: Iterable[Path],
        *,
        on_result: Callable[[ScanEntry, bool], None],
        cancel: CancelToken | None = None,
        rescan: bool = False,
    ) -> int:
        """Tarar ve her sonucu `on_result(kayit, onbellekten_mi)` ile bildirir.

        Donus: taranan (onbellekten gelmeyen) dosya sayisi. Iptalde
        `CancelledError` firlar; o ana kadarki sonuclar onbellege yazilmistir.
        """
        pending: list[Path] = []
        for path in paths:
            cached = None
            if self.cache is not None and not rescan:
                try:
                    stat = path.stat()
                except OSError:
                    stat = None
                if stat is not None:
                    cached = self.cache.get(path, stat.st_size, stat.st_mtime_ns)
            if cached is not None:
                on_result(cached, True)
            else:
                pending.append(path)

        scanned = 0
        executor = self._executor_factory(self.workers)
        try:
            for entry in self._parallel(executor, pending, cancel):
                scanned += 1
                if self.cache is not None:
                    self.cache.put(entry)
                    if scanned % SAVE_EVERY == 0:
                        self.cache.save()
                on_result(entry, False)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if self.cache is not None:
                self.cache.save()
        return scanned

    def _parallel(
        self, executor: Executor, paths: list[Path], cancel: CancelToken | None
    ) -> Iterator[ScanEntry]:
        """Isleri sinirli sayida kuyrukta tutar: iptal bekleyen isi baslatmaz."""
        queue = iter(paths)
        running: dict[Future[ScanEntry], Path] = {}

        def submit() -> None:
            path = next(queue, None)
            if path is not None:
                future = executor.submit(self._scan, self.ffmpeg, self.ffprobe, path, cancel=cancel)
                running[future] = path

        for _ in range(self.workers * 2):
            submit()
        while running:
            if cancel is not None:
                cancel.raise_if_cancelled()
            finished, _ = wait(running, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in finished:
                running.pop(future)
                entry = future.result()
                submit()
                yield entry
