"""Toplu tarama: kesif, kademeli tarama, onbellek, iptal-surdurme, paralellik."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from app.batch import scanner
from app.batch.cache import ScanCache
from app.batch.model import ScanEntry
from app.core.errors import CancelledError
from app.core.ffmpeg_locate import FFmpegTools
from app.core.ffmpeg_runner import CancelToken
from app.core.probe import probe
from app.single import verdict
from app.single.verdict import Verdict

NOISE = "anoisesrc=color=pink:sample_rate=44100:duration={d}:seed={s},tremolo=f=1.1:d=0.85"


def _flac(ffmpeg: Path, out: Path, *, seconds: int = 40, seed: int = 5) -> Path:
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            NOISE.format(d=seconds, s=seed),
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def library(ffmpeg_tools: FFmpegTools, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("library")
    ff = ffmpeg_tools.ffmpeg
    (root / "album").mkdir()
    (root / ".hidden").mkdir()
    clean = _flac(ff, root / "album" / "01 clean.flac")
    lossy = root / "tmp.mp3"
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(clean),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(lossy),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(lossy),
            "-c:a",
            "flac",
            str(root / "album" / "02 fake.flac"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(ff),
            "-y",
            "-v",
            "error",
            "-i",
            str(clean),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(root / "album" / "03 aac.m4a"),
        ],
        check=True,
    )
    (root / "album" / "04 broken.flac").write_bytes(b"fLaC\x00\x00")
    _flac(ff, root / ".hidden" / "skip.flac", seconds=5)
    return root


def test_discover_finds_lossless_candidates_only(tmp_path: Path) -> None:
    for name in ("b.flac", "a.WAV", "c.mp3", "d.opus", "e.txt", ".x/f.flac", "sub/g.aiff"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    found = [p.relative_to(tmp_path).as_posix() for p in scanner.discover(tmp_path)]
    assert found == ["a.WAV", "b.flac", "sub/g.aiff"]


@pytest.mark.needs_ffmpeg
def test_staged_scan_gives_each_file_its_bucket(ffmpeg_tools: FFmpegTools, library: Path) -> None:
    entries = {
        p.name: scanner.scan_file(ffmpeg_tools.ffmpeg, ffmpeg_tools.ffprobe, p)
        for p in scanner.discover(library)
    }
    assert set(entries) == {"01 clean.flac", "02 fake.flac", "03 aac.m4a", "04 broken.flac"}
    assert entries["01 clean.flac"].bucket == "consistent_lossless"
    assert entries["01 clean.flac"].stage == 1
    fake = entries["02 fake.flac"]
    assert fake.bucket == "consistent_lossy" and fake.suspicious and fake.cutoff_hz
    assert fake.cutoff_hz < 17_500 and fake.headline
    assert entries["03 aac.m4a"].bucket == "not_applicable" and entries["03 aac.m4a"].stage == 0
    assert entries["04 broken.flac"].bucket == "error" and entries["04 broken.flac"].error


@pytest.mark.needs_ffmpeg
def test_stage_two_runs_only_when_undetermined(
    ffmpeg_tools: FFmpegTools, library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[float, ...]] = []

    def fake_verify(ffmpeg, info, *, stream_index=0, excerpt_s=30.0, extra_at=(), cancel=None):  # type: ignore[no-untyped-def]
        calls.append(tuple(extra_at))
        return Verdict("undetermined" if not extra_at else "consistent_lossless", (), (), ())

    monkeypatch.setattr(verdict, "verify", fake_verify)
    entry = scanner.scan_file(
        ffmpeg_tools.ffmpeg, ffmpeg_tools.ffprobe, library / "album" / "01 clean.flac"
    )
    assert calls == [(), scanner.STAGE2_AT] and entry.stage == 2
    assert entry.bucket == "consistent_lossless"


@pytest.mark.needs_ffmpeg
def test_extra_excerpts_widen_the_evidence(ffmpeg_tools: FFmpegTools, library: Path) -> None:
    info = probe(ffmpeg_tools.ffprobe, library / "album" / "01 clean.flac")
    one = verdict.verify(ffmpeg_tools.ffmpeg, info, excerpt_s=8.0)
    three = verdict.verify(ffmpeg_tools.ffmpeg, info, excerpt_s=8.0, extra_at=(0.15, 0.8))
    assert one.spectral is not None and three.spectral is not None
    assert three.spectral.frames == pytest.approx(3 * one.spectral.frames, rel=0.05)
    assert any(getattr(n, "key", "") == "single.combined" for n in three.notes)


def _fake_entry(path: Path, bucket: str = "consistent_lossless") -> ScanEntry:
    stat = path.stat()
    return ScanEntry(path, stat.st_size, stat.st_mtime_ns, bucket, 1, rules=scanner.RULES)


def test_cache_skips_unchanged_files_and_notices_changes(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        f = tmp_path / f"{i}.flac"
        f.write_bytes(b"x" * (i + 1))
        files.append(f)
    scanned: list[Path] = []

    def scan(ffmpeg, ffprobe, path, *, cancel=None):  # type: ignore[no-untyped-def]
        scanned.append(path)
        return _fake_entry(path)

    cache_file = tmp_path / "cache.json"
    run = scanner.Scanner(Path("f"), Path("p"), ScanCache(cache_file, scanner.RULES), scan=scan)
    assert run.run(files, on_result=lambda e, c: None) == 3
    # Yeni oturum, ayni dosyalar: hicbiri yeniden taranmaz
    again = scanner.Scanner(Path("f"), Path("p"), ScanCache(cache_file, scanner.RULES), scan=scan)
    seen: list[bool] = []
    assert again.run(files, on_result=lambda e, c: seen.append(c)) == 0 and seen == [True] * 3
    # Dosya degisti -> yeniden; kurallar degisti -> hepsi yeniden
    files[1].write_bytes(b"changed")
    assert again.run(files, on_result=lambda e, c: None) == 1
    newer = scanner.Scanner(Path("f"), Path("p"), ScanCache(cache_file, "9.9.9"), scan=scan)
    assert newer.run(files, on_result=lambda e, c: None) == 3
    # Bozuk onbellek bos sayilir
    cache_file.write_text("{not json", encoding="utf-8")
    assert len(ScanCache(cache_file, scanner.RULES)) == 0


def test_cancel_keeps_finished_results_and_resumes(tmp_path: Path) -> None:
    files = []
    for i in range(12):
        f = tmp_path / f"{i:02d}.flac"
        f.write_bytes(bytes([i]))
        files.append(f)
    token = CancelToken("scan")

    def scan(ffmpeg, ffprobe, path, *, cancel=None):  # type: ignore[no-untyped-def]
        time.sleep(0.05)
        return _fake_entry(path)

    cache_file = tmp_path / "cache.json"
    done: list[Path] = []

    def on_result(entry: ScanEntry, cached: bool) -> None:
        done.append(entry.path)
        if len(done) == 4:
            token.cancel()

    first = scanner.Scanner(
        Path("f"), Path("p"), ScanCache(cache_file, scanner.RULES), workers=2, scan=scan
    )
    with pytest.raises(CancelledError):
        first.run(files, on_result=on_result, cancel=token)
    kept = len(ScanCache(cache_file, scanner.RULES))
    assert 4 <= kept < 12
    second = scanner.Scanner(
        Path("f"), Path("p"), ScanCache(cache_file, scanner.RULES), workers=2, scan=scan
    )
    assert second.run(files, on_result=lambda e, c: None) == 12 - kept


def test_files_are_scanned_in_parallel(tmp_path: Path) -> None:
    files = [tmp_path / f"{i}.flac" for i in range(8)]
    for f in files:
        f.write_bytes(b"x")
    active, peak = [0], [0]
    lock = threading.Lock()

    def scan(ffmpeg, ffprobe, path, *, cancel=None):  # type: ignore[no-untyped-def]
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.1)
        with lock:
            active[0] -= 1
        return _fake_entry(path)

    run = scanner.Scanner(Path("f"), Path("p"), None, workers=4, scan=scan)
    started = time.perf_counter()
    assert run.run(files, on_result=lambda e, c: None) == 8
    assert peak[0] == 4 and time.perf_counter() - started < 0.6


def test_entries_survive_the_cache_round_trip(tmp_path: Path) -> None:
    from app.core.messages import Message

    f = tmp_path / "a.flac"
    f.write_bytes(b"abc")
    entry = ScanEntry(
        f,
        3,
        os.stat(f).st_mtime_ns,
        "consistent_lossy",
        2,
        codec="flac",
        cutoff_hz=16000.0,
        reasons=(Message("single.cutoff", "content stops at {khz:.1f} kHz", khz=16.0),),
        rules=scanner.RULES,
    )
    back = ScanEntry.from_json(f, entry.to_json())
    assert back == entry and getattr(back.reasons[0], "key", "") == "single.cutoff"
