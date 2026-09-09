"""Ayarlar ve gizlilik temizligi testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core import privacy
from app.core.settings import SCHEMA_VERSION, Settings, load, save

# -- ayarlar ----------------------------------------------------------------


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.clamped() == s
    assert s.language == "en"
    assert s.schema_version == SCHEMA_VERSION


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = Settings(language="tr", max_workers=4, temp_root="D:\\Temp")
    assert save(original, path)
    assert load(path) == original.clamped()


def test_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load(tmp_path / "yok.json") == Settings()


def test_corrupt_file_returns_defaults(tmp_path: Path) -> None:
    """Bozuk ayar dosyasi uygulamayi acilmaz yapmamali."""
    path = tmp_path / "settings.json"
    path.write_text("{ bu json degil", encoding="utf-8")
    assert load(path) == Settings()


def test_non_object_json_returns_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load(path) == Settings()


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    """Ileri surumden gelen bir dosya cokme yaratmamali."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"language": "tr", "quantum_mode": True}), encoding="utf-8")
    loaded = load(path)
    assert loaded.language == "tr"
    assert not hasattr(loaded, "quantum_mode")


def test_wrong_types_fall_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"max_workers": "cok", "keep_temp": "evet", "language": 5}),
        encoding="utf-8",
    )
    loaded = load(path)
    assert loaded.max_workers == Settings().max_workers
    assert loaded.keep_temp is False
    assert loaded.language == "en"


def test_bool_is_not_accepted_as_int(tmp_path: Path) -> None:
    """isinstance(True, int) dogrudur; JSON'daki true sessizce 1 olmamali."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"max_workers": True}), encoding="utf-8")
    assert load(path).max_workers == Settings().max_workers


def test_out_of_range_values_are_clamped(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"max_workers": 999}), encoding="utf-8")
    assert load(path).max_workers == 12
    path.write_text(json.dumps({"max_workers": -5}), encoding="utf-8")
    assert load(path).max_workers == 1


def test_invalid_enum_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"language": "de", "depth": "instant"}), encoding="utf-8")
    loaded = load(path)
    assert loaded.language == "en"
    assert loaded.depth == "full"


def test_schema_version_is_forced_current(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"schema_version": 0, "language": "tr"}), encoding="utf-8")
    assert load(path).schema_version == SCHEMA_VERSION


def test_save_failure_is_reported_not_raised(tmp_path: Path) -> None:
    """Yazilamayan ayar dosyasi cokme degil, False donusu uretmeli."""
    blocker = tmp_path / "blocked"
    blocker.write_text("bir dosya", encoding="utf-8")
    assert save(Settings(), blocker / "settings.json") is False


def test_temp_dir_uses_configured_root(tmp_path: Path) -> None:
    s = Settings(temp_root=str(tmp_path))
    assert s.temp_dir() == tmp_path / "CodecDelta"


# -- gizlilik ---------------------------------------------------------------


def test_scrub_removes_absolute_paths() -> None:
    text = r"Analysed D:\Music\Album\01 - Track.flac successfully"
    out = privacy.scrub(text)
    assert "D:\\Music" not in out
    assert "01 - Track.flac" in out


def test_scrub_removes_user_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USERPROFILE", r"C:\Users\someone")
    monkeypatch.setenv("USERNAME", "someone")
    out = privacy.scrub(r"loaded from C:\Users\someone\Downloads\a.flac by someone")
    assert "someone" not in out


def test_scrub_handles_unc_paths() -> None:
    out = privacy.scrub(r"reading \\NAS01\music\x.flac")
    assert "NAS01" not in out


def test_scrub_prefers_longest_secret_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kisa dize once temizlenirse uzun olanin kalintisi geride kalir."""
    monkeypatch.setenv("USERPROFILE", r"C:\Users\bob")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\bob\AppData\Local")
    monkeypatch.setenv("USERNAME", "bob")
    out = privacy.scrub(r"cache at C:\Users\bob\AppData\Local\CodecDelta\caps.json")
    assert "bob" not in out


def test_scrub_path_keeps_only_filename() -> None:
    assert privacy.scrub_path(r"D:\a\b\c.flac") == "c.flac"


def test_audit_finds_what_scrub_missed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USERNAME", "testuser")
    assert privacy.audit("nothing personal here") == []
    assert privacy.audit(r"see C:\Users\testuser\x.flac")


def test_audit_is_clean_after_scrub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temizlik ve denetim tutarli olmali; bu ikisi birbirini kontrol eder."""
    monkeypatch.setenv("USERPROFILE", r"C:\Users\testuser")
    monkeypatch.setenv("USERNAME", "testuser")
    text = r"input C:\Users\testuser\Music\a.flac output D:\out\b.opus (user testuser)"
    assert privacy.audit(privacy.scrub(text)) == []


def test_scrub_command_labels_known_paths() -> None:
    args = ["ffmpeg", "-i", r"D:\Music\ref.flac", "-c:a", "libopus", r"D:\out\test.opus"]
    out = privacy.scrub_command(
        args, {r"D:\Music\ref.flac": "<INPUT_A>", r"D:\out\test.opus": "<OUT>"}
    )
    assert out == "ffmpeg -i <INPUT_A> -c:a libopus <OUT>"
    assert privacy.audit(out) == []


def test_scrub_command_reduces_unlabelled_paths() -> None:
    out = privacy.scrub_command(["ffmpeg", "-i", r"D:\Secret Folder\x.flac"])
    assert "Secret Folder" not in out
    assert "x.flac" in out


def test_content_id_is_stable_and_distinguishes(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world" * 100)
    b.write_bytes(b"hello worlD" * 100)
    assert privacy.content_id(a) == privacy.content_id(a)
    assert privacy.content_id(a) != privacy.content_id(b)
    assert len(privacy.content_id(a)) == 16


def test_content_id_reads_head_and_tail_of_large_files(tmp_path: Path) -> None:
    """Buyuk dosyalarda sadece bas+son okunur ama son yine de ayirt edici."""
    size = 4 << 20
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"\x00" * size)
    b.write_bytes(b"\x00" * (size - 1) + b"\xff")
    assert privacy.content_id(a) != privacy.content_id(b)


def test_file_identity_has_no_path(tmp_path: Path) -> None:
    f = tmp_path / "track.flac"
    f.write_bytes(b"x" * 64)
    identity = privacy.file_identity(f)
    assert identity.startswith("track.flac (")
    assert str(tmp_path) not in identity
