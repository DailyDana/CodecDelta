"""Kullanilabilir bir ffmpeg + ffprobe cifti bulur ve yeteneklerini olcer.

Altin kural
-----------
    Bir aday dizin, AYNI dizinde ffprobe da barindirmiyorsa reddedilir.

Bu tek kural gercek bir tuzagi kapatir: bu makinede PATH uzerinde duran
`C:\\Program Files (x86)\\MPV Player\\ffmpeg.exe` `--disable-ffprobe` ile
derlenmis. `shutil.which("ffmpeg")` cagirmak, PATH sirasina gore, ffprobe'suz
bir kuruluma baglanmak demek. Bu yuzden `which` TEK BASINA asla kullanilmaz.

Kademeli bozulma
----------------
Zorunlu minimum yalnizca sudur: calisan bir ffprobe ve ham PCM uretebilen bir
ffmpeg. Geri kalan her yetenek opsiyoneldir; `libopus` yoksa arayuz Opus
secenegini griler ve sebebini yazar, sert hata vermez.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import CodecDeltaError, FFmpegCapabilityError, FFmpegNotFoundError
from app.core.ffmpeg_runner import run_capture

_EXE_SUFFIX = ".exe" if os.name == "nt" else ""

# Bu ikisi olmadan arac hicbir sey yapamaz.
REQUIRED_FILTERS = frozenset({"aresample", "astats"})

# Olmadiginda ozellik kaybi olan ama arac calismaya devam eden yetenekler.
# Arayuz bunlari ilgili kontrolleri grilemek icin okur.
OPTIONAL_FILTERS = frozenset(
    {"showspectrumpic", "showwavespic", "ebur128", "axcorrelate", "acrossover"}
)
OPTIONAL_ENCODERS = frozenset(
    {
        "libopus",
        "libvorbis",
        "aac",
        "aac_mf",
        "libmp3lame",
        "ac3",
        "eac3",
        "mp2",
        "wmav2",
        "flac",
        "alac",
        "wavpack",
        "tta",
    }
)

# Yetenek onbelleginin bicim surumu. Ayristirma degisirse artirilir ki eski
# onbellek sessizce yanlis cevap vermesin.
_CACHE_VERSION = 1

_VERSION_RE = re.compile(r"^ffmpeg version (\S+)")
_ENABLE_RE = re.compile(r"--enable-([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class Capabilities:
    """Bir ffmpeg derlemesinin olculen yetenekleri."""

    version: str
    build_flags: frozenset[str]
    encoders: frozenset[str]
    filters: frozenset[str]

    def missing_required(self) -> list[str]:
        return sorted(REQUIRED_FILTERS - self.filters)

    def has_encoder(self, name: str) -> bool:
        return name in self.encoders

    def has_filter(self, name: str) -> bool:
        return name in self.filters

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "build_flags": sorted(self.build_flags),
            "encoders": sorted(self.encoders),
            "filters": sorted(self.filters),
        }

    @staticmethod
    def from_json(data: dict[str, object]) -> Capabilities:
        def _set(key: str) -> frozenset[str]:
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"onbellekte bozuk alan: {key}")
            return frozenset(str(x) for x in value)

        return Capabilities(
            version=str(data.get("version", "")),
            build_flags=_set("build_flags"),
            encoders=_set("encoders"),
            filters=_set("filters"),
        )


@dataclass(frozen=True)
class FFmpegTools:
    """Birlikte calisan ffmpeg/ffprobe cifti ve yetenekleri."""

    ffmpeg: Path
    ffprobe: Path
    caps: Capabilities

    @property
    def directory(self) -> Path:
        return self.ffmpeg.parent


def _pair_in(directory: Path) -> tuple[Path, Path] | None:
    """Altin kural: ikisi de AYNI dizinde olmali."""
    ffmpeg = directory / f"ffmpeg{_EXE_SUFFIX}"
    ffprobe = directory / f"ffprobe{_EXE_SUFFIX}"
    if ffmpeg.is_file() and ffprobe.is_file():
        return ffmpeg, ffprobe
    return None


def candidate_dirs(*, explicit: Path | None = None, app_dir: Path | None = None) -> list[Path]:
    """Aranacak dizinleri oncelik sirasina gore dondurur."""
    out: list[Path] = []

    def add(p: Path | None) -> None:
        if p is None:
            return
        try:
            resolved = p.resolve()
        except OSError:
            return
        if resolved not in out:
            out.append(resolved)

    add(explicit)
    if app_dir is not None:
        add(app_dir / "bin")

    local = os.environ.get("LOCALAPPDATA")
    if local:
        local_path = Path(local)
        add(local_path / "CodecDelta" / "bin")
        # winget'in yt-dlp.FFmpeg paketi: surum klasoru adi her guncellemede
        # degistigi icin glob sart.
        winget = local_path / "Microsoft" / "WinGet" / "Packages"
        for pkg in sorted(winget.glob("yt-dlp.FFmpeg_*")):
            for build in sorted(pkg.glob("ffmpeg-*")):
                add(build / "bin")

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.strip():
            add(Path(entry.strip()))

    return out


def parse_version(version_output: str) -> tuple[str, frozenset[str]]:
    """`ffmpeg -version` ciktisindan surum ve --enable-* bayraklarini cikarir."""
    version = ""
    match = _VERSION_RE.search(version_output)
    if match:
        version = match.group(1)
    flags = frozenset(_ENABLE_RE.findall(version_output))
    return version, flags


def _parse_listing(output: str) -> frozenset[str]:
    """`-encoders` / `-filters` tablolarindan ad sutununu toplar.

    Iki tablonun da bicimi ayni: bayrak sutunu, ad, aciklama. Basliklar
    `-----` satirindan once gelir ve ad sutunu asla `-` ile baslamaz.
    """
    names: set[str] = set()
    started = False
    for line in output.splitlines():
        if not started:
            if set(line.strip()) == {"-"}:
                started = True
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        flags, name = parts[0], parts[1]
        # Bayrak sutunu harf ve nokta karisimidir; aciklama satirlari elenir.
        if not flags or any(c.isspace() for c in flags):
            continue
        names.add(name)
    return frozenset(names)


def probe_capabilities(ffmpeg: Path, *, timeout: float = 30.0) -> Capabilities:
    """Uc kisa cagriyla derlemenin yeteneklerini olcer (~300 ms)."""
    version_out = run_capture(ffmpeg, ["-hide_banner", "-version"], timeout=timeout)
    encoders_out = run_capture(ffmpeg, ["-hide_banner", "-encoders"], timeout=timeout)
    filters_out = run_capture(ffmpeg, ["-hide_banner", "-filters"], timeout=timeout)

    version, flags = parse_version(version_out.stdout_text())
    return Capabilities(
        version=version,
        build_flags=flags,
        encoders=_parse_listing(encoders_out.stdout_text()),
        filters=_parse_listing(filters_out.stdout_text()),
    )


def _cache_key(ffmpeg: Path) -> dict[str, object]:
    """Onbellek anahtari: yol + boyut + mtime. Dosya degisirse yeniden olculur."""
    stat = ffmpeg.stat()
    return {
        "version": _CACHE_VERSION,
        "path": str(ffmpeg),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def load_cached_capabilities(ffmpeg: Path, cache_file: Path) -> Capabilities | None:
    """Onbellekten yetenekleri okur; anahtar tutmuyorsa None dondurur."""
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("key") != _cache_key(ffmpeg):
        return None
    caps = data.get("caps")
    if not isinstance(caps, dict):
        return None
    try:
        return Capabilities.from_json(caps)
    except ValueError:
        return None


def save_cached_capabilities(ffmpeg: Path, caps: Capabilities, cache_file: Path) -> None:
    """Yetenekleri onbellege yazar. Basarisizlik sessizce yutulur.

    Onbellek yalnizca bir hizlandirmadir; yazilamamasi kullaniciyi ilgilendiren
    bir hata degildir (salt okunur klasor, disk dolu, esli erisim).
    """
    payload = {"key": _cache_key(ffmpeg), "caps": caps.to_json(), "saved_at": int(time.time())}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def discover(
    *,
    explicit: Path | None = None,
    app_dir: Path | None = None,
    cache_file: Path | None = None,
    require: bool = True,
) -> FFmpegTools:
    """Ilk kullanilabilir ffmpeg+ffprobe ciftini bulur ve olcer.

    `require=True` ise zorunlu yetenekler eksikse hata firlatir. Aday dizinler
    sirayla denenir; calistirilamayan bir aday (bozuk veya yarim indirilmis
    kopya) butun keşfi durdurmaz, ama sessizce de yutulmaz: reddedilme sebebi
    toplanir ve bulunamama hatasinda kullaniciya gosterilir.
    """
    searched: list[str] = []
    for directory in candidate_dirs(explicit=explicit, app_dir=app_dir):
        pair = _pair_in(directory)
        if pair is None:
            searched.append(f"{directory} - ffmpeg+ffprobe pair not found")
            continue
        ffmpeg, ffprobe = pair

        caps = None
        if cache_file is not None:
            caps = load_cached_capabilities(ffmpeg, cache_file)
        if caps is None:
            try:
                caps = probe_capabilities(ffmpeg)
            except (CodecDeltaError, OSError) as exc:
                searched.append(f"{directory} - could not be run: {exc}")
                continue
            if cache_file is not None:
                save_cached_capabilities(ffmpeg, caps, cache_file)

        missing = caps.missing_required()
        if missing:
            if require:
                raise FFmpegCapabilityError(str(ffmpeg), missing)
            searched.append(f"{directory} - missing {', '.join(missing)}")
            continue
        return FFmpegTools(ffmpeg=ffmpeg, ffprobe=ffprobe, caps=caps)

    raise FFmpegNotFoundError(searched)
