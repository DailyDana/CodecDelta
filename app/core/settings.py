"""Kalici tercihler.

Aniflow'un `%APPDATA%\\Aniflow.settings.json` deseni izleniyor, iki eklemeyle:
`schema_version` alani ve alanlarin kelepcelenmesi.

Savunmaci yukleme kurali: dosyadan gelen HICBIR deger dogrudan guvenilmez.
Eksik anahtar varsayilana duser, bilinmeyen anahtar yok sayilir, tur uymazsa
varsayilana donulur, sayisal degerler araliga kelepcelenir. Bozuk bir ayar
dosyasi uygulamayi acilmaz hale getirmemeli -- ayarlar bir tercih kaydidir,
kritik veri degil.

Ayarlar bilincli olarak uygulama klasorunun DISINDA (`%APPDATA%`) durur;
boylece uygulamayi guncellemek veya tasinabilir kopyayi silmek tercihleri
goturmez.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

APP_NAME = "CodecDelta"

# Bicim surumu. Alan anlamlari degisirse artirilir ve gerekirse gocurulur.
SCHEMA_VERSION = 1

_LANGUAGES = ("en", "tr")
_DEPTHS = ("full", "quick")


def config_dir() -> Path:
    """Ayarlarin durdugu dizin (`%APPDATA%\\CodecDelta`)."""
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def settings_file() -> Path:
    return config_dir() / "settings.json"


def caps_file() -> Path:
    """ffmpeg yetenek onbellegi. Ayarlardan ayri: silinmesi zararsiz."""
    return config_dir() / "caps.json"


@dataclass(frozen=True)
class Settings:
    """Kullanici tercihleri. Tum alanlarin gecerli bir varsayilani var."""

    schema_version: int = SCHEMA_VERSION

    # Arayuz. Varsayilan Ingilizce (Aniflow ile ayni tercih).
    language: str = "en"

    # ffmpeg dizini. Bos = otomatik keşif.
    ffmpeg_dir: str = ""

    # Gecici dosya koku. Bos = sistem TEMP. Bu makinede C: 144 GB, D: 378 GB
    # bos; buyuk video isleri icin kullanici D:'yi secebilmeli.
    temp_root: str = ""

    # Cikti klasoru. Bos = kaynak dosyanin yanina.
    output_dir: str = ""
    output_suffix: str = "_enc"

    # Analiz derinligi: tum dosya ya da ornekleme.
    depth: str = "full"

    # Eszamanli ffmpeg isi. Tarayici ve kodlayici bu butceyi PAYLASIR.
    max_workers: int = 6

    keep_temp: bool = False
    check_updates: bool = True

    def clamped(self) -> Settings:
        """Alanlari gecerli araliga cekilmis bir kopya dondurur."""
        return replace(
            self,
            schema_version=SCHEMA_VERSION,
            language=self.language if self.language in _LANGUAGES else "en",
            depth=self.depth if self.depth in _DEPTHS else "full",
            max_workers=max(1, min(12, self.max_workers)),
            output_suffix=self.output_suffix or "_enc",
        )

    def temp_dir(self) -> Path:
        """Gecici dosyalarin yazilacagi dizin."""
        root = Path(self.temp_root) if self.temp_root else Path(os.environ.get("TEMP", "."))
        return root / APP_NAME


def _coerce(value: object, default: object) -> object:
    """Bir alani beklenen ture zorlar; olmazsa varsayilana duser.

    bool kontrolu int'ten ONCE gelmeli: Python'da `isinstance(True, int)`
    dogrudur ve JSON'daki `true` sessizce 1'e donusurdu.
    """
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, int):
        return value if isinstance(value, int) and not isinstance(value, bool) else default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return default


def load(path: Path | None = None) -> Settings:
    """Ayarlari yukler. Dosya yoksa veya bozuksa varsayilanlari dondurur."""
    target = path or settings_file()
    defaults = Settings()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults
    if not isinstance(raw, dict):
        return defaults

    values: dict[str, object] = {}
    for f in fields(Settings):
        default = getattr(defaults, f.name)
        # Bilinmeyen anahtarlar zaten hic okunmuyor: yalnizca tanimli alanlar
        # dolasiliyor. Ileri surumden gelen bir dosya bu sayede cokme yaratmaz.
        values[f.name] = _coerce(raw[f.name], default) if f.name in raw else default

    return Settings(**values).clamped()  # type: ignore[arg-type]


def save(settings: Settings, path: Path | None = None) -> bool:
    """Ayarlari yazar. Basarili olursa True dondurur.

    Hata yutulur ama SESSIZ degil: donus degeri arayuze "kaydedilemedi"
    diyebilme imkani verir. Salt okunur bir profil veya dolu disk yuzunden
    uygulamanin cokmesi kabul edilemez.
    """
    target = path or settings_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(settings.clamped())
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        return False
    return True
