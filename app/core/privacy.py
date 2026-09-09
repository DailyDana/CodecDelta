"""Paylasilabilir cikti icin kisisel iz temizligi.

Kural: rapor VARSAYILAN OLARAK paylasima hazir olmali. Kullanici bir analiz
raporunu bir foruma yapistirdiginda Windows kullanici adini, makine adini veya
disk duzenini birlikte yollamis olmamali.

Temizlenenler:
  - Mutlak yollar -> yalnizca dosya adi (baslik, alt metin, tooltip dahil)
  - Kullanici profili, kullanici adi, makine adi, surucu harfleri, %TEMP%
  - ffmpeg komut satirlarindaki yollar -> <INPUT_A> / <INPUT_B> / <OUT>

Dosya kimligi olarak dosya adi + icerik SHA-256'sinin ilk 16 hex'i kullanilir:
iki rapor ayni dosyayi konustugunu gosterebilir, ama yol sizmaz.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

# Icerik kimligi icin okunacak miktar. Tum dosyayi ozetlemek 10 GB'lik bir
# MKV'de dakikalar surer; bas + son + boyut pratikte fazlasiyla ayirt edici.
_HASH_HEAD_BYTES = 1 << 20
_HASH_TAIL_BYTES = 1 << 20

_ID_HEX_LEN = 16

# Windows mutlak yolu (surucu harfli veya UNC), iki secenekli:
#
#   1. Bosluk ICEREN yollar: uzantiya kadar tembel eslesme. Muzik
#      kutuphanelerinde bosluklu klasor adi kuraldir (D:\Secret Folder\x.flac)
#      ve bosluk kabul etmeyen bir desen o klasor adini OLDUGU GIBI sizdirir.
#      Uzunluk MAX_PATH ile sinirli, boylece asiri eslesme bir cumleyi yutmaz.
#   2. Uzantisiz yollar (klasorler): bosluk kabul etmeyen dar desen.
#
# Serbest metinde bir yolun nerede bittigi bicimsel olarak belirsizdir, bu
# yuzden `scrub` son care bir agdir. Yapisal alanlar (dosya adi, komut satiri)
# `scrub_path` ve `scrub_command` ile yol farkindaligiyla temizlenir; `audit`
# de geriye kalani yakalamak icin ayni deseni kullanir.
_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\)"
    r"(?:[^\"'<>|\r\n]{0,260}?\.[A-Za-z0-9]{1,8}(?=[\s\"'<>|]|$)"
    r"|[^\s\"'<>|]*)"
)


def _secret_strings() -> list[str]:
    """Ciktida gorunmemesi gereken makineye ozgu dizeler.

    Uzundan kisaya siralanir: `C:\\Users\\someone\\AppData` once temizlenmezse
    geriye `\\AppData` kalir ve kullanici adi zaten gitmis olsa da yol
    parcalari birbirine karisir.
    """
    candidates = [
        os.environ.get("USERPROFILE", ""),
        os.environ.get("APPDATA", ""),
        os.environ.get("LOCALAPPDATA", ""),
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
        os.environ.get("USERNAME", ""),
        os.environ.get("COMPUTERNAME", ""),
    ]
    seen: list[str] = []
    for c in candidates:
        c = c.strip()
        if len(c) >= 3 and c not in seen:
            seen.append(c)
    return sorted(seen, key=len, reverse=True)


def scrub(text: str, *, extra: Iterable[str] = ()) -> str:
    """Bir metinden makineye ozgu izleri siler.

    Once bilinen ozel dizeler, sonra kalan her mutlak yol temizlenir. Sira
    onemli: yol deseni once calissaydi kullanici adi yol disinda (orn. bir
    etiket icinde) gecti ise ayakta kalirdi.
    """
    out = text
    for secret in list(extra) + _secret_strings():
        if secret:
            out = out.replace(secret, "<REDACTED>")
    return _ABS_PATH_RE.sub(lambda m: Path(m.group(0)).name or "<PATH>", out)


def scrub_path(path: Path | str) -> str:
    """Bir yolu paylasilabilir hale getirir: yalnizca dosya adi kalir."""
    return Path(path).name


def content_id(path: Path) -> str:
    """Dosyanin icerigine bagli kisa kimlik.

    Bas ve son 1 MB ile dosya boyutu ozetlenir. Kriptografik bir taahhut degil,
    bir esitlik ipucu: ayni kimlik iki raporun ayni dosyayi konustugunu gosterir.
    Tum dosyayi okumak buyuk konteynerlerde kabul edilemez.
    """
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as fh:
        digest.update(fh.read(_HASH_HEAD_BYTES))
        if size > _HASH_HEAD_BYTES + _HASH_TAIL_BYTES:
            fh.seek(-_HASH_TAIL_BYTES, os.SEEK_END)
            digest.update(fh.read(_HASH_TAIL_BYTES))
    return digest.hexdigest()[:_ID_HEX_LEN]


def file_identity(path: Path) -> str:
    """Raporda gorunecek dosya kimligi: ad + icerik ozeti."""
    return f"{path.name} ({content_id(path)})"


def scrub_command(args: Iterable[str], labels: Mapping[str, str] | None = None) -> str:
    """Bir ffmpeg komut satirini yeniden uretilebilir ama anonim hale getirir.

    Komut satiri raporda degerlidir (kullanici olcumu tekrarlayabilir), ama ham
    haliyle tum kutuphane duzenini sizdirir. Bilinen girdi/cikti yollari
    `labels` ile etiketlenir, kalan yollar dosya adina indirgenir.
    """
    mapping = dict(labels or {})
    parts: list[str] = []
    for arg in args:
        replaced = mapping.get(arg)
        if replaced is not None:
            parts.append(replaced)
            continue
        # Burada belirsizlik YOK: her arguman tek basina tam bir degerdir, yani
        # bir yolsa tumuyle yoldur. Regex tahminine gerek kalmadan dosya adina
        # indirgenir; bosluklu klasor adlari bu yolda sizamaz.
        cleaned = Path(arg).name if _ABS_PATH_RE.fullmatch(arg) else scrub(arg)
        parts.append(f'"{cleaned}"' if " " in cleaned else cleaned)
    return " ".join(parts)


def audit(text: str, *, extra: Iterable[str] = ()) -> list[str]:
    """Metinde kalan kisisel izleri listeler. Bos liste = temiz.

    Rapor uretiminin sonunda cagrilir ve bir test bunu bos bekler. Sizinti
    sessizce gecmesin diye ayri bir denetim: `scrub` ileride bir alani atlarsa
    hata burada yakalanir.
    """
    found: list[str] = []
    for secret in list(extra) + _secret_strings():
        if secret and secret in text:
            found.append(secret)
    found.extend(m.group(0) for m in _ABS_PATH_RE.finditer(text))
    return found
