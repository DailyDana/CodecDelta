"""Cikti dosyasi adlandirma: asla ustune yazma.

Aniflow'daki kural (`_2`, `_3` + oturum ici kullanilan adlar) birebir: ayni
kaynaktan ayni ayarla ikinci kez kodlandiginda da, henuz diske yazilmamis bir
isin adi da korunur. Diskteki dosyanin varligi tek basina yetmez; kuyrukta
bekleyen iki is ayni adi secebilirdi.
"""

from __future__ import annotations

from pathlib import Path


def output_path(
    source: Path,
    label: str,
    extension: str,
    *,
    directory: Path | None = None,
    suffix: str = "_enc",
    taken: set[Path] | None = None,
) -> Path:
    """`<kaynak><suffix>_<label>.<uzanti>`; varsa `_2`, `_3`...

    `directory` None ise kaynagin yani. `taken` verilirse secilen ad ona
    eklenir (oturum ici rezervasyon). Kaynagin kendisi de hic secilmez.
    """
    folder = directory if directory is not None else source.parent
    base = f"{source.stem}{suffix}_{label}"
    reserved = taken if taken is not None else set()
    candidate = folder / f"{base}.{extension}"
    n = 2
    while candidate.exists() or _same(candidate, source) or candidate in reserved:
        candidate = folder / f"{base}_{n}.{extension}"
        n += 1
    reserved.add(candidate)
    return candidate


def _same(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False
