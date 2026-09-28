"""Referanssiz dogrulama ozelliklerini etiketli bir sette olcer.

Kullanim:
    python tools/calibrate_transcode.py <dizin>

Dizindeki dosyalar adlariyla etiketlenir: `*_real.flac` gercek kayipsiz,
`*_<codec>_<ayar>*.flac` o codec'ten transcode. Her dosya icin
`app.single.spectral` ozellikleri hesaplanir ve sinif basina min / medyan /
max tablosu basilir. `app/single/thresholds.py` icindeki her sayi bu ciktidan
gelir; tablo oradaki yorumlara kopyalanir.

Set uretimi icin bkz. bu dosyanin sonundaki not.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ffmpeg_locate import discover
from app.core.probe import probe
from app.single import spectral

FEATURES = (
    ("knee_hz", "diz kHz", 1e-3),
    ("knee_drop_db", "diz dusus dB/500Hz", 1.0),
    ("floor_rel_db", "taban rel dB", 1.0),
    ("cutoff_median_hz", "kesim p50 kHz", 1e-3),
    ("cutoff_iqr_hz", "kesim IQR Hz", 1.0),
    ("side_hf_rel_db", "side HF dB", 1.0),
)


def label_of(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_real"):
        return "real"
    parts = stem.split("_")[1:]
    return "_".join(parts)


def main(directory: Path) -> None:
    tools = discover()
    rows: dict[str, list[spectral.SpectralEvidence]] = defaultdict(list)
    files = sorted(directory.glob("*.flac"))
    for i, path in enumerate(files):
        info = probe(tools.ffprobe, path)
        stream = info.audio[0]
        evidence = spectral.analyse(
            tools.ffmpeg,
            path,
            sample_rate=stream.sample_rate,
            channels=stream.channels,
        )
        rows[label_of(path)].append(evidence)
        print(f"\r{i + 1}/{len(files)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    labels = ["real", *sorted(k for k in rows if k != "real")]
    header = f"{'sinif':14} {'n':>3}  " + "  ".join(f"{name:>22}" for _, name, _ in FEATURES)
    print(header)
    print("-" * len(header))
    for label in labels:
        items = rows[label]
        cells = []
        for attr, _, scale in FEATURES:
            values = [getattr(e, attr) * scale for e in items if not math.isnan(getattr(e, attr))]
            if not values:
                cells.append(f"{'--':>22}")
                continue
            values.sort()
            median = values[len(values) // 2]
            cells.append(f"{values[0]:7.1f}/{median:6.1f}/{values[-1]:6.1f}")
        print(f"{label:14} {len(items):3d}  " + "  ".join(cells))
    print("\nher hucre: min / medyan / max")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    main(Path(sys.argv[1]))

# Set uretimi (27 Eyl 2026 kalibrasyonu): 9 gercek CD parcasi (tek album,
# Loreena McKennitt "An Ancient Muse", 44.1/16), her birinden 60 s kesit, ffmpeg
# ile mp3 128/192/320/V0, opus 96/128/160 (48 kHz'de ve 44.1'e soxr ile geri),
# aac 128/256, vorbis q5 kodlanip FLAC'a geri cozuldu. ORNEKLEM TEK MASTERING:
# baska turlerde ve baska kayit zincirlerinde dagilimlar farkli olabilir.
