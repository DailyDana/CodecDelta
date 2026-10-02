"""Hizalama ("ayni kayit mi") esiklerini etiketli GERCEK ciftlerle olcer.

Kullanim:
    python tools/calibrate_match.py <set-dizini> [<is-dizini>]

<set-dizini>, `tools/calibrate_transcode.py` ile ayni adlandirmayi kullanir:
`<parca>_real.flac` gercek kayipsiz kesit, `<parca>_<codec>_<ayar>.flac` onun
kayipli kodlamasi. Ciftler adlardan etiketlenir:

- same       : bir parcanin gercegi ile kendi kodlamasi (beklenen: hizali)
- different  : bir parcanin gercegi ile BASKA bir parcanin kodlamasi
               (beklenen: farkli kayit)
- master     : bir parcanin gercegi ile kendisinin agir EQ + dinamik sikistirma
               uygulanmis hali (SENTETIK "farkli master"; gercek remaster
               ciftimiz yok). <is-dizini>'ne uretilir.

Her cift icin planin olcutleri hesaplanir: zarf korelasyonu (rho, L1) ve ince
hizalama korelasyonu (L3), ayrica planin hukmu. Sinif basina min / medyan /
max ve esiklere gore yanlis siniflanan cift sayisi basilir.
`app/align/thresholds.py` yorumlarindaki "gercek veri" tablolari buradan gelir.
"""

from __future__ import annotations

import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.align import envelope, plan
from app.align.thresholds import MIN_ALIGNMENT_CORRELATION, MIN_ENVELOPE_CORRELATION
from app.compare.pipeline import open_track
from app.compare.reader import FFmpegWindowReader
from app.core import settings
from app.core.ffmpeg_locate import discover

# Sentetik "farkli master": belirgin EQ (bas +6, tiz -6, orta -4 dB) ve
# sert dinamik sikistirma. Gercek bir remaster kadar farkli olmayabilir.
MASTER_FILTER = (
    "equalizer=f=80:t=q:w=1:g=6,equalizer=f=1500:t=q:w=1:g=-4,"
    "equalizer=f=9000:t=q:w=1:g=-6,acompressor=threshold=-20dB:ratio=6:attack=5:release=80,"
    "volume=-3dB"
)


def measure(tools, reference: Path, test: Path) -> tuple[float, float, str]:  # type: ignore[no-untyped-def]
    ref, tst = open_track(tools.ffprobe, reference), open_track(tools.ffprobe, test)
    rate = max(ref.stream.sample_rate, tst.stream.sample_rate)
    ref_env = envelope.build(tools.ffmpeg, ref.path)
    test_env = envelope.build(tools.ffmpeg, tst.path)
    result = plan.build(
        ref_env,
        test_env,
        FFmpegWindowReader(tools.ffmpeg, ref.source(rate)),
        FFmpegWindowReader(tools.ffmpeg, tst.source(rate)),
        rate,
    )
    fine = result.fine.correlation if result.fine is not None else float("nan")
    return result.envelope.rho, abs(fine), result.verdict


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "_match"
    work.mkdir(parents=True, exist_ok=True)
    tools = discover(cache_file=settings.caps_file())

    reals = sorted(root.glob("*_real.flac"))
    encodes = defaultdict(list)
    for path in sorted(root.glob("*.flac")):
        if not path.stem.endswith("_real"):
            encodes[path.stem.split("_")[0]].append(path)
    rng = random.Random(7)  # noqa: S311 (tekrarlanabilir ornekleme, kriptografi degil)
    pairs: list[tuple[str, Path, Path]] = []
    names = [r.stem.split("_")[0] for r in reals]
    for real, name in zip(reals, names, strict=True):
        pairs += [("same", real, e) for e in encodes[name]]
        others = [n for n in names if n != name]
        for other in rng.sample(others, min(3, len(others))):
            pairs.append(("different", real, rng.choice(encodes[other])))
        master = work / f"{name}_master.flac"
        if not master.exists():
            subprocess.run(  # noqa: S603 (sabit ffmpeg argumanlari)
                [
                    str(tools.ffmpeg),
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(real),
                    "-af",
                    MASTER_FILTER,
                    "-c:a",
                    "flac",
                    str(master),
                ],
                check=True,
            )
        pairs.append(("master", real, master))

    rows: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for index, (label, reference, test) in enumerate(pairs, 1):
        rows[label].append(measure(tools, reference, test))
        if index % 25 == 0:
            print(f"  {index}/{len(pairs)}", file=sys.stderr)

    print(
        f"\nEsikler: zarf rho >= {MIN_ENVELOPE_CORRELATION}, ince korelasyon >= "
        f"{MIN_ALIGNMENT_CORRELATION}\n"
    )
    print(
        f"{'sinif':10} {'n':>4}  {'zarf rho min/medyan/max':>26}  {'ince |r| min/medyan/max':>26}"
    )
    for label in ("same", "master", "different"):
        data = rows[label]
        if not data:
            continue
        for column, title in ((0, "rho"), (1, "|r|")):
            values = sorted(v[column] for v in data if v[column] == v[column])
            if not values:
                continue
            mid = values[len(values) // 2]
            print(
                f"{label if column == 0 else '':10} {len(data) if column == 0 else '':>4}  "
                f"{title:>4} {values[0]:7.3f} {mid:7.3f} {values[-1]:7.3f}"
            )
        verdicts = Counter(v[2] for v in data)
        print(f"{'':10}       hukumler: {dict(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
