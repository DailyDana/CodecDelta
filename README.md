# CodecDelta

Measure what a lossy encoder actually did to your audio.

CodecDelta compares two encodes of the same recording — FLAC against Opus, an original
against a YouTube rip, MP3 against AAC — and reports how much information was lost, where in
the spectrum it went, and whether it is audible. It is also meant to answer the other common
question from a single file with no reference: *is this FLAC really lossless, or a transcode?*

> **Work in progress.** The analysis engine is being built bottom-up and is not yet an
> application. There is no user interface. See [Status](#status) for what actually runs
> today.

## Status

| Layer | State |
|---|---|
| `core/` — ffmpeg discovery, PCM streaming, ffprobe inspection, settings, privacy, ABX statistics | working |
| `bitstream/` — Ogg/Opus, Vorbis, FLAC, MP3 container and codec metadata | working |
| `dsp/`, `align/` — sub-sample delay estimation and alignment validity | working |
| coarse alignment, drift/PAL detection, "same recording?" verdict | not started |
| difference signal, per-band SNR, measurement-floor calibration | not started |
| psychoacoustic verdict, referenceless transcode detection | not started |
| user interface, encoder panel, HTML report, ABX test, batch scan, packaging | not started |

231 tests, `ruff` + `mypy --strict` clean, CI on Windows.

## Why this repository might be worth reading

The interesting part is not the feature list, it is the evidence trail. Several plausible
ideas were implemented, measured, and then removed because the measurement disagreed:

- **FLAC blocksize does not identify an encoder.** The same ffmpeg build produced 1024, 4096
  and 4608 depending only on how the input was fed. 4096 is the value usually attributed to
  the reference encoder, so the rule would have mislabelled most modern ffmpeg output.
- **ffmpeg does not write a real LAME tag.** The tag is in the right place with correct
  encoder delay, but every field from +9 to +20 is zero — so the lowpass value it is famous
  for is simply absent.
- **Reading an ffmpeg pipe with `read(n)` is 57× slower than `readinto`** on Windows: 117.6 s
  against 2.1 s for the same 211 MB stream.
- **ffmpeg's default soxr resampler silently walls off the top of the band**, which would
  have made the transcode detector flag clean files.
- **A phase-slope delay estimator failed on 79% of the input class this tool exists for.**
  An adversarial audit measured it across 12,600 synthetic trials plus real music and real
  codec pairs; it was replaced, and the reasoning is written down rather than lost.

These are recorded in [docs/DECISIONS.md](docs/DECISIONS.md), with the numbers behind them.
[docs/SPEC-alignment.md](docs/SPEC-alignment.md) is the contract for the alignment core,
written before the code it describes.

## Planned

- Compare two files with automatic sample-accurate alignment (offsets, PAL speed-up, drift,
  polarity inversion, channel swaps), then a single-pass streaming analysis producing
  per-band SNR, lowpass cutoff, residual level, and a coherence decomposition separating
  *linear* differences (EQ, level, resampling) from real *codec noise*.
- Judge audibility against an anchor ladder built from the user's own reference file:
  *"the measured difference is equivalent to Opus at roughly 112 kbps for this track."*
- Verify a single file with no reference: per-frame cutoff statistics, knee sharpness,
  above-cutoff noise floor, joint-stereo collapse, bit-level entropy and container metadata,
  reported as readable reasons rather than one opaque score.
- Encode and compare, with a bitrate ladder, to find how many kbps a particular track needs.
- A blind ABX test with sample-aligned, level-matched, click-free switching and statistics
  that never report "no difference" from a failed test.
- Batch-scan a library for files whose signatures are consistent with a lossy source.

Video files are accepted as input and never have their video decoded — a 20 GB concert MKV
costs the same as its audio track alone. This already works: probe and decode were verified
at 1242× realtime on a real 684 s YouTube m4a.

## Requirements

- Windows 10/11
- Python 3.11+ (developed and tested on 3.14)
- ffmpeg **and** ffprobe on the system — discovered automatically

```
uv venv --python 3.14 .venv
uv pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

## ffmpeg

CodecDelta does **not** bundle or redistribute ffmpeg; it locates an ffmpeg already present
on the system. ffmpeg is separately licensed and its own terms govern that binary.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

You may use, study, modify and redistribute this software. If you distribute a modified
version, you must release its source under the same licence. Commercial use is permitted;
making it closed-source is not.
