# CodecDelta

Measure what a lossy encoder actually did to your audio.

CodecDelta compares two encodes of the same recording — FLAC against Opus, an original
against a YouTube rip, MP3 against AAC — and reports how much information was lost, where in
the spectrum it went, and whether it is audible. It is also meant to answer the other common
question from a single file with no reference: *is this FLAC really lossless, or a transcode?*

> **Early version (0.2.0).** The analysis engine, the desktop window, encoding and shareable
> reports work; blind ABX testing and batch scanning are not built yet. See [Status](#status).

## Status

| Layer | State |
|---|---|
| `core/` — ffmpeg discovery, PCM streaming, ffprobe inspection, settings, privacy, ABX statistics | working |
| `bitstream/` — Ogg/Opus, Vorbis, FLAC, MP3 container and codec metadata | working |
| `dsp/`, `align/` — sub-sample delay estimation and alignment validity | working |
| coarse alignment, drift/PAL detection, "same recording?" verdict | working (engine only) |
| single-pass comparison: per-band SNR (mid/side), linear vs codec-noise split, measured floor, clock-drift tracking | working (engine only) |
| ERB masking model + NMR, anchor ladder verdict ("equivalent to Opus ~128 kbps"), referenceless lossy-source detection with calibrated thresholds | working (engine only) |
| desktop window: analyze tab (compare, verify, ladder), English/Turkish | working — v0.1.0 |
| encode panel: 12 codecs discovered from the ffmpeg build, bitrate / quality / advanced options, never overwrites, encode-then-compare | working |
| HTML report: single self-contained file, charts as inline SVG, light/dark, paths and user names removed and audited before writing | working — v0.2.0 |
| ABX test, batch scan, packaging | not started |

492 tests, `ruff` + `mypy --strict` clean, CI on Windows.

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
- **Dividing a correlation by the overlap count finds the wrong place half the time.**
  Locating a 4 s excerpt inside a 30 s recording failed on 20 of 40 seeds, with errors up
  to 32 s. A true Pearson coefficient over the overlap: 40 of 40.
- **Seeking with `ffmpeg -ss` is not sample-accurate on every format.** Measured against
  a full decode: exact on WAV, FLAC, Ogg Opus and MP3; off by 70–820 samples on M4A/AAC
  depending on position, a constant −128 on Ogg Vorbis, −48 on WebM Opus. Only measured
  formats take the fast path.
- **ffmpeg can requantise to 16 bits in the middle of a resampling chain.** A 16-bit source
  round-tripped 44.1→48→44.1 kHz measured 81 dB instead of 148 dB until the chain was
  forced to floating point.
- **One tonality choice moves the noise-to-mask ratio by 15 dB.** On the same near-transparent
  Opus file the median NMR read +5.8, −9.2 or +0.8 dB depending on how tonality was estimated.
  No verdict rests on an absolute NMR; the headline comes from a ladder of the user's own
  reference encoded at known bitrates, which is monotonic and self-validating.
- **The plan's transcode heuristics were partly backwards.** On nine CD tracks and 117
  transcodes: the CD itself has a steep knee at 19.6 kHz, per-frame cutoff variance runs the
  opposite way from the plan, and the side channel does not discriminate. What does: the
  median per-frame cutoff (real ≥ 21.2 kHz, every non-transparent codec ≤ 20.2 kHz).
- **ffmpeg's FLAC encoder takes its block size from the decoder feeding it.** An MP3 decoded
  straight to FLAC gets 47-sample blocks and barely compresses (0.97), which had made
  compression ratio look like a strong lossy-source signal. It is not.
- **PAL speed-up cannot be measured by alignment anchors — so it is tested instead.**
  Anchors lose lock beyond about 1000 ppm; PAL is 41,667 ppm. But PAL is not a continuous
  unknown, it is one entry in a short table, and testing each entry recovers it exactly.

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
.venv\Scripts\python -m app            # open the window
.venv\Scripts\python -m pytest         # run the tests
```

After setup, `CodecDelta.bat` starts the window without a console. Drop one or two audio
files onto it to load them as reference and test; `CodecDelta.bat --console` keeps a console
open to show errors if the window does not appear.

## ffmpeg

CodecDelta does **not** bundle or redistribute ffmpeg; it locates an ffmpeg already present
on the system. ffmpeg is separately licensed and its own terms govern that binary.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

You may use, study, modify and redistribute this software. If you distribute a modified
version, you must release its source under the same licence. Commercial use is permitted;
making it closed-source is not.
