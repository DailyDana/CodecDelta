# CodecDelta

Measure what a lossy encoder actually did to your audio.

CodecDelta compares two encodes of the same recording — FLAC against Opus, an original
against a YouTube rip, MP3 against AAC — and tells you how much information was lost,
where in the spectrum it went, and whether you can actually hear it.

It also works with a single file, with no reference, to answer the other common question:
*is this FLAC really lossless, or is it a transcode?*

## Status

Early development. Not yet usable.

## What it does

- **Compare two files.** Automatic sample-accurate alignment (handles offsets, PAL speed-up,
  drift, polarity inversion, channel swaps), gain matching, then a single-pass streaming
  analysis producing per-band SNR, lowpass cutoff, residual level, and a coherence
  decomposition that separates *linear* differences (EQ, level, resampling) from real
  *codec noise*.
- **Judge audibility honestly.** An ERB-band noise-to-mask ratio, anchored against a ladder
  of encodes made from your own reference file: *"the measured difference is equivalent to
  Opus at roughly 112 kbps for this track."*
- **Verify a single file.** Per-frame cutoff statistics, knee sharpness, above-cutoff noise
  floor, joint-stereo collapse, bit-level entropy and container metadata — reported as a list
  of readable reasons, not one opaque score.
- **Encode and compare.** Built-in encoder panel (Opus, Vorbis, AAC, MP3, AC3, MP2, WMA,
  FLAC, ALAC, WavPack, TTA) with an optional bitrate ladder, so you can find out how many
  kbps this particular track actually needs.
- **Prove it by ear.** A blind ABX test with sample-aligned, level-matched, click-free
  switching and correct statistics.
- **Scan a library.** Batch-check a folder for files whose signatures are consistent with a
  lossy source.

Video files are accepted as input and never have their video decoded — a 20 GB concert MKV
costs the same as its audio track alone.

## Design notes

Choices that are not obvious from the code — and the heuristics that were measured
and then removed — are recorded in [docs/DECISIONS.md](docs/DECISIONS.md).

## Requirements

- Windows 10/11
- Python 3.14 (3.10+ works; the project pins 3.14)
- ffmpeg **and** ffprobe — discovered automatically, or downloaded by `tools/setup.ps1`

## ffmpeg and licensing

CodecDelta itself is MIT licensed. It does **not** bundle or redistribute ffmpeg; it locates
an ffmpeg already on the system, or downloads an official build at setup time. If you use a
GPL-licensed ffmpeg build, that build's license governs that binary, not this source tree.

## License

MIT — see [LICENSE](LICENSE).
