# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the API may break between minor versions.

## [Unreleased]

## [0.2.0] - 2026-09-30

### Added
- **HTML reports** ("Save report…" in the Analyze tab) for a comparison, with
  the bitrate ladder when it was computed, or for a lossless verification.
  One self-contained file: charts are inline SVG that follow the reader's
  light or dark theme, the noise-to-mask map is an embedded PNG, and there is
  no script and no external resource. Written in the interface language.
- Reports are shareable by default: files appear by name and a content hash,
  never by path, and user and computer names are removed. A privacy audit runs
  before writing, and a report that still contains a trace is not written.
  Text from files and tags is escaped, so it cannot inject HTML.

### Fixed
- The privacy audit took the "s:/" in "https://" for a drive path, so any
  report with a link would have been refused.
- **Encode tab**: pick a source, a codec (Opus, AAC, AAC via Media Foundation,
  MP3, Vorbis, AC-3, E-AC-3, MP2, WMA, FLAC, ALAC, WavPack), a bitrate or a
  quality level and codec-specific options, and encode. Codecs missing from
  the ffmpeg build are shown greyed out with the reason. Progress is shown as
  a percentage and the encode can be cancelled; a cancelled or failed encode
  leaves no partial file. Output files never overwrite anything (`_2`, `_3`).
  With "compare when finished" the Analyze tab opens and compares the source
  with the result.
- Every codec, mode and option value in the table is encoded once by the test
  suite with the real ffmpeg (38 variants), so no option in the panel is a
  button that fails.
- `CodecDelta.bat` launcher; drop files on it to open them.

## [0.1.0] - 2026-09-28

First usable version: a desktop window that compares two encodes of the same
recording, or checks a single lossless file, and explains its verdict.
Everything in this list was measured before it was kept; the measurements and
the heuristics that were removed are in `docs/DECISIONS.md`.

### Added
- **Analyze window** (`python -m app [reference] [test]`): drag-and-drop file
  slots with a track picker for multi-track containers, compare and verify
  actions that run in the background and can be cancelled, a results panel
  with the verdict, a summary, a per-band table and chart, a noise-to-mask
  spectrogram over time, and on-demand placement on an Opus bitrate ladder.
  English and Turkish; engine messages are translated too.
- **Alignment** that finds a short excerpt inside a long recording, detects
  PAL/NTSC transfers and clock drift, handles inverted polarity and swapped
  channels, and refuses to measure pairs that are not the same recording.
- **Single-pass comparison** with per-band codec SNR for mid and side, a split
  of the difference into linear (EQ, level) and codec-noise parts, and a
  measurement floor measured on every run rather than assumed.
- **Audibility**: an ERB masking model and noise-to-mask ratio, and a verdict
  from an anchor ladder built from the user's own reference ("equivalent to
  Opus at roughly 128 kbps").
- **Referenceless verification** of lossless files, calibrated on four albums
  and 231 files: 21/21 real sources and 34/34 full tracks judged consistent
  with lossless, 163/168 non-transparent transcodes caught, none falsely
  accused. The margin between the classes is narrow and documented.
- Bitstream readers for Ogg/Opus, Vorbis, FLAC and MP3; ffmpeg discovery,
  sample-exact streaming and window reads; ABX statistics; settings and
  privacy scrubbing.

### Changed
- Licence is GPL-3.0-or-later (was MIT).
