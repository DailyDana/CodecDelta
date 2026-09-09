# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x`, the API may break between minor versions.

## [Unreleased]

### Added
- Repository scaffold: MIT license, README, ruff/mypy strict/pytest configuration
  and GitHub Actions CI on Windows, plus a test that keeps Qt out of the engine
  layers so the analysis core stays headless-testable.
- `core.ffmpeg_locate` — finds an ffmpeg/ffprobe pair, rejecting any directory
  that lacks ffprobe, and caches probed capabilities per binary.
- `core.ffmpeg_runner` — the single place subprocesses are started: never
  `shell=True`, stderr always drained, cancellation kills the process tree.
- `core.ffmpeg_stream` — raw PCM streaming that only ever uses `readinto`, with
  resampling that always states its cutoff explicitly.
- `core.probe` — ffprobe inspection that never decodes video, distinguishes the
  audio ordinal from the container index, and resolves bitrate in three tiers
  without ever guessing.
- `core.stats` — closed-form ABX statistics: exact binomial, Wilson intervals,
  power, Šidák correction and a Wald SPRT.
- `core.settings` — defensive `%APPDATA%` preferences with clamping.
- `core.privacy` — path and identity scrubbing so reports are shareable by
  default, with an `audit()` backstop.
