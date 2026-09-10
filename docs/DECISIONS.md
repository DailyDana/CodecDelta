# Decisions

Choices that are not obvious from the code, each with the measurement behind it.
Most live as docstrings next to the code they govern; this file collects the ones
that have no natural home — chiefly the things deliberately **not** built, and the
heuristics that were removed before they shipped.

If you are about to add something here that was already removed, read the entry first.

---

## Rules that must not be relaxed

### `readinto`, never `read(n)`, on an ffmpeg pipe

Reading a 211 MB f32le stream on Windows, same file, same process:

| Method | Time |
|---|---|
| `stdout.read(256 KB)` | 2.06 s |
| `stdout.read(2 MB)` | 7.12 s |
| `stdout.read(16 MB)` | **117.60 s** |
| `readinto(memoryview)`, any size | ~2.1 s |

A 57× cliff, in the direction opposite to intuition. The root cause was not
confirmed — most likely the resize after a partial read — but the behaviour
reproduces, so the rule is absolute. `tests/test_ffmpeg_stream.py` holds a
30 MB/s floor against regression.

Lives in `app/core/ffmpeg_stream.py`.

### Resampling always states its cutoff

ffmpeg's default soxr silently walls off the top of the band. Measured on a
44.1 → 48 kHz **upward** resample, where no information need be lost at all:

| cutoff | 20.8k | 21.0k | 21.5k | 21.8k |
|---|---|---|---|---|
| default | −1.2 dB | −5.4 | **−30.4** | −73.9 |
| 0.99 | −0.2 | −1.4 | +1.7 | +1.1 |

Left at the default, the transcode detector would flag a clean FLAC as "cut at
20.9 kHz". Every resample therefore passes
`aresample=<rate>:resampler=soxr:precision=28:cutoff=0.99`, and file-local
measurements (cutoff, transcode evidence, bit entropy) run at native rate with
no resampler at all.

### A directory without ffprobe is not an ffmpeg installation

`C:\Program Files (x86)\MPV Player\ffmpeg.exe` is on PATH and built
`--disable-ffprobe`. `shutil.which("ffmpeg")` returns it. Discovery therefore
rejects any candidate directory that does not also contain ffprobe, and `which`
is never used alone. Verified: the winget build is accepted, MPV is rejected.

### Ogg CRC is sampled, and "not checked" is not "intact"

Verifying every page of a 10 MB `.opus` took 1393 ms — 100% of the scan cost,
since decoding the headers alone takes 1 ms. Sampling the first 16 pages plus
every 64th brought a full scan to 137 ms with identical results.

Resync still verifies every candidate: a stray `OggS` inside audio data is a real
occurrence and CRC is the only thing that rejects it. `Page.crc_ok` is tri-state
so a report can never turn "not checked" into "verified intact".

---

## Removed heuristics

### FLAC blocksize does not identify an encoder

A rule was written that read blocksize 4096 as the reference encoder and 4608 as
ffmpeg. The same ffmpeg build, same version, produced three different values:

| Source | Blocksize |
|---|---|
| lavfi source directly | 1024 (the input filter's frame size) |
| a wav file | **4096** |
| `-frame_size 4608` | 4608 |

4096 is precisely the value attributed to the reference encoder, so the rule
would have labelled most modern ffmpeg output as a reference rip. Blocksize is
now reported as data — an unusual value such as 1024 is still worth seeing — and
the encoder comes from the vendor string alone.
`tests/test_flac.py::test_blocksize_is_data_not_a_fingerprint` exists only to
stop this coming back.

### ffmpeg does not write a real LAME tag

The LAME tag's lowpass field records the cutoff the encoder applied, which would
answer "where was this cut" without measuring anything. On ffmpeg output it is
unavailable. Raw bytes from a `libmp3lame` encode:

```
4c 61 76 63 36 33 2e 37 2e | 00 00 ... 00 | 24 03 a8
<-- encoder "Lavc63.7." -->  <-- +9..+20 --> delay 576, pad 936
```

The tag is in the right place (Xing + 0x78) and delay/padding are correct — 576
is LAME's own standard delay, which confirms the offset chain — but every field
from +9 to +20 is zero. So lowpass survives only in files from genuine LAME
binaries, and a zero must never be read as "cut at 0 Hz".

Still unverified: no real LAME encoder exists on this machine, so a populated
lowpass field has only been exercised against a synthetic tag.

### Parabolic interpolation cannot cross-check a delay estimate

Sub-sample alignment was to be measured two ways so the methods could confirm
each other. Parabolic interpolation on the correlation peak was the second
method until it was measured. True delay 0.30 samples:

| Method | Estimate |
|---|---|
| parabolic on the PHAT correlation | 0.020 |
| parabolic on a plain correlation | 0.216 |
| phase slope | 0.29999 |

Worse, the bias is signal-dependent. For the same 0.30 delay, parabolic on a
plain correlation gave 0.284 when the signal was limited to 0.20 of Nyquist and
0.0998 when it reached 0.48. A parabola does not fit a sinc-like peak, and the
peak's width moves with the signal's bandwidth, so the error cannot be
calibrated away. A measurement whose bias depends on the input cannot serve as
independent verification of another measurement.

Replaced by a golden-section search that directly maximises the inner product of
the shifted reference against the test, which is equivalent to minimising
residual energy — the quantity the whole pipeline is about. It agrees with the
phase slope to better than 0.01 samples.

---

## Deliberately not built

### No AAC bitstream parser

A bitstream reader earns its place by surfacing what ffprobe cannot. Measured per
format:

| Format | What the bitstream layer adds beyond ffprobe | Value |
|---|---|---|
| Opus | mode / bandwidth / frame-duration histograms, pre-skip, packet sizes | High — none of it in ffprobe, and bandwidth cross-checks the measured cutoff |
| FLAC | STREAMINFO MD5, blocksize, vendor, compression ratio, block inventory | High |
| MP3 | LAME lowpass, VBR method, encoder delay/padding, Xing vs Info | Medium-high |
| Vorbis | nominal bitrate, blocksizes | Low |
| **AAC** | — | **~zero** |

ffprobe on a real YouTube `.m4a`:

```
mime_codec_string = mp4a.40.2      audio object type: LC (HE would be .5, PS .29)
profile           = LC
nb_frames         = 29460
bit_rate          = 127999
initial_padding   = 0
handler_name      = "ISO Media file produced by Google Inc."
encoder           = Lavf62.12.102
```

`mp4a.40.2` is the decisive line. SBR/PS presence changes how a spectrum must be
read — above the SBR crossover the band is synthesised rather than waveform-coded
— and it was assumed only a bitstream parser could reveal it. ffprobe already
decodes the AudioSpecificConfig object type. `handler_name` even supplies
provenance, the AAC analogue of a vendor string.

Against that, an MP4 atom walker plus an ADTS frame parser is the largest parser
of the set, reproducing information already available.

**AAC files are already fully supported.** Probe and decode were verified on a
real 684 s YouTube m4a: 1242× realtime. What is deferred is a redundant parser,
not the format.

Unverified: HE-AAC could not be produced locally (this build has
`--disable-libfdk-aac`, and `aac_mf` rejects the `aac_he` profile), so ffprobe's
object type reporting for SBR streams rests on the specification rather than a
measurement. Low practical risk — YouTube serves 128k AAC as LC.

This would need revisiting if ffprobe's `mime_codec_string` proves wrong for
HE-AAC, if per-frame size distribution turns out to be needed, or if `iTunSMPB`
in Apple-sourced files breaks alignment.

### No scipy, no matplotlib

FFT is about 1% of the analysis budget — 0.16 s for a 10-minute track — so
scipy's float32 advantage is real but irrelevant; a cp314 wheel exists if
profiling ever disagrees. Plotting goes through pyqtgraph, numpy → QImage and
hand-written inline SVG rather than a 60 MB theme-locked dependency, and the same
data model feeds both the screen and the report.

---

## Corrections to the plan

Planning notes claimed ABX power of ~0.45 at n=16 against 75% discrimination, and
n≈40 for 80% power. Both are wrong. Computed exactly:

| n | threshold | power |
|---|---|---|
| 16 | 12 | 0.630 |
| 20 | 15 | 0.617 |
| 24 | 17 | 0.766 |
| 30 | 20 | 0.894 |
| 40 | 26 | 0.946 |

80% power arrives around n = 26. Power is also not monotonic in n — n = 20 scores
below n = 16 — because the binomial threshold is an integer and 15/20 is a harder
bar than 12/16. The UI shows these numbers before a test starts.
