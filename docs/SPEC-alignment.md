# Specification — sub-sample alignment

Contract for `app.align.refine`. Written before the implementation it describes,
and every number in it comes from a measurement recorded in
[DECISIONS.md](DECISIONS.md).

## Purpose

Given a reference and a test signal already aligned to the nearest whole sample,
report the remaining fractional delay and how much that answer can be trusted.

Everything downstream — the difference signal, per-band SNR, the audibility
verdict — is computed after shifting by this number. Half a sample of residual
error is 80 degrees of phase error at 10 kHz, which appears in the difference
signal as something indistinguishable from codec noise. A wrong delay does not
degrade the analysis; it invalidates it while still looking plausible.

## Inputs

| | |
|---|---|
| `reference`, `test` | one-dimensional, finite, float arrays at the same sample rate |
| `coarse_lag` | integer sample offset from `gccphat.estimate` |

Contract: after `coarse_lag` is applied the remaining delay is strictly within
±1 sample. The estimator searches only that range and cannot recover from a
wrong integer lag — see *Known limits*.

Malformed input (two-dimensional array, NaN, ±inf) raises `ValueError`. This is
a calling error, not a data condition. Degenerate but finite input (silence,
constant DC) is a data condition and yields `status="invalid"`.

## Estimator

One estimator: **residual minimisation**. The reference is shifted by candidate
fractional delays and the delay maximising the inner product against the test is
found by golden-section search, which is equivalent to minimising residual
energy once gain is optimally chosen at each candidate.

There is deliberately no second delay estimator and no branch that could select
one. Measured across 12,600 synthetic trials plus real music and real codec
pairs, residual minimisation failed 0.00% of the time on every dataset, while
the phase-slope alternative failed 5.67% to 79.17% depending on material. The
selection branch that preferred phase on agreement was the original P0 defect;
removing the branch, rather than fixing its condition, is what makes the defect
structurally unable to return.

`phase_slope_delay` remains in the module as a documented, tested primitive with
its failure characteristics recorded. It is not called by `refine`.

## Validity

Two measures accompany every delay:

**`correlation`** — signed normalised correlation of the pair after shifting by
the reported delay. Answers *are these two signals actually related*.

**`sharpness`** — residual energy half a sample either side of the optimum,
divided by residual energy at the optimum. Answers *is this delay well
localised*. Bounded below by 1 by construction.

### What was measured

| Case | correlation | sharpness |
|---|---|---|
| lossless round-trip | 1.0000 | 495384 |
| MP3 320k | 1.0000 | 374.9 |
| MP3 128k | 0.9998 | 9.8 |
| Opus 128k | 0.9986 | 2.1 |
| AAC 96k | 0.9997 | 5.2 |
| Opus 32k | 0.9895 | 1.1 |
| real music + 24 dB noise | 0.9980 | 2.4 |
| real music + 6 dB noise | **0.8930** | 1.0 |
| flat spectrum cut to 0.25 of Nyquist | 0.7858 | 1.3 |
| half the signal unrelated | 0.4854 | 1.1 |
| delay outside the search range | −0.2311 | 1.0 |
| unrelated white noise | 0.0035 | 1.0 |

**Sharpness does not discriminate.** Valid cases reach as low as 1.02 and invalid
cases as high as 1.131; the ranges overlap completely. An earlier six-order-of-
magnitude separation was an artefact of comparing a signal with itself, where
the residual goes to zero. Sharpness is reported because it is informative about
how sharply the delay is defined, but nothing is gated on it.

**Correlation discriminates.** Valid material stays at or above 0.893 even at
6 dB SNR and 32 kbps; invalid material stays at or below 0.485.

## Status

| Status | Meaning | `delay` |
|---|---|---|
| `ok` | delay computed, correlation above threshold | finite |
| `weak` | delay computed, correlation below threshold — the two signals may not be the same recording, or the relationship is not a pure time shift | finite |
| `invalid` | no usable delay | NaN |

`invalid` arises when the window is too short, either signal has no energy, or
the search converged onto its boundary. A boundary result is rejected rather
than returned: after correct integer alignment the remaining delay is below 1 by
construction, so a value at the limit means either the caller violated the
contract or the objective was flat. The previous implementation returned exactly
+1.0 in these cases, marked as verified.

`weak` still carries a delay. The value may well be correct — in the EQ-tilt and
lowpass cases it was accurate to 0.0003 samples — but a low correlation means a
difference measurement built on it will not be meaningful, and callers must say
so rather than reporting an SNR.

## Threshold

`MIN_ALIGNMENT_CORRELATION` lives in `app/align/thresholds.py`, the single home
for numbers awaiting calibration. It sits in the measured gap between 0.485 and
0.786.

The sample of invalid cases is small — four constructed scenarios — so this is a
placeholder with evidence, not a calibrated value. It should be re-derived from
a labelled set of real pairs, as `tools/calibrate_match.py` is meant to do.

## Known limits

**Capture range is ±1 sample.** A wrong `coarse_lag` is not always detectable:
the autocorrelation's sidelobe can create a local optimum inside the search
range, so the result is not necessarily clamped. Measured: a true delay of 1.6
returned −0.192 with no boundary hit. Correlation catches it (−0.23 against
≥0.89 for valid material) but only through the threshold, not by construction.
The integer stage is responsible for not violating the contract.

**A periodic signal offset by one period** passes every check here — correlation
and sharpness are both high because the alignment genuinely is good, just at the
wrong period. Detecting it belongs to the integer stage, where GCC-PHAT's
peak-to-sidelobe ratio separates matched from unmatched pairs by roughly 200×.

**Alignment is global and single-channel.** The module operates on
one-dimensional arrays; per-channel delay differences are outside its scope and
must be handled by the caller.

## Cost

Measured on this machine, per call:

| n | residual search | validity | total |
|---|---|---|---|
| 16,384 | 11.3 ms | 2.4 ms | 13.7 ms |
| 65,536 | 42.3 ms | 11.4 ms | 53.7 ms |
| 262,144 | 257 ms | 51 ms | 308 ms |

With twelve anchors a full alignment pass costs roughly 165 ms at 16k windows.
