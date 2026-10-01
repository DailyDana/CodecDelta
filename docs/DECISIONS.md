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

### Sub-sample delay is selected by residual search, never by phase slope

An adversarial audit measured both estimators over 12,600 synthetic trials plus
real music and real codec pairs. Phase-slope failure rate (error > 0.05 samples):

| Dataset | phase | residual |
|---|---|---|
| band-limited noise, 24 dB SNR | 5.67% | 0.00% |
| sparse/tonal spectrum, **no noise at all** | 31.67% | 0.00% |
| real music, **no noise at all** | 22.92% | 0.00% |
| real music vs lowpassed copy (the tool's actual input) | **79.17%** | 0.00% |
| real FLAC vs Opus/MP3/AAC/Vorbis pairs | 6 of 6 | 0 of 6 |

Errors reached 122 samples against a true delay of 0.30.

The cause is `np.unwrap`, not noise. It runs over every bin in the fitted band;
a single low-magnitude bin flips the branch by 2π and every subsequent bin
inherits the offset. Magnitude weighting reduces that bin's own contribution but
cannot undo the corruption of the good bins after it. Sparse spectra therefore
fail at infinite SNR — and tonal music is sparse: for a tonal signal the top 1%
of bins carry 78.6% of the fit weight against 6.2% for white noise.

Filtering low-magnitude bins before unwrapping was tried and rejected. It fixes
the tonal and lowpass cases but fails under spectral dropout — errors of 150 to
400 samples — because removing bins destroys the adjacency that unwrap depends
on. Any unwrap-based estimator is structurally fragile for this input class.

Selection therefore has no branch: when the residual search produces a value it
is always returned, even when the two agree. Preferring phase on agreement was
the original defect, and leaving that branch in place is the only way it could
return. Measured after the change: 0.00% failure on every dataset above.

The phase slope is kept purely as a cross-check. `agree` was measured as a
detector at 99.97% capture with 0.00% false alarms, which is why the fix is
cheap: the information needed was already being computed correctly and simply
discarded.

Note for later: on real codec pairs the status is now `disagree` almost always,
because the phase estimator really is wrong there. Until it is repaired the flag
carries little information for the tool's primary use case.

### Alignment confidence is a validity measure, not a second estimator

Replacing the broken phase-slope cross-check raised the question of what should
take its place. Three candidates were measured on identical data: an unwrap-free
circular phase fit, split-band residual searches, and a validity measure that is
not a delay estimator at all.

All three tracked the truth equally well, so accuracy did not decide it. What
decided it was the case a cross-check exists for. Given two unrelated signals:

| | delay returned | correlation |
|---|---|---|
| residual search | −0.3723 | — |
| circular phase fit | −0.6029 | — |
| validity measure | — | **0.0035** |

A delay estimator always returns a number. Two of them return two different
plausible numbers, both wrong. Only a validity measure can answer whether there
is a delay to find at all, which is the actual failure mode — unrelated
material, a delay outside the search range, or a relationship that is not a pure
time shift.

The validity measure is also the cheapest of the three: 2.4 ms against 8.0 ms
for the circular fit and 23.0 ms for split-band, at a 16k window.

### Sharpness does not discriminate — an earlier reading was an artefact

Peak sharpness — residual half a sample off the optimum over residual at the
optimum — first appeared to separate valid from invalid by six orders of
magnitude, 9.6 million against 1.03. That measurement compared a signal with
itself, where the residual goes to zero and the ratio explodes. It does not
generalise.

Measured on real inputs:

| | sharpness | correlation |
|---|---|---|
| valid (lossless … Opus 32k, down to 6 dB SNR) | **1.02** – 495384 | **0.893** – 1.0000 |
| invalid (unrelated, out-of-range delay) | 1.000 – **1.131** | −0.231 – **0.485** |

The sharpness ranges overlap completely: Opus at 32 kbps gives 1.1 and MP3 at
64 kbps gives 1.6, while a deliberately unrelated pair gives up to 1.131.
Correlation separates cleanly instead, and it is what the status is derived
from. Sharpness is still reported because it describes how sharply the delay is
defined, but a test exists specifically to stop a threshold being attached to it.

The threshold lives in `app/align/thresholds.py` with its evidence, and is
marked as a placeholder: the invalid sample is four constructed cases, not a
labelled set of real pairs.

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

## Zarf korelasyonu ortusme sayisina degil, Pearson'a bolunur

Kaba (L1) hizalamada capraz korelasyonu her gecikmedeki ortusme SAYISINA bolmek
akla yatkin ve yanlis. Az ortusen gecikmelerde bolen kucuktur, gurultu
boyutlandirilarak buyur ve gercek tepeyi gecer.

Olculdu -- 30 s'lik bir kayitta 18.3 s'den baslayan 4 s'lik kesiti aramak,
40 farkli tohum:

| Normalizasyon | Dogru gecikme | En buyuk hata |
|---|---|---|
| ortusme sayisi (`raw / counts`) | 20/40 | **32.25 s** |
| ortusen bolgenin Pearson'i | **40/40** | 0.00 s |

Ayrica tepe secerken **mutlak deger alinmaz**. Dalga formunun aksine bir enerji
zarfinin polaritesi yoktur; negatif korelasyon "ters cevrilmis" degil
"eslesmiyor" demektir. Basarisiz 20 vakanin yarisinda kazanan gecikmenin rho'su
negatifti (en dusuk -1.42, ki bu ayni zamanda kuresel z-skorunun kismi
ortusmede sinir disina tastigini da gosteriyordu -- Pearson artik ortalamayi ve
olcegi ortusen bolgeden hesapliyor ve +-1 ile sinirli).

## Zarf duzeyinde PSR ayirt etmiyor, o yuzden ariza raporlanmiyor

Plan L1 icin `PSR > 20` bekliyordu. Ayni 40 tohumda olculen:

| | ILGILI | ILGISIZ |
|---|---|---|
| `rho` | 0.949 .. 0.977 | 0.058 .. 0.458 |
| `PSR` | 2.385 .. 6.393 | 1.877 .. 7.097 |

`rho` temiz ayiriyor, PSR araliklari TAMAMEN ortusuyor. Sebep yapisal: zarf
duzgun bir sinyaldir, komsu gecikmeler neredeyse tepe kadar iyidir, dolayisiyla
yan lob medyani hicbir zaman dusmez. PSR yalnizca GCC-PHAT'in beyazlatilmis
keskin tepesinde anlamlidir ve orada kaliyor.

`CoarseMatch.psr` alani kaldirildi. Ayirt etmeyen bir sayiyi raporda tasimak,
kullaniciya kanit gibi gorunen bir sey vermek olurdu. Bu, `refine`'da
"keskinlik ayirt etmiyor" bulgusuyla ayni desen.

## PAL olculmez, SINANIR

Capa tabanli surukelenme olcumunun yakalama araligi fiziksel bir sinirla
bagli: bir pencerede biriken kayma `window * (ratio - 1)` ornektir ve icerigin
periyodunun yarisini astiginda GCC-PHAT tepesi dagilir. OLCULEN (8 kHz mono,
300 s, 30 capa, `min_correlation=0.3`):

| sapma | 0.25 s pencere | 1.0 s | 2.0 s |
|---|---|---|---|
| 10 ppm | 0.1 ppm hata | 0.1 ppm | 0.0 ppm |
| 100 ppm | 0.0 ppm | 0.0 ppm | 0.0 ppm |
| 1000 ppm | **1.0 ppm** | capa YOK | capa YOK |
| 5000 ppm | capa YOK | capa YOK | capa YOK |
| 41667 ppm (PAL) | capa YOK | capa YOK | capa YOK |

Varsayilan pencere bu yuzden 2.0 s degil **0.25 s**. Yakalama araligi ~1000 ppm;
NTSC pulldown (1001 ppm) tam sinirda calisir.

PAL 41667 ppm'dir, yani araligin 40 kati disinda ve **dogrudan olculemez**.
Cozum orani daha hassas olcmeye calismak degil, soruyu degistirmek: PAL surekli
bir bilinmeyen degildir, kisa ve AYRIK bir tablodan gelir. O yuzden olculmez,
sinanir -- her aday oran icin test on telafi edilir ve capalar yeniden
toplanir; dogru hipotezde pencere ici kayma sifirlanir, capa sayisi ve
korelasyon birlikte yukselir. Kazanan hipotezin uzerindeki artik oran normal
yoldan olculur.

OLCULEN (`estimate_from_audio`, 300 s): surukelenme yok, 50 ppm, NTSC, PAL
hizlandirmasi ve PAL yavaslatmasi -- **besinde de hata 0.0 ppm**.

## Snap toleransi mutlak degil, oransal

Tabloya yakalanma toleransi once sabit `2e-4` idi. Yanlis: tablodaki 1.0001
girdisinin kendi sapmasi 1e-4'tur, yani +-2e-4'luk bir pencere surukelenmesi
OLMAYAN bir dosyayi (oran tam 1.0) "1.0001 zamanlama" diye etiketliyordu. Bir
test bunu yakaladi. Tolerans artik sapmanin %2'si (taban 1e-5), yani her girdi
kendi olceginde degerlendiriliyor.

## PHAT sessizlikte NaN uretiyordu

`phat_correlation`in bolme tabani GORELIDIR (`1e-12 * magnitude.max()`), ki bu
sinyal olcegi degistiginde davranisin degismemesi icin dogru karar. Ama girdi
tamamen sessizse taban da sifir olur ve bolme `0/0 = NaN` verir; NaN
korelasyondan gecikmeye, oradan surukelenme fit'ine yayilir. `drift`in sessiz
bir capa penceresine denk gelmesiyle gercekten gozlendi.

Ayrica tumu sifir olan bir korelasyon dizisinde `argmax` ilk elemani, yani
`-limit` gecikmesini seciyordu -- "bilgi yok" durumu icin uydurulmus ve
tamamen yaniltici bir cevap. Ikisi de duzeltildi: sessizlikte `lag=0`,
`correlation=0.0`.

## Zarf bir kapi degil, bir ipucu

Zarf korelasyonu dusukse bu tek basina "farkli kayit" demek degildir. Dinamigi
duz icerikte zarf yalnizca cerceve enerjisinin rastgele dalgalanmasidir ve
bilgi tasimaz. Iki yerde olculdu:

- `drift.estimate_from_audio` hipotezleri once zarf korelasyonuna gore
  eliyordu. Duragan test sinyalinde DOGRU hipotez de elendi ve 50 ppm ile NTSC,
  capalar tek basina kusursuz calisirken, "bilmiyorum" dondu. Artik zarf
  yalnizca kaba gecikme kaynagi; bilgisizse sifira dusulur, karari capalar verir.
- Ayni icerik + 3 dB S/N bagimsiz gurultu: zarf rho **0.475** (esik 0.70), hizali
  dalga formu korelasyonu **0.818**, gecikme 800.002 ornek (dogrusu 800). Zarfa
  guvenen bir plan bunu "farkli kayit" diye reddederdi.

`plan.build` "farkli kayit" hukmunu ancak zarf VE capa tabanli kanit BIRLIKTE
basarisiz oldugunda verir.

## Yalnizca olculemeyen oranlar sinanir

Tablodaki 1.0001 girdisi capalarin yakalama araligi icinde (100 ppm < 1000 ppm),
yani 1.0 hipotezinin artigindan zaten olculur. Ayri bir hipotez olarak da
sinandiginda gurultude 1.0 ile neredeyse berabere skor aliyor ve rastgele
kazaniyordu. 0 dB S/N'de 24 denemenin birinde **0.06 ppm "surukelenme"**
raporlandi -- hipotez yolu kendi siniflandirmasini yapip "ihmal edilebilir"
kuralini da atliyordu.

Iki duzeltme: sinanan hipotezler 1.0 ve yakalama araligini asan tablo
oranlariyla sinirli (NTSC 1001, PAL 41667 ve tersi); ve her yol ayni
siniflandiricidan (`_classify`) geciyor. Duzeltmeden sonra ayni 24 denemede
sahte surukelenme 0; bes tablo senaryosunda hata yine 0.0 ppm.

## Kucuk saat kaymasi "farkli master" degildir

`DriftEstimate.is_resampling` once 5 ppm ustundeki HER orani "ham S/N
gosterme" sinifina koyuyordu. Plan bunu ayiriyor: PAL/NTSC (ya da yakalama
araligini asan her oran) yeniden orneklenmis, PAL'de perdesi kaymis baska bir
master'dir ve codec farki olculemez. Birkac 10 ppm'lik etiketsiz saat kaymasi
ise ayni icerigin iki farkli saatle calinmasidir; global yeniden ornekleme
yapilmaz, gecikme blok-yerel izlenir (`needs_tracking`).

## `-ss` ile arama her formatta ornek-dogru degil

Hizalama plani gecikmeyi `-ss` ile okunan pencerelerden olcer, ana gecis ise
dosyayi bastan cozer. Iki zaman cizgisi arasindaki her fark dogrudan gecikme
hatasidir. OLCULEN (pembe gurultu, 0.5 s on okuma, tam cozumle karsilastirma):

| Kap / codec | Sonuc |
|---|---|
| WAV/PCM, FLAC, Ogg Opus | bit-exact |
| MP3 (Xing'li ve Xing'siz VBR, 40 s ve 25 dk) | bit-exact |
| M4A/AAC | sapma KONUMA gore 70..820 ornek |
| Ogg Vorbis | sabit -128 ornek |
| WebM Opus | sabit -48 ornek (Matroska damgasi ms hassasiyetinde) |
| MKA FLAC | +-1 ornek titresim |

On okuma olmadan MP3 ve Opus'ta pencerenin ilk ~2000 ornegi de bozuk
(kodlayicinin arama sonrasi isinmasi, max fark 0.30).

Hizli yol bu yuzden bir BEYAZ liste: yalnizca olculmus (kap, codec) ciftleri.
Gerisi, bilinmeyenler dahil, dosyayi bastan cozup pencereye kadar atar --
yavas ama tanim geregi dogru. YouTube sesinin tipik kabi (WebM Opus) guvenli
yoldan gecer.

Yeniden orneklemede ikinci bir kosul var: arama noktasi iki hizin ortak
izgarasina (`1/gcd(giris, cikis)` s; 44.1/48 icin 1/300 s) oturmazsa soxr
farkli bir fazdan baslar. 13.3712 s'den okuma r = 0.9796 verdi. Okuyucu arama
noktasini izgaraya yuvarlar ve farki kirpar. Sozlesme testi 7 format x 2 hiz
x izgara ici/disi baslangiclarda `np.array_equal` istiyor; izgara kurali
kapatildiginda yeniden ornekleyen hizli yol testleri dusuyor (dogrulandi).

## Spektrumda kesirli gecikme: taban -67 dB

Blok-yerel gecikme takibi kesirli gecikmeyi STFT cercevelerine faz rampasi
olarak uygular (`stft.phase_shift`). Bu yaklasiktir -- pencere kaymaz, yalnizca
icerik kayar -- ve hatasi bir olcum tabani olusturur. Ilk yazdigim docstring
"ihmal edilebilir" diyordu; olcum bunu yalanladi. OLCULEN (4096'lik cerceve,
gercek kesirli kaydirilmis sinyalin STFT'sine gore):

| bant (Nyquist orani) | 0.25 ornek | 0.5 ornek |
|---|---|---|
| 0 .. 0.99 | -73 dB | -67 dB |
| 0.99 .. 1.0 (Nyquist haric) | -23.5 dB | -20.5 dB |
| son 4 bin (Nyquist haric) | -16.2 dB | -13.2 dB |
| Nyquist bini | -0.1 dB | +2.9 dB |
| beyaz gurultu, tum bant | -32 dB | -29 dB |

Beyaz gurultude tum bant hatasi (-29 dB) codec gurultusu gibi gorunecek kadar
buyuk, ama tamamini en ust birkac bin belirliyor; orada kesirli kayma
tanimsiz (gercek bir sinyalin Nyquist bileseni gercek olmali). Gercek ses bu
bantta enerji tasimaz ve bant zaten yeniden orneklemenin tabani yuzunden
"olculemez". Asil sonuc: kesirli gecikme telafi edildiginde ~64 dB ustundeki
S/N olculemez. Kalibrasyon referansi ayni yoldan gecirdigi icin bu taban
raporda satir satir gorunecek.

## Olcum tabani, olctugu sayiyla ayni metrikle olculur

Taban modulu ilk surumde duz farki (`|B - A|^2`) kullaniyordu, gerekcesi
"resampler'in genlik egimi de hatadir" idi. Yanlis: tabanin karsilastirildigi
mansettaki S/N INKOHERENT'tir; gecis bandi dalgalanmasi gibi dogrusal etkiler
oradan zaten ayiklanir. Farkli metrikle olculen taban 18-20 kHz'de 75.5 dB
verdi, inkoherent 120.7 dB. Birinci sayi olculebilir bantlari "olculemez"
diye isaretliyordu.

Duzeltilmis taban (FLAC 44.1 -> 48 -> 44.1, 30 s kesit):

| bant | 1-4 k | 4-8 k | 8-12 k | 12-16 k | 16-18 k | 18-20 k | 20-21 k |
|---|---|---|---|---|---|---|---|
| taban (dB) | 149.8 | 147.6 | 145.6 | 143.9 | 135.1 | 120.7 | 112.1 |
| plan olcumu | 148 | | | 140 | | 118 | 50 |

Plandaki 20-21 kHz satiri (50 dB) yeniden uretilemedi; o olcumun yontemi
kayitli degil. Guncel sayi tekrarlanabilir olan.

## Yeniden ornekleyen zincirde bantlar soxr kesiminde biter

Iki dosya farkli hizdaysa soxr `cutoff * Nyquist` ustunu tanim geregi
gecirmez (0.99 x 22050 = 21.83 kHz). Bant duzeni Nyquist'e kadar gidiyordu ve
gercek bir FLAC/Opus ciftinde 22.00-22.05 kHz bandini, tabani -11 dB iken,
"olculebilir" gosterdi: `S/N < taban - 3` kurali yalnizca "S/N'e inanmak icin
fazla yuksek" yonunu korur. Bant duzeni artik kesime kirpiliyor; yeni bir
sayi gerekmedi, `ResampleCfg.cutoff` zaten belgelenmis.

## Ilk gercek sonuc: planlama oturumunun elle analiziyle ortusuyor

Loreena McKennitt, "Beneath a Phrygian Sky": FLAC 44.1/16 ile YouTube Opus
(~141 kbps), 9.5 dk, 8.7 s'de:

| | elle analiz (planlama) | boru hatti |
|---|---|---|
| gecikme | 0 | -0.020 ornek |
| hizali r | ~0.998 | 0.9983 |
| genis bant S/N | ~24 dB | duz 24.33 dB, inkoherent 25.62 dB |
| kazanc | 0.0 dB | -0.01 dB |

Side kanali mid'den belirgin kotu (4-8 kHz: 15.3 dB'e karsi 5.4 dB) --
joint-stereo'nun beklenen izi. 20-21.83 kHz'de -10.3 dB Opus'un kendi
kesimi; o bantta taban 91.8 dB oldugu icin zincirden gelmiyor.

## Saat kaymasi: cerceve basina hizalama yetmez, egim yinelemeyle bulunur

Kayipsiz, bilinen kaymali kopya (pembe gurultu, 90 s; `asetrate` hizi tamsayiya
yuvarladigi icin 44102 -> 45.35 ppm, 44109 -> 204.08 ppm), genis bant codec S/N:

| yontem | 45 ppm | 204 ppm |
|---|---|---|
| izlemesiz (orta noktada sabit gecikme) | -4.5 dB | -11.3 dB |
| cerceve basina gecikme, plan noktalarindan dogru | 33.4 | 20.4 |
| surekli warp, plan noktalarindan dogru | 41.8 | 28.4 |
| surekli warp + yinelemeli egim | **72.8** | **89.5** (taban 88.2) |

Uc ayri sebep, uc ayri duzeltme:

1. **Cerceve ici kayma.** 4096'lik cercevede gecikme 45 ppm'de 0.2, 204 ppm'de
   0.8 ornek degisir. Referans artik her ornek icin kendi konumundan
   ornekleniyor (`dsp/warp.py`: Kaiser-sinc, 64 tap, 8192 faz). 1024 fazda
   hata tap sayisindan bagimsiz -66 dB'de takiliyordu -- sinirlayan faz
   cozunurluguydu; 8192 faz ayni maliyette -82..-87 dB.
2. **Yanli egim.** Egim asiri hassas olmali: 204 ppm'de binde birlik hata 60
   s'de 0.5 ornek eder ve kayipsiz S/N'i 166 dB'den 12 dB'e dusurur (sentetik).
   Plan noktalari kayan pencerelerde olculur; olculen gecikme pencere
   merkezinin degil ENERJI AGIRLIKLI konumun gecikmesidir. Plan noktalarindan
   gecen dogru -204.126 ppm verdi (gercek -204.082); uclarda 0.12 ornek. Artik
   mevcut modelle warp edilmis 1 s'lik pencerelerde kalan gecikme olculup
   dogruya ekleniyor; warp sonrasi pencere ici kayma ihmal edilebilir, olcumler
   yansiz.
3. **Hiz.** Yinelemeler pencereleri her seferinde ffmpeg ile yeniden okuyordu
   (90 s'lik dosyada 21-31 s). Pencereler bir kez payla okunuyor: 14.5-16 s.

45 ppm'deki 72.8 dB'in kaynagi AYRISTIRILMADI. Dogru uyumu 0.001 ornek
sapmada ve yakinsama esigini 1e-3'ten 1e-5'e cekmek bir sey degistirmedi;
en guclu supheli ffmpeg'in 44102 -> 44100 donusumunun tam dogrusal olmamasi,
yani test verisinin uretimi. Codec gurultusu 20-40 dB'de oldugu icin pratik
etkisi yok, ama iddia edilmiyor.

Plan, "kucuk kayma 1e-9 hassasiyetinde olculemez, global yeniden ornekleme
yapma" diyordu. Olcum ilk yarisini yalanladi: tam hizda olculen noktalardan
gecen dogru o hassasiyete ulasiyor. Ama global yeniden ornekleme yine
ffmpeg ile yapilamiyor -- `asetrate` hizi tamsayiya yuvarliyor -- o yuzden
warp numpy'de.

## En ust %1 hicbir zaman raporlanmaz

44.1/44.1 bir ciftte (yeniden ornekleme yok, soxr kirpmasi devrede degil)
22.00-22.05 kHz bandi taban 3 dB iken "olculebilir" cikti. `S/N < taban - 3`
kurali yalnizca bir yonu korur. Bant duzeni artik her zaman
`min(cutoff, 0.99) x Nyquist`te bitiyor: kesirli gecikme en ust %1'de tanimsiz
(olculen, `stft.phase_shift`) ve bant duyulabilir aralikta degil.

## NMR: tonalite bant basina (MPEG tepe kurali), mutlak degere esik baglanmaz

Maskeleme modeli Johnston/MPEG-1 model 1 bicimi (ERB bantlari, Schroeder
yayilmasi, Terhardt ATH). Tonalite icin uc yontem ayni gercek FLAC/Opus
ciftinde (~141 kbps, dinlemede seffafa yakin, 9.5 dk) olculdu:

| tonalite | MPEG toplam NMR p50 | p95 | 0 dB ustu cerceve | en kotu bant p50 |
|---|---|---|---|---|
| kuresel SFM (cerceve basina tek sayi) | +5.8 | +10.2 | %91.4 | +14.2 |
| bant basina SFM | -9.2 | -6.8 | %0.0 | -2.8 |
| **MPEG tonal bilesen kurali (bant basina)** | **+0.8** | +8.2 | %57.5 | -- |

Ayni dosya, ayni gurultu, medyanda **15 dB yayilim**. Plan "NMR yuzde esikleri
dayanaksiz" demisti; bu onun olcumu. Hicbir mutlak NMR degerine hukum
baglanmaz; hukmu capa merdiveni verir. NMR'in savunulan tek ozelligi bitrate
ile MONOTON olmasi (merdiven testi sinar) ve spektrogramda farkin NEREDE
oldugunu gostermesi.

Neden tepe kurali: kuresel SFM'de Johnston'in tonal payi (14.5 + z, 8 kHz'de
35 dB) tum spektruma yayiliyordu -- MPEG model 1 tonal bilesenleri yerel
siniflandirir. Bant basina SFM ise bant genisligine bagli: 20 binlik bantta
saf bir tonun SFM'i Hann sizintisi yuzunden ~-34 dB'de takiliyor (Johnston'in
-60 dB hedefi binlerce binlik tam spektrum icin) ve saf ton "yari tonal"
(alpha 0.57) sayiliyordu. Tepe kurali (komsularini 7 dB asan yerel maksimum,
ana lob enerjisi tonal) 200 Hz'de de 8 kHz'de de saf tona >0.9 veriyor.
Bilinen zayifligi: 2-4 binlik en dar bantlarda tesadufi bir gurultu tepesi
tum bandi tonal yapabilir; o bantlar dusuk frekansta ve tonal pay orada kucuk.

Olcek kalibrasyonu: tam olcekli sinusun BANT ENERJISI 1.0 (tepe bini degil;
Hann ana lobu +1.76 dB veriyordu). ATH icin "0 dBFS = 96 dB SPL" varsayimi.

## Capa merdiveni kendi kendini dogruladi

Mansetteki hukum mutlak bir esikten degil, kullanicinin kendi referansindan
kodlanan bir merdivenden geliyor. Gercek FLAC ile YouTube Opus (kap 152 kbps,
gercek ~141; 9.5 dk), her basamak ~15 s:

| Opus kbps | codec S/N | NMR p50 | NMR p95 | 0 dB ustu cerceve |
|---|---|---|---|---|
| 64 | 19.62 dB | +7.2 | +13.8 | %94.0 |
| 96 | 23.15 | +3.8 | +10.8 | %78.5 |
| 128 | 25.65 | +0.8 | +8.8 | %57.7 |
| 192 | 28.66 | -3.2 | +4.2 | %27.1 |
| 256 | 30.93 | -7.2 | -0.2 | %4.9 |
| **YouTube** | **25.62** | +0.8 | **+8.2** | %57.5 |

Iki eksen de bitrate ile MONOTON (NMR'in savunulan tek ozelligi buydu) ve test
dosyasini ayni yere koyuyor: S/N ile ~128 kbps, NMR ile ~134 kbps. YouTube'un
Opus kodlayicisinin ffmpeg libopus 128k'siyla ayni bozulmayi vermesi, kabin
yazdigi 152'den dusuk ama makul: YouTube VBR hedefini ve kodlayici surumunu
aciklamiyor. Gorece kullanimda NMR'in mutlak ofseti sadelesiyor -- ayni
model iki tarafa da uygulaniyor.

Ara degerleme log2(bitrate) uzerinde dogrusal. Eksen monoton degilse konum
yine verilir ama "yaklasik" diye isaretlenir; iki eksen bir basamaktan fazla
ayrisirsa ikisi de soylenir, hangisinin dogru oldugu iddia edilmez.

## Referanssiz dogrulama: kesim medyani ayiriyor, IQR ve side ayirmiyor

Ozellikler etiketli sette olculdu (9 gercek CD parcasi, tek album, 60 s
kesitler; her birinden ffmpeg ile 13 transcode; toplam 126 dosya; tablo
`app/single/thresholds.py` icinde). Uc bulgu plani degistirdi:

1. **Gercek CD de dik bir diz gosteriyor.** 19.6 kHz'de 500 Hz icinde ~5 dB
   dusus; "dB/oktav"a cevrilince -110 gorunuyordu cunku 20 kHz'de 500 Hz
   0.035 oktavdir. Planin "dogal roll-off 6-18 dB/okt, codec 60-200" tablosu bu
   master icin gecersiz. Metrik "500 Hz'de dB dusus" oldu: gercek 4.6-10.2,
   MP3/Opus/Vorbis 32.7-69.2. Dik diz tek basina kanit degil.
2. **Cerceve kesim IQR planin TERSINE calisiyor.** Gercek 0/43/1669 Hz,
   codec'ler 65-2500 Hz. Gercek CD'de icerik Nyquist'e kadar var, kesim tepede
   sabit; codec'te sfb21 kuantalamasi kesimi cerceveden cerceveye kaydiriyor.
   Kanit olarak KULLANILMIYOR.
3. **Side kanali ayirt etmiyor.** ffmpeg kodlayicilari joint-stereo'da side'i
   cokertmiyor (gercek -2.7 dB medyan, codec'ler -2..-7). Kullanilmiyor.

En temiz ayirici cerceve kesim medyani (referans-55 dB): gercek >= 21.2 kHz,
seffaf olmayan her codec <= 20.2 kHz. Taban ve diz dususu bunu destekler ama
AAC-PNS tabani -24.6'ya kadar cikariyor (sentezlenmis gurultu): bilinen yanlis
negatif yonu. AAC 256 ve MP3 V0'in bazi parcalari uc ozellikte de gercekle
ayni: spektral olarak ayirt edilemez. Arac onlara "kayipsiz kaynakla tutarli"
der; dil kurali ("kanitlandi" degil "tutarli") bu yuzden var.

Sette KOYU gercek kayit yok (eski, bant sinirli). Boyle bir kayit dusuk kesim
gosterir; onu "belirsiz"e tasiyacak karsi-kanitlar (diz ustu icerik, yumusak
diz) yalnizca sentetik veriyle sinandi.

## Kalibrasyon seti 16 bit olmali; sikistirma orani blok boyuna bagimli

Ilk etiketli set 24 bitlik transcode'larla uretilmisti: ffmpeg kayipli
codec'i float cozer ve FLAC'a s32 yazar. Spektral ozellikler etkilenmez
(float yol) ama diz ustu taban -92..-128 gorunuyordu; 16 bitte kuantalama
gurultusu tabani **-64**'e cekiyor. Sahte bir FLAC hemen her zaman 16 bittir.
Esikler 16 bit tabloya gore yeniden konuldu: taban -60 -> **-55** (gercek min
-44.4, codec max -64.1), duvar 25 -> **18 dB/500 Hz** (gercek max 10.2,
codec min 20.7).

FLAC sikistirma orani (plan: gercek 0.55-0.70, kayipli kaynakli 0.40-0.55):

| kaynak | blok boyu | oran |
|---|---|---|
| gercek CD | 3600 | 0.37 / 0.53 / 0.61 |
| MP3 -> FLAC (ffmpeg varsayilan) | **47** | 0.81 / 0.96 / 0.97 |
| MP3 -> FLAC, `-frame_size 4096` | 4096 | 0.452 (tek parca) |
| Opus -> FLAC | 5010 | 0.35 / 0.50 / 0.58 |
| AAC -> FLAC | 1024 | 0.37 / 0.51 / 0.59 |
| Vorbis -> FLAC | 128 | 0.47 / 0.64 / 0.70 |

ffmpeg'in flac kodlayicisi dogrudan bir cozucuden beslendiginde blok boyunu
cozucunun paket boyundan aliyor; 47 ornekli bloklarla dosya neredeyse
sikismiyor. Blok duzeltilince planin yonu tutuyor ama fark zayif (0.45-0.50'ye
karsi 0.52). Oran KANIT DEGIL, rapor notu; 256'nin altindaki blok boyu ise
"cozucuden dogrudan kodlanmis" izi olarak not ediliyor -- hukme girmeden.
(Blok boyunun KODLAYICIYI belirlemedigi karari gecerli; burada iddia farkli:
47'lik blok bir kodlama YOLUNUN izi.)

## Dort albumle yeniden kalibrasyon: Nyquist'e yakin duvar kanit degil

Kullanici D:/music'e uc album ekledi (Radiohead "Amnesiac" CD 16/44.1,
"A Moon Shaped Pool" 24/48, Fleetwood Mac "Mirage" 2016 remaster 24/48).
Referanssiz dogrulayici once 34 tam parcanin 17'sine "belirsiz" dedi --
hicbirine yanlis "kayipli" demedi, ama kural fazla urkekti:

- 24/48 remaster: 23.7 kHz'de 60-80 dB duvar (Nyquist'in %99'u).
- Amnesiac CD: 21.1-21.2 kHz'de 17-24 dB dusus (Nyquist'in %96'si).

Ikisi de kaydin anti-alias / ornekleme hizi donusumu filtresi. Tek albumlu
set bunu icermiyordu, duvar tek basina kanit sayiliyordu. Duzeltme: kesim
medyani ANA KAPI; kesim Nyquist'e yakinsa sonuc "kayipsizla tutarli" ve duvar
yalnizca not; duvar ancak Nyquist'in %95'inin altindaysa codec izi.

Genisletilmis set (21 gercek kaynak, 4 album, 231 dosya): gercek 21/21 dogru,
sifir yanlis "kayipli"; seffaf olmayan codec'ler 168'de 163 yakalandi, 0
kacirildi, 5 belirsiz (hepsi AAC 128, PNS). 34 tam parca: 34/34.

Marj daraldi ve bu belgelenmeli: gercek min kesim 20.87 kHz, codec max 20.63
kHz. Esik 20.75'e kondu -- her iki yana ~120 Hz. Ilk (tek album) kalibrasyonda
marj 1 kHz gorunuyordu; daha cok veri daha dar ama daha gercek bir marj
verdi. Ayni dersin tekrari: kucuk orneklem fazla iyimser.

Yan bulgular: "A Moon Shaped Pool"da vendor `Mutagen 1.45.1` ve MD5 yok --
etiket kutuphanesi vendor'i kendi adiyla yazmis; `encoder_family` artik bunu
kodlayici saymiyor. Hazirlik betigi kapak resmini video izi olarak M4A'ya
kodlamaya calisip dustu: `-vn` kuralinin gerekcesi, bu kez kendi betigimde.

## Arayuz (v0.1.0): nasil kuruldu, ne ogrenildi

- **Motor Qt'ye bulasmaz**, arayuz ise ikiye ayrilir: `ui/present.py` ve
  `ui/i18n.py` Qt'siz (sonucu satira/metne ceviren kurallar ekransiz test
  edilir), widget'lar yalnizca bu satirlari cizer.
- **Motor mesajlari cevrilebilir**: `core.messages.Message` bir `str` alt
  sinifi; degeri Ingilizce cumle, ayrica sabit anahtar ve parametre tasir.
  34 mesajin tumu dokunulmadan eski testlerden gecti. Bir test motorda
  kullanilan her anahtarin Turkcesi oldugunu, bir digeri ceviri sablonunun
  motorun parametre adlarini kullandigini kilitler.
- **Uzun isler QThread'de**, iptal motorun `CancelToken`'i ile. Iptalden sonra
  motor, oldurulen ffmpeg yuzunden `CancelledError` yerine "basarisiz" hatasi
  da firlatabilir; token isaretliyse sonuc her zaman "iptal edildi" sayilir.
  Bir test gercek bir iptalde hata kutusu acilmadigini sinar.
- **Zengin metin HTML'den kacirilir**: notlar dosyadan gelen dizeler tasir
  (FLAC vendor, etiketler).

Yazarken bulunan kusurlar:

- `CodecDeltaError.user_message` bir METOT; uc yerde cagrilmadan
  kullanmistim. mypy yakaladi: hata kutusunda metot nesnesi basilacakti,
  iscide sinyal calisma aninda dusecekti.
- Ekransiz Qt testinde `QApplication`'a referans tutulmayinca Python onu
  topluyor; sonraki widget uygulamasiz kuruluyor ve Qt sureci traceback'siz
  sonlandiriyor (cikis 127). Uygulama modul duzeyinde tutuluyor.
- Yorumlayici kapanirken pyqtgraph'in yari yikilmis LabelItem'lari `_sizeHint`
  traceback'i basiyordu; testler modul sonunda pencereleri kapatip olay
  dongusunu bosaltiyor. Gercek uygulamanin kapanisi temiz olculdu.
- PyQt6 stub'lari hemen her getter'i `X | None` isaretliyor; `union-attr`
  yalnizca `app.ui.*` icin kapali, motor tam strict.

## Kodlama paneli: tablo kodda, kullanilabilirlik calisma aninda

Codec tablosu (`encode/matrix.py`) elle yazili: hangi bitrate'ler, hangi
kalite olcegi, hangi secenekler. Ama bir kodlayicinin VAR olup olmadigi
ffmpeg derlemesinden olculur; yoksa secenek gri ve sebebi ipucunda
(libfdk_aac: `--disable-libfdk-aac`). Baska bir ffmpeg ile ayni arayuz
dogru davranir.

Tablo bir testle dogrulaniyor: her kullanilabilir codec, her modda ve her
secenegin varsayilan olmayan her degeriyle 3 s kodlanir ve ffprobe beklenen
codec'i gormeli -- 38 varyant, hicbiri dusmedi. Tabloya yazilip hic
denenmemis bir secenek, arayuzde tiklayinca hata veren bir dugme olurdu.

Tutarlilik capraz kontrolu: panelden Opus 96k kodlanip otomatik
karsilastirilan gercek parca 23.15 dB verdi -- merdivenin 96k basamaginda
bagimsiz olarak olculen deger (23.15) ile ayni.

Ilerleme icin Runner'a yeni bir sinyal eklenmedi: kodlama isi asama
anahtari olarak "progress:<0..1>" yayiyor. Runner genel kaliyor, is tipi
bilmiyor.

## HTML rapor (v0.2.0)

- **Tek dosya, dis kaynak yok.** Grafikler elle uretilen inline SVG; renk
  tasimiyorlar, CSS degiskenlerinden aliyorlar, bu yuzden ayni rapor okuyucunun
  acik ya da koyu temasinda dogru gorunuyor (basliksiz Edge ile iki temada
  bakildi).
- **WebP yerine bagimliliksiz PNG.** Plan spektrogramlar icin WebP
  ongoruyordu; rapordaki tek goruntu ~41 bant x <=1000 sutunluk NMR izgarasi.
  `zlib` ile yazilan PNG birkac on KB tutuyor ve ffmpeg ya da Qt gerektirmiyor.
  Gercek 9.5 dk'lik karsilastirma raporu merdiven dahil 87 KB.
- **Autoescape acik ve sinaniyor**: bir nota konan `<script>` metin olarak
  cikiyor.
- **Gizlilik denetimi yazmayi engelliyor**: kullanici profil yolu sizdiran bir
  rapor diske hic yazilmiyor. Denetim, base64 goruntu verisini disarida
  birakiyor (icinde tesadufen bir kullanici adi harf dizisi gecebilir).
- Denetim ilk haliyle `https://` icindeki `s:/`'yi surucu yolu sayiyordu;
  baglanti iceren her rapor reddedilirdi. Surucu harfinden once harf olmamasi
  sarti eklendi.
- Grafikte gorunmeyen seri lejantta da yok (taban cogu zaman 90-150 dB,
  gorunur araligin disinda; degeri tabloda).

## Denetim duzeltmeleri (Eylul 2026)

Bulgular ve durumlari: `docs/AUDIT-2026-09.md`.

- **D1 -- blok kapisi.** Plan gecikmeyi birkac pencereden olcuyor ve dosyanin
  geri kalaninin ayni gecikmeyle hizali oldugunu varsayiyordu; duzenlenmis bir
  dosyada hizasiz bolum codec gurultusu diye olculuyordu. Ana gecis artik 1 s'lik
  bloklarin her birinde mid capraz spektrumunun PHAT tepesine bakiyor: tepe
  sifirdan +-2 ornekten uzaksa blok olcume girmiyor. Korelasyon degil tepe
  KONUMU: hizali ama gurultulu blok (sessiz pasaj, dusuk bitrate) dusuk
  korelasyon verir, atilsaydi S/N oldugundan iyi gorunurdu.
- Kesimin ustune dusen blok yari hizali ve tepe yine sifirda cikiyor; birkac
  saniyelik ilgisiz ses bile manseti bozuyordu (2 s'lik bloklarla 12.05 yerine
  6.94 dB). Hizasiz her bolgenin iki yanindaki birer blok da atiliyor.
- Bloklarin yarisindan azi tutulursa sonuc olculmemis sayiliyor: plan gecikmesi
  dosyanin cogunluguna ait degil, "kalan kisim" okuyucuyu yaniltir.
- Plan sabit gecikmede ilk kullanilabilir pencerede degil ilk "tamam" pencerede
  duruyor; kesime denk gelen pencere tum dosyayi "farkli master" yapiyordu.
- Olculen: sentetik (gercek 12.05 dB) son 13 s farkli 12.08, 45-47 s kesik
  muzik 21.95 (gercek 21.69). Gercek Loreena cifti degismedi (25.62 dB).
- **D2 -- cok kanalli mid.** Mid artik tum kanallarin ortalamasi (stereo'da
  yine (L+R)/2), side on sol - on sag. Plan hizalamayi zaten tum kanallarin
  ortalamasiyla yapiyordu; olcum ilk iki kanali aliyordu ve yalnizca merkezde
  icerigi olan 5.1 dosya S/N -inf veriyordu. Kanal basina ayri olcum daha
  dogru olurdu ama sonuc modelini ve raporu degistirir; simdilik yapilmadi ve
  rapora hangi kanalin olculdugunu soyleyen bir not dusuluyor. Dogrulama:
  merkez-yalniz AC-3 448k 50.42 dB; ayni merkez kanali mono cikarilip
  karsilastirilinca da 50.42 dB.
- **D3 -- buyutulmus dosya kaynak hizinda yargilanir.** 48 kHz'in ustundeki
  dosyada icerik standart bir hizin Nyquist'ine sigiyorsa kanit o hizda yeniden
  cikariliyor: duvar 22.05/24 kHz'teyse o hiz ("buyutulmus gorunuyor" notuyla),
  degilse 48 kHz. Esikler 44.1/48 kHz'te olculdu; 96 kHz'te kesim medyani bin
  cozunurlugu yuzunden ~240 Hz dusuyor ve 22.05 kHz'teki resampler duvari codec
  duvari saniliyordu. Yeniden orneklenmis analizde diz, taban ve kesim soxr gecis
  bandinin altinda (0.985 Nyquist) araniyor; aksi halde kendi resampler'imizin
  duvari gercek dizi golgeliyordu. Dogrulama (D:/music, 231 dosya, 21 gercek):
  96 kHz'e buyutulmus her dosya dogal hizdaki hukmunu aliyor (21/21 gercek
  kayipsiz); 88.2/176.4/192 kHz'te 7 dosyalik ornek ayni.
- **D6 -- periyodik sinyal.** Sabit tonda her periyotta esit tepe var ve
  capalar rastgele periyot katlarini seciyordu: 1 kHz sinus +3004 ppm, 440 Hz
  "NTSC pulldown". Capa basina bir belirsizlik olculuyor (ana tepenin lobu
  disindaki en yuksek tepe / ana tepe) ve medyan 0.5'i asarsa hiz olcumu
  guvenilmez, plan "sinyal kendini tekrar ediyor, gecikme belirsiz" diyor.
  Karar PAL/NTSC hipotez yarisindan ONCE, dogal (1.0) hipotezin capalariyla
  veriliyor: olceklenmis periyodik sinyal sahte bir dogru uretebiliyordu.
- Olcut PHAT ve duz korelasyonun KUCUGU. PHAT tek basina bandi daraltilmis
  sinyalde bos bantlari beyazlatip rastgele tepe uretiyor (agir EQ 0.86); duz
  korelasyon tek basina muzigi bas notalari yuzunden periyodik gosteriyor
  (0.50-0.98). Maskeli PHAT denendi ve muzigi de belirsiz gosterdi (yalnizca
  baskin bas binleri kaliyor). Olculen medyanlar: sinus/iki ton ~1.0; 5 gercek
  parca ve sentetik muzik <= 0.21; agir EQ 0.24. 21 parcanin 42 gercek cifti
  hizalandi, hicbiri periyodik sayilmadi.
- Bilinen dar marj: bin merkezine dusmeyen bir ton cok dusuk (-60 dB) bagimsiz
  gurultuyle tek pencerede 0.58 veriyor; karar medyana bakiyor ve esigin
  ustunde kaliyor ama marj dar.
- **D4 -- kullanici adi yol bileseni olarak aranir.** Profil yollari
  (USERPROFILE, APPDATA, TEMP...) ayirt edici oldugu icin her yerde alt dize
  olarak aranmaya devam ediyor. Kullanici ve makine adi ise siradan kelime
  olabiliyor (`test`, `user`, `mark`): denetim onlari yalnizca `\` ya da `/`
  ardindaki yol bileseni olarak sayiyor, temizlik butun kelime olarak siliyor.
  Rapor bu adlari yol disinda hic yazmiyor; mutlak yollar zaten ayri desenle
  yakalaniyor. Engellenen raporun iletisi ilk uc izi gosteriyor.
- **D5 -- bellek.** Karsilastirmanin tepe calisma kumesi 15 dk'da 504 -> 158 MB,
  60 dk'da 1.9 GB -> 267 MB. Dogrusal kalan tek kalem planin kabul ettigi int16
  8 kHz zarf ornekleri (iki dosya, dakikada 1.9 MB). Yapilanlar: zarf akisli
  hesaplaniyor (tum PCM float64'e cevrilmiyor); hiz tahmini int16 ornekleri
  pencere pencere float'a ceviriyor ve PAL hipotezi telafisini yalnizca okunan
  pencerede yapiyor (tam boy yeniden orneklenmis kopya yok); zarf korelasyonu
  64k'lik gecikme bloklarinda. Sonuclar degismedi (4 cift, 4 ondalik).
- Referanssiz dogrulama da akisli: kanit yalnizca uzun donem ortalamalar ve
  kare basina kesim istiyor. 30 s kesit 44.1/96/192 kHz'te 163/353/705 MB'tan
  7 MB'a indi; 40 dosyada kanit degerleri 3e-14 icinde ayni. Uzunlugunu
  belirtmeyen dosyada (boruya yazilmis FLAC) tum dosya bellege aliniyordu;
  artik bastan 60 s okunuyor ve bu not ediliyor.
