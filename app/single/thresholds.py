"""Referanssiz dogrulama esikleri. TEK yer burasi.

Her sayi `tools/calibrate_transcode.py` ciktisindan geliyor (28 Eyl 2026):
21 gercek kaynak, DORT album -- Loreena McKennitt "An Ancient Muse" (CD,
44.1/16, 9 parca), Radiohead "Amnesiac" (CD, 44.1/16, 4), Radiohead "A Moon
Shaped Pool" (24/48, 4), Fleetwood Mac "Mirage" 2016 remaster (24/48, 4) --
60 s kesitler ve her birinden ffmpeg ile 16 BITE geri yazilmis transcode'lar.
Toplam 231 dosya. Hucreler min / medyan / max.

    sinif        kesim p50 kHz        diz dusus dB/500Hz    taban rel dB
    real         20.9 / 21.5 / 23.5    4.6 /  8.8 / 79.1   -51.9 / -34.6 / -18.7
    mp3_128      15.4 / 16.1 / 16.8   32.2 / 52.1 / 58.1   -87.4 / -77.7 / -63.4
    mp3_192      16.0 / 18.5 / 19.0   20.7 / 48.8 / 59.4   -87.6 / -78.2 / -64.2
    mp3_320      19.8 / 20.1 / 20.5   26.7 / 45.9 / 60.8   -87.9 / -78.7 / -60.2
    opus 96-160  20.0 / 20.1 / 20.5   21.8 / 36.0 / 39.3   -86.9 / -73   / -50.4
    vorbis_q5    17.1 / 18.6 / 20.6   11.8 / 32.7 / 43.8   -87.1 / -74.2 / -40.7
    aac_128      19.8 / 20.0 / 20.3    8.6 / 23.3 / 38.5   -86.1 / -64.1 / -24.6
    aac_256      20.9 / 21.5 / 21.8    4.6 / 14.6 / 31.5   -67.9 / -44.5 / -18.8
    mp3_v0       16.0 / 20.8 / 22.8    6.3 / 14.2 / 39.1   -83.1 / -51.5 / -19.3

Gercek kayitta diz dususu 79 dB'e kadar cikiyor: 24/48 remaster'da 23.7 kHz'de
(Nyquist'in %99'u) anti-alias filtresi. Bu yuzden duvar, Nyquist'e yakinsa
kanit sayilmaz (verdict.py).

Siniflandirma sonucu (ayni set): gercek 21/21 kayipsizla tutarli, SIFIR yanlis
"kayipli". Seffaf olmayan codec'ler (mp3 128/192/320, opus, vorbis, aac 128):
168 dosyadan 163 kayipli, 0 kayipsiz, 5 belirsiz (hepsi AAC 128, PNS).
(Esik 20.5 iken 161 / 1 / 6: kacan vorbis'in kesimi 20.63 kHz'di.)
Ayrica D:/music'teki 34 tam parca: 34/34 kayipsizla tutarli.

Ilk kalibrasyon 24 bitlik transcode'larla yapilmisti (ffmpeg float cozumu
FLAC'a s32 yazar) ve taban -92..-128 gorunuyordu; 16 bitte kuantalama
gurultusu tabani -64'e cekiyor. Sahte bir FLAC hemen her zaman 16 bittir;
esikler 16 bit tabloya gore.

Dort album, uc mastering donemi; ama koyu (bant sinirli) gercek kayit
ICERMIYOR: eski bir
kayit dusuk kesim gosterir; onu kurtaracak olan taban ve diz karsi-kanitlari
bu sette yalnizca sentetik olarak sinandi. AAC 256 ve MP3 V0'in bazi parcalari
spektral olarak gercekten AYIRT EDILEMEZ; arac onlara "kayipsiz kaynakla
tutarli" der -- bu bir sinir, hata degil, dil kurali bu yuzden var.

Ayirt ETMEYEN ve kullanilmayan olcumler (kayit icin):
- cerceve kesim IQR: gercek 0/43/1669 Hz, codec'ler 65-2500 Hz. Planin
  "transcode'da dar, dogalda genis" iddiasinin TERSI: gercek CD'de icerik
  Nyquist'e kadar var, kesim tepede sabit; codec'te sfb21 kuantalamasi
  kesimi cerceveden cerceveye kaydiriyor.
- side HF seviyesi: gercek -12.7/-2.7/-0.7 dB, codec'ler -16.8/-5/+2.4 dB.
  ffmpeg kodlayicilari side'i cokertmiyor. Baska kodlayicilarda (eski Xing,
  iTunes) DOGRULANMADI.
"""

from __future__ import annotations

# Cerceve kesim medyani bunun altindaysa kayipli izi. Gercek min 20.87 kHz,
# seffaf olmayan codec max 20.63 kHz (vorbis): esik tam ortada ve MARJ DAR
# (her iki yana ~120 Hz). 44.1 kHz'de codec alcak gecireni (~20.5) ile CD'nin
# kendi anti-alias filtresi (~21) birbirine bu kadar yakin; daha fazla gercek
# veriyle yeniden olculmeli. Tek albumlu ilk kalibrasyonda marj 1 kHz'di.
MAX_LOSSY_CUTOFF_HZ = 20_750.0
# ... ve Nyquist'in bu kesrinin altinda olmali (32 kHz'lik bir dosyada 15.5 kHz
# kesim, kayip degil dosyanin dogasidir). Gercek 44.1 kHz min: 0.961.
MAX_LOSSY_CUTOFF_NYQUIST_FRACTION = 0.95

# Diz ustu taban bunun altindaysa "dizin ustunde hicbir sey yok".
# Gercek min -44.4; 16 bit MP3/Opus/Vorbis max -64.1 (kuantalama tabani). Esik
# ortada. AAC-PNS -24.6'ya kadar cikiyor ve bu kanitin disinda kalir (bilinen
# yanlis negatif yonu).
EMPTY_FLOOR_REL_DB = -55.0
# Taban bunun ustundeyse dizin ustunde icerik var: karsi-kanit.
CONTENT_FLOOR_REL_DB = -50.0

# 500 Hz'de bu kadar dusus "duvar". Gercek max 10.2; MP3/Opus/Vorbis min 20.7.
BRICKWALL_DROP_DB = 18.0
# Bunun altindaki diz yumusak: karsi-kanit. Gercek medyan 7.2, max 10.2.
GENTLE_KNEE_DROP_DB = 14.0

# Bilinen kodlayici kesimleri (kHz) ve etiketleri; kesim medyani bunlardan
# birine bu kadar yakinsa etiket eklenir. Yalnizca bu sette OLCULEN degerler.
# Olculen: mp3_128 16.0-16.6 (muzik), 16.8 (pembe gurultu); mp3_192 16.2-18.7;
# 320/Opus/AAC 19.8-20.2. Etiket olarak yardimci, kanit olarak DEGIL.
KNOWN_CUTOFFS_KHZ: tuple[tuple[float, str], ...] = (
    (16.5, "MP3 ~128 kbps (LAME lowpass)"),
    (18.5, "MP3 ~192 kbps (LAME lowpass)"),
    (20.0, "MP3 320 kbps, Opus, or AAC"),
)
CUTOFF_SNAP_KHZ = 0.6

# Analiz icin gereken en az aktif (sessiz olmayan) cerceve; alti "belirsiz".
MIN_ACTIVE_FRAMES = 40
