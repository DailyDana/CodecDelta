"""Referanssiz dogrulama esikleri. TEK yer burasi.

Her sayi `tools/calibrate_transcode.py` ciktisindan geliyor (28 Eyl 2026):
9 gercek CD parcasi (TEK album, Loreena McKennitt "An Ancient Muse", 44.1/16,
60 s kesitler) ve her birinden ffmpeg ile uretilmis, 16 BITE geri yazilmis
transcode'lar. Hucreler min / medyan / max.

    sinif        kesim p50 kHz        diz dusus dB/500Hz    taban rel dB
    real         21.2 / 21.5 / 21.5    4.6 /  7.2 / 10.2   -44.4 / -24.0 / -18.7
    mp3_128      16.0 / 16.1 / 16.6   39.5 / 52.4 / 58.1   -84.5 / -77.7 / -64.1
    mp3_192      16.2 / 18.4 / 18.7   20.7 / 48.8 / 55.5   -84.7 / -77.9 / -64.2
    mp3_320      19.8 / 20.0 / 20.1   26.7 / 44.7 / 57.3   -84.9 / -78.1 / -64.4
    opus 96-160  20.0 / 20.0 / 20.2   28.3 / 37.5 / 39.3   -85.6 / -78.3 / -64.4
    vorbis_q5    17.1 / 18.3 / 20.1   25.2 / 37.8 / 43.8   -85.2 / -78.1 / -64.3
    aac_128      19.8 / 19.9 / 20.0    8.6 / 18.1 / 38.5   -83.3 / -42.9 / -24.6
    aac_256      21.2 / 21.5 / 21.5    4.6 /  7.2 / 10.2   -44.5 / -24.2 / -18.8
    mp3_v0       16.0 / 20.0 / 21.5    6.3 /  8.8 / 14.2   -83.1 / -24.2 / -19.3

Ilk kalibrasyon 24 bitlik transcode'larla yapilmisti (ffmpeg float cozumu
FLAC'a s32 yazar) ve taban -92..-128 gorunuyordu; 16 bitte kuantalama
gurultusu tabani -64'e cekiyor. Sahte bir FLAC hemen her zaman 16 bittir;
esikler 16 bit tabloya gore.

ORNEKLEM TEK MASTERING ve koyu (bant sinirli) gercek kayit ICERMIYOR: eski bir
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

# Cerceve kesim medyani bunun altindaysa kayipli izi. Gercek min 21.2 kHz,
# seffaf olmayan codec max 20.2 kHz; esik aralarinda, gercek tarafa yakin.
MAX_LOSSY_CUTOFF_HZ = 20_500.0
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
