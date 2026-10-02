"""Kalibrasyon bekleyen esikler. TEK yer burasi.

Bu depoda dayanaksiz sayi bulundurulmuyor. Buradaki her esigin yaninda hangi
olcumden geldigi ve orneklemin ne kadar kucuk oldugu yaziyor; kalibre edilene
kadar bunlar KANITLI YER TUTUCU, kalibre edilmis deger degil.

Neden tek dosya: bir esik koda dagilirsa hangi sayinin nereden geldigi
kaybolur ve kalibrasyon imkansizlasir.

GERCEK VERIYLE KALIBRASYON (tools/calibrate_match.py, Ekim 2026; 21 gercek
parca, 60 s kesitler; MP3/Opus/AAC/Vorbis 96-320 kbps):

    sinif                      n    zarf rho            ince |r|
    ayni kayit               210    0.995 .. 1.000      0.990 .. 1.000   210/210 hizali
    farkli kayit              63    0.072 .. 0.420      -                63/63 farkli kayit
    farkli master (SENTETIK)  21    0.506 .. 0.978      0.875 .. 0.944
    24-64 kbps kodlama        39    -                   0.953 .. 0.997   39/39 hizali

Ayni/farkli kayit ayrimi gercek veride genis (0.420 / 0.995); asagidaki
esikler bu boslugun icinde. "Farkli master" sinifi SENTETIK (EQ + sikistirma):
gercek remaster ciftimiz yok. Ince korelasyon farkli masteri cok dusuk bit
hizli kodlamadan saglam ayiramiyor (0.944 / 0.953); bu yuzden ona bir hukum
degil yalnizca bir uyari baglandi (`compare.pipeline.MASTER_SUSPECT_CORRELATION`).
"""

from __future__ import annotations

# Hizalamanin gecerli sayilmasi icin bulunan gecikmedeki en kucuk normalize
# korelasyon.
#
# OLCULEN (ayrinti: docs/SPEC-alignment.md):
#
#     GECERLI malzeme        0.8930 .. 1.0000
#       en dusuk: gercek muzik + 6 dB S/N gurultu -> 0.8930
#       Opus 32 kbps -> 0.9895, MP3 64k -> 0.9978, AAC 32k -> 0.9946
#
#     GECERSIZ malzeme      -0.2311 .. 0.4854
#       yarisi iliskisiz -> 0.4854, aralik disi gecikme -> -0.2311,
#       iliskisiz beyaz gurultu -> 0.0035
#
#     SINIRDA
#       duz spektrumun 0.25'e kesilmesi -> 0.7858
#
# Esik 0.485 ile 0.786 arasindaki bosluga konuldu. ORNEKLEM KUCUK: dort
# kurgulanmis gecersiz vaka. Gercek "farkli master" ve "farkli cekim"
# ciftleriyle yeniden turetilmeli.
MIN_ALIGNMENT_CORRELATION = 0.60

# Keskinlik (artik(d+-0.5) / artik(d)) icin esik YOK ve bilincli olarak yok.
#
# Olculdu: gecerli vakalar 1.02'ye kadar iniyor (Opus 32k -> 1.1, 6 dB gurultu
# -> 1.0), gecersiz vakalar 1.131'e kadar cikiyor. Araliklar TAMAMEN ortusuyor,
# yani keskinlik gecerli/gecersiz ayrimi yapamaz.
#
# Onceki olcumdeki alti basamaklik ayrim, sinyalin KENDISIYLE karsilastirildigi
# sentetik vakalarin artefaktiydi: orada artik sifira gittigi icin oran
# patliyordu. Gercek codec ciftinde artik codec gurultusu kadardir.
#
# Keskinlik yine de raporlanir -- gecikmenin ne kadar keskin tanimlandigini
# soyler ve kullanicinin gormesi anlamlidir -- ama hicbir karar ona baglanmaz.


# Zarf (L1) duzeyinde "bu ayni kayit mi" esigi. Bunun altinda kaba hizalama
# guvenilmezdir ve boru hatti "ayni kaydi icermiyor" demeye hazirlanmalidir.
MIN_ENVELOPE_CORRELATION = 0.70
# OLCULEN (40 tohum, sentetik dinamik gurultu, 8 kHz zarf):
#   ILGILI  (3 s kayma + alcak geciren 0.18) 0.949 .. 0.977
#   KESIT   (30 s icinde 4 s)                1.000 .. 1.000
#   ILGISIZ (bagimsiz kaynak)                0.058 .. 0.458
# Bosluk genis (0.458 / 0.949); esik ortasina degil, ilgili tarafin guvenli
# altina kondu. ORNEKLEM SENTETIK: gercek etiketli ciftlerle dogrulanmali.
#
# PSR (tepe/yan-lob) icin esik YOK ve bilincli olarak yok. Ayni 40 tohumda
# ILGILI 2.385..6.393 ve ILGISIZ 1.877..7.097 -- TAMAMEN ortusuyor. Sebep:
# zarf duzgundur, komsu gecikmeler neredeyse tepe kadar iyidir, yani yan lob
# medyani hicbir zaman dusmez. PSR yalnizca GCC-PHAT'in beyazlatilmis keskin
# tepesinde anlamlidir; orada kalir (bkz. gccphat.LagEstimate).
