"""Kalibrasyon bekleyen esikler. TEK yer burasi.

Bu depoda dayanaksiz sayi bulundurulmuyor. Buradaki her esigin yaninda hangi
olcumden geldigi ve orneklemin ne kadar kucuk oldugu yaziyor; kalibre edilene
kadar bunlar KANITLI YER TUTUCU, kalibre edilmis deger degil.

Neden tek dosya: bir esik koda dagilirsa hangi sayinin nereden geldigi
kaybolur ve kalibrasyon imkansizlasir. `tools/calibrate_match.py` (henuz yok)
etiketlenmis gercek ciftlerle bu degerleri yeniden turetecek.
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
