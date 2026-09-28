"""Cevrilebilir motor mesajlari.

Motor kullaniciya gosterilecek gerekceler uretir ("content stops at 16.1 kHz").
Bunlar duz Ingilizce dize olsaydi arayuz Turkce modda bile Ingilizce gosterirdi.

`Message` bir `str` ALT SINIFIDIR: degeri Ingilizce cumlenin kendisidir, yani
loglar, testler (`"brickwall" in reason`) ve raporlar hicbir degisiklik
gerektirmez. Ek olarak sabit bir `key` ve bicimlendirme `params` tasir; arayuz
anahtari kendi dilinde bulursa cevirisini, bulamazsa Ingilizceyi gosterir.

Motor Qt'ye de ceviri sozlugune de bagimli degildir: sozluk arayuz katmaninda.
"""

from __future__ import annotations

from typing import Any


class Message(str):
    key: str
    params: dict[str, Any]

    def __new__(cls, key: str, template: str, **params: Any) -> Message:
        obj = super().__new__(cls, template.format(**params))
        obj.key = key
        obj.params = params
        return obj

    def __reduce__(self) -> tuple[Any, ...]:
        # Cogaltma/pickle: str alt siniflari varsayilan olarak ek alanlari kaybeder.
        return (_rebuild, (self.key, str(self), self.params))


def _rebuild(key: str, text: str, params: dict[str, Any]) -> Message:
    obj = str.__new__(Message, text)
    obj.key = key
    obj.params = params
    return obj
