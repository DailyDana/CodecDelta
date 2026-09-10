"""Ogg sayfa/paket okuyucu. Opus ve Vorbis bunun uzerine oturur.

ffprobe konteyner duzeyinde ne varsa soyler; buradaki amac bir kademe asagisi:
kodlayicinin gercekte NE YAPTIGI. Opus'ta paket basina mod (SILK/Hybrid/CELT),
bant genisligi ve cerceve suresi yalnizca bit akisindan okunur ve "bu dosya
gercekten 20 ms fullband CELT mi" sorusunun tek cevabi odur.

Neden saf Python: bu is bir kez, dosya basina birkac yuz milisaniye. Bir C
bagimliligi tasimaya degmez ve ffmpeg bu istatistikleri disari vermiyor.

Buyuk dosya politikasi
----------------------
Tipik bir .opus 10 MB'dir ve tumuyle taranir. Esik asilirsa bas + orta + son
bolgelerden ornek alinir; ornekleme noktalarinda akisin ortasina duselim diye
"OggS" ile yeniden senkron olunur. Yanlis senkrona karsi sayfa CRC'si
dogrulanir -- ses verisinin icinde "OggS" dizisi rastlantiyla gecebilir ve
CRC bunu eleyen tek guvenilir olcut.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"OggS"

# Sabit sayfa basligi uzunlugu (segment tablosu haric).
_HEADER_BYTES = 27

# Sayfa bayraklari.
FLAG_CONTINUED = 0x01
FLAG_FIRST = 0x02
FLAG_LAST = 0x04

# Bu boyutun altindaki dosyalar tumuyle taranir. 128 MB, gercekte karsilasilan
# her .opus/.ogg dosyasini kapsar; ustu icin ornekleme devreye girer.
FULL_SCAN_LIMIT = 128 << 20

# Ornekleme yapilirken her bolgeden okunacak miktar ve bolge sayisi.
_SAMPLE_REGION_BYTES = 8 << 20
_MIDDLE_SAMPLES = 3

# CRC dogrulama politikasi. Saf Python bayt dongusu ~7 MB/s: 10 MB'lik bir
# .opus dosyasinin HER sayfasini dogrulamak 1393 ms suruyor ve bu, tarama
# maliyetinin %100'u (basliklari cozmek 1 ms). Bu yuzden sirali taramada CRC
# ORNEKLENIR: ilk sayfalar ve sonra her N'inci sayfa. Yeniden senkron yolu
# her zaman dogrular -- orada CRC bir hiz meselesi degil, dogruluk meselesi.
#
# Kaybedilen sey: dosyanin derinlerinde tek bir bozuk sayfa gozden kacabilir.
# Yakalanan sey: yaygin bozulma, yanlis kapsayici, kirpilmis indirme. Rapor
# kac sayfanin dogrulandigini SOYLER, "hepsi saglam" demez.
_CRC_CHECK_HEAD_PAGES = 16
_CRC_CHECK_STRIDE = 64


def _crc_table() -> list[int]:
    """Ogg'un CRC32'si: polinom 0x04c11db7, yansitma yok, son XOR yok.

    Standart zlib CRC32'sinden farklidir; hazir bir fonksiyon kullanilamaz.
    """
    table: list[int] = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
                if crc & 0x80000000
                else (crc << 1) & 0xFFFFFFFF
            )
        table.append(crc)
    return table


_CRC_TABLE = _crc_table()


def page_crc(data: bytes) -> int:
    """Tam bir sayfanin (baslik + govde) CRC'sini hesaplar.

    Cagiran taraf CRC alanini sifirlamis olmalidir.
    """
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[((crc >> 24) & 0xFF) ^ byte]
    return crc


@dataclass(frozen=True)
class Page:
    """Cozulmus bir Ogg sayfasi."""

    offset: int
    version: int
    flags: int
    granule: int
    serial: int
    sequence: int
    segments: tuple[int, ...]
    body: bytes
    # None = dogrulanmadi (bkz. CRC ornekleme politikasi). True/False =
    # dogrulandi. Ucuncu bir durum olmasi bilincli: "kontrol etmedim" ile
    # "saglam" ayni sey degil ve rapor bu ikisini karistirmamali.
    crc_ok: bool | None

    @property
    def continued(self) -> bool:
        return bool(self.flags & FLAG_CONTINUED)

    @property
    def first(self) -> bool:
        return bool(self.flags & FLAG_FIRST)

    @property
    def last(self) -> bool:
        return bool(self.flags & FLAG_LAST)

    @property
    def total_size(self) -> int:
        return _HEADER_BYTES + len(self.segments) + len(self.body)


@dataclass
class ScanStats:
    """Tarama sirasinda biriken sayfa duzeyi bilgiler."""

    pages: int = 0
    crc_checked_pages: int = 0
    bad_crc_pages: int = 0
    bytes_read: int = 0
    sampled: bool = False
    first_granule: int | None = None
    last_granule: int | None = None
    serials: set[int] = field(default_factory=set)


def parse_page(data: bytes, offset: int = 0, *, verify_crc: bool = True) -> Page | None:
    """`data[offset:]` konumundaki sayfayi cozer. Eksik/bozuksa None.

    Bozuk CRC sayfayi REDDETMEZ: dosya kirpilmis veya hafifce bozulmus olabilir
    ve elimizdeki bilgiyi yine de raporlamak isteriz. Bayrak `Page.crc_ok` ile
    tasinir, sayimi `ScanStats`'te tutulur.

    `verify_crc=False` ile CRC hic hesaplanmaz ve `crc_ok` None kalir; sirali
    tarama bunu kullanir (maliyet gerekcesi modul basindaki nota bakin).
    """
    if len(data) < offset + _HEADER_BYTES or data[offset : offset + 4] != MAGIC:
        return None
    header = data[offset : offset + _HEADER_BYTES]
    version = header[4]
    flags = header[5]
    granule = int.from_bytes(header[6:14], "little")
    serial = int.from_bytes(header[14:18], "little")
    sequence = int.from_bytes(header[18:22], "little")
    stored_crc = int.from_bytes(header[22:26], "little")
    nsegs = header[26]

    table_end = offset + _HEADER_BYTES + nsegs
    if len(data) < table_end:
        return None
    segments = data[offset + _HEADER_BYTES : table_end]
    body_len = sum(segments)
    body_end = table_end + body_len
    if len(data) < body_end:
        return None

    crc_ok: bool | None = None
    if verify_crc:
        raw = bytearray(data[offset:body_end])
        raw[22:26] = b"\x00\x00\x00\x00"
        crc_ok = page_crc(bytes(raw)) == stored_crc

    return Page(
        offset=offset,
        version=version,
        flags=flags,
        granule=granule,
        serial=serial,
        sequence=sequence,
        segments=tuple(segments),
        body=data[table_end:body_end],
        crc_ok=crc_ok,
    )


def find_page(data: bytes, start: int = 0) -> Page | None:
    """`start`ten itibaren CRC'si tutan ilk sayfayi arar.

    Ornekleme sonrasi yeniden senkron icin. Yalnizca CRC'si dogru bir sayfayi
    kabul eder: ses verisinin icinde "OggS" rastlantiyla gecebilir ve o
    durumda basliktaki segment tablosu sacma bir govde uzunlugu uretir.
    """
    pos = start
    while True:
        pos = data.find(MAGIC, pos)
        if pos < 0:
            return None
        page = parse_page(data, pos)
        if page is not None and page.crc_ok and page.version == 0:
            return page
        pos += 1


def _should_verify(index: int) -> bool:
    """CRC ornekleme politikasi: ilk sayfalar, sonra her N'inci."""
    return index < _CRC_CHECK_HEAD_PAGES or index % _CRC_CHECK_STRIDE == 0


def iter_pages(data: bytes, start: int = 0) -> Iterator[Page]:
    """Ardisik sayfalari uretir; ilk bozuk noktada yeniden senkron olur.

    CRC yalnizca ornek sayfalarda hesaplanir. Yeniden senkron gerektiginde
    `find_page` devreye girer ve o HER ZAMAN dogrular, cunku yanlis bir
    "OggS" eslesmesini eleyen tek olcut CRC'dir.
    """
    pos = start
    index = 0
    while pos < len(data):
        page = parse_page(data, pos, verify_crc=_should_verify(index))
        if page is None:
            resynced = find_page(data, pos + 1)
            if resynced is None:
                return
            page = resynced
            pos = page.offset
        yield page
        pos += page.total_size
        index += 1


def iter_packets(pages: Iterator[Page], stats: ScanStats | None = None) -> Iterator[bytes]:
    """Sayfalardan paketleri yeniden kurar.

    Ogg'da bir paket sayfa sinirlarini asabilir: 255 uzunlugundaki her segment
    "devam ediyor" demektir, 255'ten kisa olan paketi bitirir. Tarama bir
    dosyanin ortasindan basladiysa ilk paket yarim olabilir; `Page.continued`
    bayragi bunu soyler ve o paket atilir -- yarim bir paketin TOC bayti
    anlamsizdir ve istatistigi bozar.
    """
    current = bytearray()
    have_start = True
    for page in pages:
        if stats is not None:
            stats.pages += 1
            stats.serials.add(page.serial)
            if page.crc_ok is not None:
                stats.crc_checked_pages += 1
                if not page.crc_ok:
                    stats.bad_crc_pages += 1
            if stats.first_granule is None:
                stats.first_granule = page.granule
            stats.last_granule = page.granule

        if page.continued and not current:
            # Sayfa, bizim gormedigimiz bir paketin devami: o paketi atla.
            have_start = False
        pos = 0
        for seg in page.segments:
            current += page.body[pos : pos + seg]
            pos += seg
            if seg < 255:
                if have_start and current:
                    yield bytes(current)
                current = bytearray()
                have_start = True
    # Dosya sonundaki yarim paket bilincli olarak atilir.


def read_regions(path: Path, *, full_scan_limit: int = FULL_SCAN_LIMIT) -> tuple[list[bytes], bool]:
    """Dosyayi tumuyle veya orneklenerek okur.

    Donen ikinci deger `sampled`: True ise istatistikler tum dosyayi degil
    orneklenen bolgeleri temsil eder ve rapor bunu SOYLEMEK zorundadir.
    """
    size = path.stat().st_size
    if size <= full_scan_limit:
        return [path.read_bytes()], False

    # Bolgeler dosyaya esit araliklarla yayilir: ilki 0'da (baslik paketleri
    # orada, atlanamaz), sonuncusu tam dosyanin sonunda (granule oradan
    # okunur). Bolge boyutu dosyaya gore kuculur; aksi halde kucuk bir dosyada
    # bolgeler ust uste biner ve ofset hesabi negatife duser.
    count = _MIDDLE_SAMPLES + 2
    region = min(_SAMPLE_REGION_BYTES, max(1, size // count))
    span = size - region

    regions: list[bytes] = []
    with path.open("rb") as fh:
        for i in range(count):
            fh.seek(span * i // (count - 1))
            regions.append(fh.read(region))
    return regions, True


def scan_packets(
    path: Path, *, full_scan_limit: int = FULL_SCAN_LIMIT
) -> tuple[list[bytes], ScanStats]:
    """Bir Ogg dosyasindan paketleri ve sayfa istatistiklerini cikarir."""
    regions, sampled = read_regions(path, full_scan_limit=full_scan_limit)
    stats = ScanStats(sampled=sampled)
    packets: list[bytes] = []
    for index, region in enumerate(regions):
        stats.bytes_read += len(region)
        if index == 0:
            pages = iter_pages(region)
        else:
            # Ornek bolgesi akisin ortasina duser; ilk saglam sayfayi bul.
            anchor = find_page(region)
            if anchor is None:
                continue
            pages = iter_pages(region, anchor.offset)
        packets.extend(iter_packets(pages, stats))
    return packets, stats
