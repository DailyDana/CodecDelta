"""Arayuz testleri.

Iki katman:
- Qt'siz: ceviri sozlugu ve sunum kurallari (her ortamda kosar).
- Ekransiz Qt (offscreen): pencere kurulur, gercek bir karsilastirma arka plan
  isciisi uzerinden bastan sona calistirilir ve sonuc paneli dolar.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from app.core.messages import Message
from app.ui import i18n, present
from app.ui.i18n import STRINGS, localize, set_language, tr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _english() -> object:
    set_language("en")
    yield
    set_language("en")


# -- ceviri -------------------------------------------------------------------


def test_languages_share_the_same_interface_keys() -> None:
    """Arayuz anahtarlari iki dilde de olmali; eksik anahtar sessizce Ingilizce kalirdi.

    `msg.*` haric: motor mesajlarinin Ingilizcesi motorun kendisinde.
    """
    keys = {lang: {k for k in table if not k.startswith("msg.")} for lang, table in STRINGS.items()}
    assert keys["en"] == keys["tr"]


def test_every_engine_message_has_a_turkish_translation() -> None:
    """Motorda `Message("anahtar", ...)` ile uretilen her mesajin Turkcesi olmali."""
    pattern = re.compile(r'Message\(\s*"([a-z0-9_.]+)"')
    used = set()
    for path in (ROOT / "app").rglob("*.py"):
        if "ui" in path.parts:
            continue
        used.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert used, "motorda hic Message bulunamadi"
    missing = sorted(k for k in used if f"msg.{k}" not in STRINGS["tr"])
    assert not missing, f"Turkcesi eksik: {missing}"
    unused = sorted(k[4:] for k in STRINGS["tr"] if k.startswith("msg.") and k[4:] not in used)
    assert not unused, f"kullanilmayan ceviri: {unused}"


def test_every_translation_formats_with_the_engine_params() -> None:
    """Ceviri sablonu motorun verdigi parametre adlarini kullanmali (yazim hatasi yakalar)."""
    pattern = re.compile(r'Message\(\s*"([a-z0-9_.]+)",\s*((?:"[^"]*"\s*)+)', re.S)
    for path in (ROOT / "app").rglob("*.py"):
        if "ui" in path.parts:
            continue
        for key, template in pattern.findall(path.read_text(encoding="utf-8")):
            english = "".join(re.findall(r'"([^"]*)"', template))
            names = set(re.findall(r"\{(\w+)", english))
            turkish = STRINGS["tr"][f"msg.{key}"]
            assert set(re.findall(r"\{(\w+)", turkish)) == names, key


def test_localize_translates_messages_and_their_nested_params() -> None:
    label = Message("drift.pal_up", "PAL speed-up (24->25 fps)")
    message = Message(
        "plan.speed", "different speed ({label}, ratio {ratio:.6f})", label=label, ratio=1.041667
    )
    assert localize(message) == str(message)
    set_language("tr")
    text = localize(message)
    assert "farklı hız" in text and "PAL hızlandırması" in text and "1.041667" in text
    assert localize("plain text") == "plain text"


def test_message_is_a_plain_string_for_everything_else() -> None:
    m = Message("single.ratio", "FLAC compression ratio {ratio:.2f}", ratio=0.5)
    assert m == "FLAC compression ratio 0.50"
    assert "ratio 0.50" in m
    assert m.key == "single.ratio" and m.params == {"ratio": 0.5}
    import copy
    import pickle

    for clone in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert clone == m and clone.key == m.key and clone.params == m.params


def test_missing_key_falls_back_to_english_then_to_the_key() -> None:
    set_language("tr")
    assert tr("no.such.key") == "no.such.key"
    set_language("xx")
    assert i18n.language() == "en"


# -- sunum ------------------------------------------------------------------------


def test_db_text_never_prints_nan_or_inf() -> None:
    assert present.db_text(math.nan) == present.DASH
    assert present.db_text(math.inf).startswith(">")
    assert present.db_text(12.345) == "12.3 dB"


def test_band_labels_use_hz_below_1_khz() -> None:
    assert present._band_label(20.0, 1000.0) == "20 Hz–1 kHz"
    assert present._band_label(1000.0, 4000.0) == "1–4 kHz"
    assert present._band_label(20000.0, 21829.5) == "20–21.8 kHz"


# -- ekransiz Qt ------------------------------------------------------------------------


_APP: list[object] = []


@pytest.fixture(scope="module", autouse=True)
def _dispose_qt() -> object:
    """Modul sonunda ust duzey widget'lari yok et ve olay dongusunu bosalt.

    Yapilmazsa yorumlayici kapanirken pyqtgraph'in yari yikilmis LabelItem'lari
    boyut sorgusu alir ve `_sizeHint` traceback'leri basar (zararsiz ama
    gurultulu; olculdu).
    """
    yield
    if _APP:
        from PyQt6.QtWidgets import QApplication

        for widget in QApplication.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        for _ in range(5):
            QApplication.processEvents()


def _qt_app():  # type: ignore[no-untyped-def]
    """Tek QApplication, modul omru boyunca TUTULUR.

    Referans tutulmazsa Python onu hemen toplar; sonraki widget uygulamasiz
    kurulur ve Qt sureci traceback'siz sonlandirir (cikis 127). Ilk yazdigim
    test tam olarak buna takildi.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from app.ui.theme import STYLESHEET

    if not _APP:
        app = QApplication.instance() or QApplication([])
        app.setStyleSheet(STYLESHEET)
        _APP.append(app)
    return _APP[0]


def _fake_tools():  # type: ignore[no-untyped-def]
    from app.core.ffmpeg_locate import Capabilities, FFmpegTools

    caps = Capabilities(
        version="test", build_flags=frozenset(), encoders=frozenset(), filters=frozenset()
    )
    return FFmpegTools(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), caps=caps)


def test_window_builds_and_switches_language_without_ffmpeg() -> None:
    _qt_app()
    from app.core.settings import Settings
    from app.ui.main_window import MainWindow

    window = MainWindow(_fake_tools(), Settings(language="en"))
    tab = window.analyze
    assert not tab.compare_button.isEnabled()
    assert not tab.verify_button.isEnabled()
    assert tab.compare_button.text() == "Compare"
    window.settings = Settings(language="en")
    import app.core.settings as settings_mod

    saved: list[Settings] = []
    original = settings_mod.save
    settings_mod.save = lambda s, path=None: saved.append(s) or True  # type: ignore[assignment,misc]
    try:
        window.set_language("tr")
    finally:
        settings_mod.save = original
    assert window.analyze.compare_button.text() == "Karşılaştır"
    assert saved and saved[-1].language == "tr"
    window.close()


@pytest.mark.needs_ffmpeg
def test_compare_runs_end_to_end_through_the_interface(
    ffmpeg_tools, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    app = _qt_app()
    from app.core.settings import Settings
    from app.ui.main_window import MainWindow

    ff = str(ffmpeg_tools.ffmpeg)
    reference = tmp_path / "ref.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=15:seed=3,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(reference),
        ],
        check=True,
    )
    test = tmp_path / "test.opus"
    subprocess.run(
        [ff, "-v", "error", "-i", str(reference), "-c:a", "libopus", "-b:a", "96k", str(test)],
        check=True,
    )

    window = MainWindow(ffmpeg_tools, Settings(language="tr"))
    tab = window.analyze
    tab.load(reference, test)
    assert tab.compare_button.isEnabled()
    stages: list[str] = []
    tab.runner.stage.connect(stages.append)
    tab.start_compare()
    deadline = time.time() + 120
    while (tab.runner.busy or tab.last_result is None) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()

    assert tab.last_result is not None and tab.last_result.status == "measured"
    assert stages == ["align", "measure", "floor"]
    # Pencere gosterilmedigi icin isVisible() hep False; gizlenmemis olmasi yeter.
    assert not tab.results.headline_card.isHidden()
    assert not tab.results.bands_card.isHidden()
    assert not tab.results.nmr_card.isHidden()
    assert tab.results.table.rowCount() == len(tab.last_result.bands)
    assert "tamamlandı" in tab.stage.text()
    assert tab.results.ladder_button.isEnabled()

    # Raporu kaydet: dosya iletisimi atlanir, rapor Turkce ve temiz yazilmali.
    from PyQt6.QtWidgets import QFileDialog

    target = tmp_path / "rapor.html"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "HTML"))
    assert tab.report_button.isEnabled()
    tab.save_report()
    text = target.read_text(encoding="utf-8")
    assert "Karşılaştırma raporu" in text and "ref.flac" in text
    assert str(tmp_path) not in text

    tab.start_verify()
    deadline = time.time() + 60
    while tab.runner.busy and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
    assert "Kayıpsız" in tab.results.title.text()
    tab.save_report()
    assert "Kayıpsızlık doğrulama raporu" in target.read_text(encoding="utf-8")
    window.close()


@pytest.mark.needs_ffmpeg
def test_cancel_reports_cancelled_not_an_error(ffmpeg_tools, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Iptal, oldurulen ffmpeg'in "basarisiz" cikisina ragmen hata kutusu acmamali."""
    app = _qt_app()
    from app.core.settings import Settings
    from app.ui.main_window import MainWindow

    ff = str(ffmpeg_tools.ffmpeg)
    reference = tmp_path / "long.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=240:seed=4",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(reference),
        ],
        check=True,
    )
    window = MainWindow(ffmpeg_tools, Settings(language="en"))
    tab = window.analyze
    tab.load(reference, reference)
    failures: list[str] = []
    cancelled: list[bool] = []
    tab.runner.failed.disconnect()
    tab.runner.failed.connect(failures.append)
    tab.runner.cancelled.connect(lambda: cancelled.append(True))
    tab.start_compare()
    time.sleep(0.3)
    app.processEvents()
    tab.runner.cancel()
    deadline = time.time() + 60
    while tab.runner.busy and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
    assert cancelled and not failures
    assert tab.stage.text() == "Cancelled."
    assert tab.compare_button.isEnabled()
    # Iptal edilen karsilastirmanin izleri kaydedilmemeli: merdiven ve rapor
    # onlari onceki sonucla eslestiriyordu (D17).
    assert tab._last_tracks is None and tab.last_result is None
    window.close()


def test_closing_during_a_job_does_not_hang() -> None:
    """Kapanista ana is parcacigi `wait()` icinde bloke; is yine de bitmeli (D16).

    `thread.quit` kuyruklu baglantiyla ana is parcacigina gidiyordu ve bloke
    ana is parcacigi onu hic calistiramiyordu: bekleme zaman asimina kadar
    (10 s) suruyordu.
    """
    _qt_app()
    from app.ui.worker import Runner

    def job(token, stage):  # type: ignore[no-untyped-def]
        while not token.cancelled:
            time.sleep(0.01)
        return None

    runner = Runner()
    runner.start(job)
    time.sleep(0.1)
    runner.cancel()
    started = time.perf_counter()
    assert runner.wait(5000)
    assert time.perf_counter() - started < 1.0


def test_finished_jobs_leave_no_listeners_behind(ffmpeg_tools) -> None:  # type: ignore[no-untyped-def]
    """Her is failed/cancelled alicisi birakiyordu (15 calistirmada 16 alici, D19)."""
    app = _qt_app()
    from app.core.settings import Settings
    from app.ui.main_window import MainWindow

    window = MainWindow(ffmpeg_tools, Settings(language="en"))
    tab = window.analyze
    before = tab.runner.receivers(tab.runner.failed)
    for _ in range(3):
        tab._connect_once(lambda result, seconds: None)
        tab.runner.start(lambda token, stage: None)
        deadline = time.time() + 10
        while tab.runner.busy and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
    assert tab.runner.receivers(tab.runner.failed) == before
    window.close()


def test_encode_tab_greys_out_missing_encoders_and_follows_the_spec() -> None:
    _qt_app()
    from PyQt6.QtGui import QStandardItemModel

    from app.core.ffmpeg_locate import Capabilities, FFmpegTools
    from app.core.settings import Settings
    from app.encode import matrix
    from app.ui.tab_encode import EncodeTab

    caps = Capabilities(
        version="test",
        build_flags=frozenset(),
        filters=frozenset(),
        encoders=frozenset({"libopus", "libmp3lame", "aac", "flac"}),
    )
    tab = EncodeTab(FFmpegTools(Path("ffmpeg"), Path("ffprobe"), caps), Settings())
    model = tab.codec.model()
    assert isinstance(model, QStandardItemModel)
    for i, spec in enumerate(matrix.SPECS):
        item = model.item(i)
        assert item is not None
        assert item.isEnabled() == (spec.encoder in caps.encoders), spec.key
    assert tab.spec.key == "opus"  # ilk kullanilabilir

    tab.codec.setCurrentIndex(tab.codec.findData("mp3"))
    assert not tab.mode_quality.isHidden()
    tab.mode_quality.setChecked(True)
    tab.quality.setValue(0)
    choice = tab.choice()
    assert choice.mode == "quality" and choice.quality == 0
    assert matrix.describe(choice) == "mp3V0"

    tab.codec.setCurrentIndex(tab.codec.findData("aac"))
    assert tab.mode_quality.isHidden()
    tab._option_boxes["aac_pns"].setCurrentIndex(1)
    assert not tab.warning.isHidden() and "SNR" in tab.warning.text()

    tab.codec.setCurrentIndex(tab.codec.findData("flac"))
    assert tab.choice().mode == "lossless"
    assert not tab.start_button.isEnabled()  # kaynak yok
    tab.close()


@pytest.mark.needs_ffmpeg
def test_encode_then_compare_runs_through_the_window(ffmpeg_tools, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Kodla -> Analiz sekmesine gec -> karsilastir, tek tikla."""
    app = _qt_app()
    import app.core.settings as settings_mod
    from app.core.settings import Settings
    from app.ui.main_window import MainWindow

    ff = str(ffmpeg_tools.ffmpeg)
    source = tmp_path / "src.flac"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:sample_rate=44100:duration=12:seed=6,tremolo=f=1.1:d=0.85",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(source),
        ],
        check=True,
    )
    original_save = settings_mod.save
    settings_mod.save = lambda s, path=None: True  # type: ignore[assignment]
    try:
        window = MainWindow(ffmpeg_tools, Settings(language="en"))
        enc = window.encode
        enc.source.set_path(source)
        enc.folder.setText(str(tmp_path / "out"))
        enc.codec.setCurrentIndex(enc.codec.findData("opus"))
        enc.bitrate.setCurrentIndex(enc.bitrate.findData(64))
        progress: list[str] = []
        enc.runner.stage.connect(progress.append)
        enc.start()
        deadline = time.time() + 180
        while (
            enc.runner.busy or window.analyze.runner.busy or window.analyze.last_result is None
        ) and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)
        for _ in range(10):
            app.processEvents()
    finally:
        settings_mod.save = original_save

    written = tmp_path / "out" / "src_enc_opus64k.opus"
    assert written.exists()
    assert any(k.startswith("progress:") for k in progress)
    assert window.tabs.currentWidget() is window.analyze
    result = window.analyze.last_result
    assert result is not None and result.status == "measured"
    assert window.analyze.test.path == written
    # Ikinci kodlama ayni adi ezmez
    assert enc.planned_output() == tmp_path / "out" / "src_enc_opus64k_2.opus"
    window.close()


@pytest.mark.needs_ffmpeg
def test_switching_language_keeps_the_work(ffmpeg_tools, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Dil degisince sonuc, iz secimi ve kodlama ayarlari kayboluyordu (D18).

    Meskulken reddedilen degisimde menu yanlis dili isaretli birakiyordu (D20).
    """
    app = _qt_app()
    from app.core.settings import Settings
    from app.encode.matrix import EncodeSettings
    from app.ui.main_window import MainWindow

    ff = str(ffmpeg_tools.ffmpeg)
    reference = tmp_path / "two.mka"
    noise = "anoisesrc=color=pink:sample_rate=44100:duration=8:seed={s},tremolo=f=1.1:d=0.85"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            noise.format(s=1),
            "-f",
            "lavfi",
            "-i",
            noise.format(s=2),
            "-map",
            "0",
            "-map",
            "1",
            "-ac",
            "2",
            "-c:a",
            "flac",
            str(reference),
        ],
        check=True,
    )
    test = tmp_path / "second.opus"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-i",
            str(reference),
            "-map",
            "0:a:1",
            "-c:a",
            "libopus",
            "-b:a",
            "96k",
            str(test),
        ],
        check=True,
    )
    window = MainWindow(ffmpeg_tools, Settings(language="en"))
    tab = window.analyze
    tab.load(reference, test)
    tab.reference.select_stream(1)
    tab.start_compare()
    deadline = time.time() + 120
    while (tab.runner.busy or tab.last_result is None) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    result = tab.last_result
    assert result is not None and result.status == "measured"

    window.encode.source.set_path(reference)
    window.encode.source.select_stream(1)
    choice = EncodeSettings(codec="mp3", mode="quality", quality=2)
    window.encode.apply_choice(choice)
    window.encode.folder.setText(str(tmp_path))

    window.set_language("tr")
    tab = window.analyze
    assert tab.reference.stream_index == 1
    assert tab.last_result is result and tab.results.table.rowCount() == len(result.bands)
    assert tab.results.ladder_button.isEnabled()
    assert window.encode.source.stream_index == 1
    kept = window.encode.choice()
    assert (kept.codec, kept.mode, kept.quality) == ("mp3", "quality", 2)
    assert window.encode.folder.text() == str(tmp_path)

    # Mesgulken dil degisimi reddedilir; menu gecerli dili gostermeli.
    def busy(token, stage):  # type: ignore[no-untyped-def]
        while not token.cancelled:
            time.sleep(0.01)

    tab.runner.start(busy)
    window._language_actions["en"].trigger()
    assert window.settings.language == "tr"
    assert window._language_actions["tr"].isChecked()
    tab.runner.cancel()
    tab.runner.wait(5000)
    for _ in range(10):
        app.processEvents()
    window.close()
