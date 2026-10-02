"""ABX oturum mantigi testleri (Qt'siz).

Sinanan seyler planin durustluk kurallari: geri bildirim yalnizca pratikte,
erken durdurmada p yok, SPRT karar verir p vermez, "fark yok" hicbir metinde
gecmez, dinlenmeden verilen yanit isaretlenir, coklu kesitte alfa duzelir.
"""

from __future__ import annotations

import json
import random

import pytest

from app.abx.session import AbxSession, Excerpt
from app.ui.i18n import STRINGS

CRITICAL = Excerpt("critical", 12.0, 15.0)
RANDOM = Excerpt("random", 40.0, 15.0)


def play(session: AbxSession, correct: list[bool], *, elapsed_ms: float = 4000.0) -> None:
    for ok in correct:
        session.note_switch()
        x = session.x_source
        answer = x if ok else ("B" if x == "A" else "A")
        session.answer(answer, elapsed_ms)


def test_x_is_random_and_redrawn_every_trial() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=200, rng=random.Random(1))
    seen = []
    for _ in range(200):
        seen.append(session.x_source)
        session.note_switch()
        session.answer("A", 3000.0)
    assert 60 < seen.count("A") < 140
    assert [t.x_is for t in session.trials] == seen


def test_no_feedback_except_in_practice() -> None:
    fixed = AbxSession("fixed", CRITICAL, rng=random.Random(2))
    assert fixed.answer("A", 3000.0) is None
    practice = AbxSession("practice", CRITICAL, rng=random.Random(2))
    x = practice.x_source
    assert practice.answer(x, 3000.0) is True


def test_fixed_session_reports_p_only_when_complete() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=16, rng=random.Random(3))
    play(session, [True] * 13 + [False] * 3)
    assert session.finished and session.complete
    outcome = session.outcome()
    assert outcome is not None and outcome.p_value < 0.05
    assert session.verdict() == "shown"
    with pytest.raises(RuntimeError):
        session.answer("A", 3000.0)


def test_stopping_early_gives_no_p_value() -> None:
    """'p<0.05 gorunce dur' alfayi bozar: erken durdurulan test p vermez."""
    session = AbxSession("fixed", CRITICAL, trials_planned=16, rng=random.Random(4))
    play(session, [True] * 8)
    session.stop()
    assert session.outcome() is None and session.verdict() == "none"
    assert any(getattr(m, "key", "") == "abx.aborted" for m in session.describe())


def test_a_failed_test_is_never_called_no_difference() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=16, rng=random.Random(5))
    play(session, [True, False] * 8)
    assert session.verdict() == "inconclusive"
    text = " ".join(str(m) for m in session.describe()).lower()
    assert "not shown" in text and "does not mean the files sound the same" in text
    assert "inconclusive" in text
    assert "no difference" not in text


def test_sprt_decides_without_a_p_value() -> None:
    session = AbxSession("sprt", RANDOM, rng=random.Random(6))
    while not session.finished:
        play(session, [True])
    # Her dogru ln(1.5) = 0.405 ekler; ust sinir ln(0.9/0.05) = 2.89 -> 8. dogruda.
    assert len(session.trials) == 8 and session.sprt_decision == "accept_h1"
    assert session.outcome() is None and session.verdict() == "shown"
    log = session.to_log()
    assert log["p_value"] is None and log["sprt_decision"] == "accept_h1"


def test_sprt_truncates_at_the_limit() -> None:
    session = AbxSession("sprt", RANDOM, rng=random.Random(7))
    # 3 dogru 2 yanlis tekrari H1/H0 sinirlarina varmadan n_max'a ulasir
    pattern = [True, False, True, True, False]
    while not session.finished:
        play(session, [pattern[len(session.trials) % 5]])
    assert session.sprt_decision == "truncated" and session.verdict() == "inconclusive"
    assert len(session.trials) == 50


def test_unheard_answers_are_flagged_not_dropped() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=4, rng=random.Random(8))
    session.answer("A", 5000.0)  # gecis yok
    session.note_switch()
    session.answer("B", 800.0)  # cok hizli
    play(session, [True, True])
    assert len(session.trials) == 4 and session.unheard == 2
    assert any(getattr(m, "key", "") == "abx.unheard" for m in session.describe())


def test_several_excerpts_tighten_alpha() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=16, excerpts_tried=7)
    assert session.alpha == pytest.approx(1 - 0.95 ** (1 / 7))
    # 12/16 tek kesitte anlamli (p=0.038), yedinci kesitte degil
    play(session, [True] * 12 + [False] * 4)
    assert session.verdict() != "shown"
    assert any(getattr(m, "key", "") == "abx.sidak" for m in session.describe())


def test_excerpt_kind_states_what_the_result_supports() -> None:
    critical = AbxSession("fixed", CRITICAL, trials_planned=1)
    random_one = AbxSession("fixed", RANDOM, trials_planned=1)
    keys = {getattr(m, "key", "") for m in critical.describe()}
    assert "abx.critical_excerpt" in keys
    keys = {getattr(m, "key", "") for m in random_one.describe()}
    assert "abx.random_excerpt" in keys


def test_log_is_json_and_practice_is_not_logged() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=2, rng=random.Random(9))
    play(session, [True, False])
    log = json.loads(json.dumps(session.to_log()))
    assert log["n"] == 2 and len(log["trials"]) == 2
    with pytest.raises(ValueError):
        AbxSession("practice", CRITICAL).to_log()


def test_every_abx_message_is_translated() -> None:
    session = AbxSession("fixed", CRITICAL, trials_planned=16, excerpts_tried=2)
    play(session, [True, False] * 8)
    keys = {getattr(m, "key", "") for m in session.describe()}
    assert keys and all(f"msg.{k}" in STRINGS["tr"] for k in keys)
