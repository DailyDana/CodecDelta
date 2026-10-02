"""ABX oturumu: deneme akisi, istatistik modu ve durust sonuc metni.

Qt IMPORT ETMEZ; ses motorundan ve arayuzden bagimsiz sinanir.

Kurallar (plan, "ABX kor dinleme testi"):

- Her denemede X, A ya da B'dir; secim kriptografik rastgele (`SystemRandom`).
  Testler tohumlu bir `Random` verir.
- Deneme sirasinda geri bildirim YOK: ogrenme denemeleri bagimsiz olmaktan
  cikarir. Geri bildirim yalnizca `practice` modunda; o mod loglanmaz ve p
  uretmez.
- Iki istatistik modu ASLA karistirilmaz:
  * `fixed`: n deneme. Erken "durdur" testi IPTAL eder ve p gosterilmez;
    "p<0.05 gorunce dur" alfayi bozar.
  * `sprt`: Wald sirali testi (H1 p=0.75, alfa 0.05, beta 0.10, n_max 50).
    Cikti p degil bir KARARDIR.
- Gecis yapilmadan ya da 1.5 s icinde verilen yanit "dinlenmeden" isaretlenir;
  silinmez, sayilir ama raporda gorunur.
- Ayni cift icin birden fazla kesit denendiyse alfa Sidak ile duzeltilir.
- Sonuc metni "fark yok" DEMEZ. Negatif sonuc "fark GOSTERILEMEDI"dir ve
  guven araligi %50'yi iceriyorsa "belirsiz" diye isaretlenir.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core import stats
from app.core.messages import Message

Mode = Literal["fixed", "sprt", "practice"]
Source = Literal["A", "B"]
ExcerptKind = Literal["critical", "random", "manual"]

DEFAULT_TRIALS = 16
SPRT_N_MAX = 50
# Bundan kisa surede ya da hic gecis yapilmadan verilen yanit "dinlenmeden".
MIN_LISTEN_MS = 1500.0
ALPHA = 0.05


@dataclass(frozen=True)
class Trial:
    x_is: Source
    answer: Source
    switches: int
    elapsed_ms: float

    @property
    def correct(self) -> bool:
        return self.answer == self.x_is

    @property
    def unheard(self) -> bool:
        """Gecis yapilmadan ya da cok hizli verilen yanit."""
        return self.switches == 0 or self.elapsed_ms < MIN_LISTEN_MS


@dataclass(frozen=True)
class Excerpt:
    """Dinlenen kesit ve nasil secildigi (sonucun hangi iddiayi destekledigi)."""

    kind: ExcerptKind
    start_s: float
    length_s: float

    @property
    def generalizes(self) -> bool:
        """Yalnizca rastgele kesit dosyanin geneli hakkinda konusur."""
        return self.kind == "random"


@dataclass
class AbxSession:
    mode: Mode
    excerpt: Excerpt
    trials_planned: int = DEFAULT_TRIALS
    # Bu cift icin denenen kesit sayisi (bu dahil); Sidak duzeltmesi icin.
    excerpts_tried: int = 1
    rng: random.Random = field(default_factory=random.SystemRandom)
    trials: list[Trial] = field(default_factory=list)
    stopped: bool = False
    _x: Source = field(init=False)
    _switches: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode == "fixed" and self.trials_planned < 1:
            raise ValueError("en az bir deneme gerekir")
        self._x = self._draw()

    # -- akis ---------------------------------------------------------------

    def _draw(self) -> Source:
        return "A" if self.rng.random() < 0.5 else "B"

    @property
    def x_source(self) -> Source:
        """X'in su anki gercek kaynagi. YALNIZCA ses motoru icin; arayuz gostermez."""
        return self._x

    def note_switch(self) -> None:
        """Dinleyici A, B ya da X arasinda gecis yapti."""
        self._switches += 1

    @property
    def finished(self) -> bool:
        if self.stopped:
            return True
        if self.mode == "fixed":
            return len(self.trials) >= self.trials_planned
        if self.mode == "sprt":
            return self.sprt_decision != "continue"
        return False

    def answer(self, choice: Source, elapsed_ms: float) -> bool | None:
        """Yaniti kaydeder ve yeni X cizer.

        Donus: pratik modunda yanitin dogru olup olmadigi; diger modlarda None
        (deneme sirasinda geri bildirim yok).
        """
        if self.finished:
            raise RuntimeError("oturum bitti")
        trial = Trial(self._x, choice, self._switches, elapsed_ms)
        self.trials.append(trial)
        self._switches = 0
        self._x = self._draw()
        return trial.correct if self.mode == "practice" else None

    def stop(self) -> None:
        """Erken durdurma. Sabit-n'de test iptal edilir: p verilmez."""
        self.stopped = True

    # -- sonuc --------------------------------------------------------------

    @property
    def correct(self) -> int:
        return sum(t.correct for t in self.trials)

    @property
    def unheard(self) -> int:
        return sum(t.unheard for t in self.trials)

    @property
    def alpha(self) -> float:
        return stats.sidak_alpha(ALPHA, self.excerpts_tried)

    @property
    def sprt_decision(self) -> stats.SprtDecision:
        n = len(self.trials)
        return stats.sprt_decision(self.correct, n - self.correct, n_max=SPRT_N_MAX)

    @property
    def complete(self) -> bool:
        """Sonuc istatistik olarak yorumlanabilir mi (iptal edilmedi, tamamlandi)?"""
        if self.mode == "fixed":
            return not self.stopped and len(self.trials) >= self.trials_planned
        if self.mode == "sprt":
            return self.sprt_decision != "continue"
        return False

    def outcome(self) -> stats.TestOutcome | None:
        """Sabit-n tamamlandiysa ozet; aksi halde None (p UYDURULMAZ)."""
        if self.mode != "fixed" or not self.complete:
            return None
        return stats.summarise(len(self.trials), self.correct, alpha=self.alpha)

    def verdict(self) -> Literal["shown", "not_shown", "inconclusive", "none"]:
        """Rozet: fark gosterildi / gosterilemedi (belirsiz) / sonuc yok."""
        if self.mode == "fixed":
            outcome = self.outcome()
            if outcome is None:
                return "none"
            if outcome.p_value < self.alpha:
                return "shown"
            return "inconclusive" if outcome.ci_low <= 0.5 else "not_shown"
        if self.mode == "sprt":
            decision = self.sprt_decision
            if self.stopped and decision == "continue":
                return "none"
            return {
                "accept_h1": "shown",
                "accept_h0": "not_shown",
                "truncated": "inconclusive",
                "continue": "none",
            }[decision]  # type: ignore[return-value]
        return "none"

    def describe(self) -> list[str]:
        """Sonucu anlatan cumleler (cevrilebilir `Message`). "Fark yok" DEMEZ."""
        n, correct = len(self.trials), self.correct
        out: list[str] = []
        if self.mode == "practice":
            out.append(
                Message(
                    "abx.practice",
                    "Practice: {correct}/{n} correct. Answers were shown, nothing is logged and "
                    "no p-value is computed.",
                    correct=correct,
                    n=n,
                )
            )
            return out
        if self.mode == "fixed":
            outcome = self.outcome()
            if outcome is None:
                out.append(
                    Message(
                        "abx.aborted",
                        "The test was stopped after {n} of {planned} trials: no p-value is "
                        "given, because stopping early would bias it.",
                        n=n,
                        planned=self.trials_planned,
                    )
                )
            else:
                params = {
                    "correct": correct,
                    "n": n,
                    "p": outcome.p_value,
                    "acc": outcome.accuracy,
                    "lo": outcome.ci_low,
                    "hi": outcome.ci_high,
                }
                if outcome.p_value < self.alpha:
                    out.append(
                        Message(
                            "abx.shown",
                            "An audible difference was shown: {correct}/{n} correct, "
                            "p = {p:.3f}. Accuracy {acc:.0%} (95% CI {lo:.0%}-{hi:.0%}).",
                            **params,
                        )
                    )
                else:
                    out.append(
                        Message(
                            "abx.not_shown",
                            "An audible difference was NOT shown: {correct}/{n} correct, "
                            "p = {p:.3f}. Accuracy {acc:.0%} (95% CI {lo:.0%}-{hi:.0%}).",
                            **params,
                        )
                    )
                    out.append(
                        Message(
                            "abx.not_same",
                            "This does NOT mean the files sound the same. The test's power was "
                            "about {power:.0%}; at most {hi:.0%} discrimination fits the data.",
                            power=outcome.power,
                            hi=outcome.max_discrimination,
                        )
                    )
                    if outcome.ci_low <= 0.5:
                        out.append(
                            Message(
                                "abx.inconclusive",
                                "The confidence interval includes 50%: the result is "
                                "INCONCLUSIVE, not negative.",
                            )
                        )
        else:
            decision = self.sprt_decision
            if self.stopped and decision == "continue":
                out.append(
                    Message(
                        "abx.sprt_stopped",
                        "The sequential test was stopped after {n} trials before reaching a "
                        "decision.",
                        n=n,
                    )
                )
            elif decision == "accept_h1":
                out.append(
                    Message(
                        "abx.sprt_h1",
                        "Decision (sequential test): an audible difference is shown after {n} "
                        "trials ({correct} correct).",
                        n=n,
                        correct=correct,
                    )
                )
            elif decision == "accept_h0":
                out.append(
                    Message(
                        "abx.sprt_h0",
                        "Decision (sequential test): the listener does not reach the 75% "
                        "level the test looks for ({correct}/{n}). This does NOT mean the files "
                        "sound the same.",
                        n=n,
                        correct=correct,
                    )
                )
            elif decision == "truncated":
                out.append(
                    Message(
                        "abx.sprt_truncated",
                        "No decision after {n} trials (the limit): the result is INCONCLUSIVE.",
                        n=n,
                    )
                )
        if self.excerpts_tried > 1:
            out.append(
                Message(
                    "abx.sidak",
                    "This is excerpt {k} tried for this pair: significance is judged at a "
                    "corrected alpha of {alpha:.4f}.",
                    k=self.excerpts_tried,
                    alpha=self.alpha,
                )
            )
        if self.excerpt.generalizes:
            out.append(
                Message(
                    "abx.random_excerpt",
                    "Random excerpt: the result speaks for the file as a whole.",
                )
            )
        elif self.excerpt.kind == "critical":
            out.append(
                Message(
                    "abx.critical_excerpt",
                    "Critical excerpt chosen by the engine: the result is about this passage "
                    "only, not the whole file.",
                )
            )
        else:
            out.append(
                Message(
                    "abx.manual_excerpt",
                    "Hand-picked excerpt: the result is about this passage only, not the whole "
                    "file.",
                )
            )
        if self.unheard:
            out.append(
                Message(
                    "abx.unheard",
                    "{count} answer(s) were given without listening (no switch or under 1.5 s); "
                    "they are counted but flagged.",
                    count=self.unheard,
                )
            )
        return out

    def to_log(self) -> dict[str, Any]:
        """Oturum kaydi (JSON'a yazilabilir). Pratik oturumu kaydedilmez."""
        if self.mode == "practice":
            raise ValueError("pratik oturumu loglanmaz")
        outcome = self.outcome()
        return {
            "mode": self.mode,
            "excerpt": {
                "kind": self.excerpt.kind,
                "start_s": self.excerpt.start_s,
                "length_s": self.excerpt.length_s,
            },
            "excerpts_tried": self.excerpts_tried,
            "alpha": self.alpha,
            "trials_planned": self.trials_planned if self.mode == "fixed" else None,
            "stopped": self.stopped,
            "verdict": self.verdict(),
            "correct": self.correct,
            "n": len(self.trials),
            "p_value": outcome.p_value if outcome is not None else None,
            "sprt_decision": self.sprt_decision if self.mode == "sprt" else None,
            "trials": [
                {
                    "x_is": t.x_is,
                    "answer": t.answer,
                    "correct": t.correct,
                    "switches": t.switches,
                    "elapsed_ms": round(t.elapsed_ms),
                    "unheard": t.unheard,
                }
                for t in self.trials
            ],
        }
