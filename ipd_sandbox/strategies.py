"""
Opponent strategies for the Iterated Prisoner's Dilemma.

Every strategy implements the same small interface:

    reset()                                -> None
        Called at the start of each episode/match.

    act(own_history, opp_history)          -> int (COOPERATE or DEFECT)
        own_history / opp_history are lists of past actions (ints), from the
        strategy's own point of view, NOT including the current round.
        Both lists always have equal length.

Strategies are stateless with respect to `own_history`/`opp_history` (they
are handed the history each call) but may keep small internal state (e.g.
Grudger's "has been betrayed" flag) that is cleared in `reset()`.
"""

from __future__ import annotations

import random
from typing import List

from .payoffs import COOPERATE, DEFECT


class Strategy:
    """Base class for all opponent strategies."""

    name: str = "base"

    def reset(self) -> None:
        pass

    def act(self, own_history: List[int], opp_history: List[int]) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Strategy {self.name}>"


class AlwaysCooperate(Strategy):
    name = "always_cooperate"

    def act(self, own_history, opp_history) -> int:
        return COOPERATE


class AlwaysDefect(Strategy):
    name = "always_defect"

    def act(self, own_history, opp_history) -> int:
        return DEFECT


class RandomStrategy(Strategy):
    name = "random"

    def __init__(self, p_defect: float = 0.5, seed: int | None = None):
        self.p_defect = p_defect
        self._rng = random.Random(seed)

    def reset(self) -> None:
        # Keep the RNG stream going across episodes rather than reseeding,
        # so repeated episodes don't all look identical.
        pass

    def act(self, own_history, opp_history) -> int:
        return DEFECT if self._rng.random() < self.p_defect else COOPERATE


class TitForTat(Strategy):
    """Cooperate first, then mirror the opponent's previous move."""

    name = "tit_for_tat"

    def act(self, own_history, opp_history) -> int:
        if not opp_history:
            return COOPERATE
        return opp_history[-1]


class TitForTwoTats(Strategy):
    """Only defect after the opponent has defected twice in a row."""

    name = "tit_for_two_tats"

    def act(self, own_history, opp_history) -> int:
        if len(opp_history) < 2:
            return COOPERATE
        if opp_history[-1] == DEFECT and opp_history[-2] == DEFECT:
            return DEFECT
        return COOPERATE


class GenerousTitForTat(Strategy):
    """Tit-for-Tat that occasionally forgives a defection."""

    name = "generous_tit_for_tat"

    def __init__(self, forgiveness: float = 0.1, seed: int | None = None):
        self.forgiveness = forgiveness
        self._rng = random.Random(seed)

    def act(self, own_history, opp_history) -> int:
        if not opp_history:
            return COOPERATE
        if opp_history[-1] == DEFECT and self._rng.random() < self.forgiveness:
            return COOPERATE
        return opp_history[-1]


class Grudger(Strategy):
    """Cooperate until the opponent defects once, then defect forever
    (a.k.a. Friedman / "Trigger" strategy)."""

    name = "grudger"

    def __init__(self):
        self._triggered = False

    def reset(self) -> None:
        self._triggered = False

    def act(self, own_history, opp_history) -> int:
        if self._triggered:
            return DEFECT
        if opp_history and opp_history[-1] == DEFECT:
            self._triggered = True
            return DEFECT
        return COOPERATE


class Pavlov(Strategy):
    """Win-Stay, Lose-Shift. Cooperate first. Afterwards: repeat the last
    move if it "won" (own and opponent's last moves matched, i.e. mutual
    cooperation or mutual defection), otherwise switch."""

    name = "pavlov"

    def act(self, own_history, opp_history) -> int:
        if not own_history:
            return COOPERATE
        last_own = own_history[-1]
        last_opp = opp_history[-1]
        won = last_own == last_opp
        return last_own if won else (1 - last_own)


# Registry -----------------------------------------------------------------

def _factory_registry():
    return {
        "always_cooperate": AlwaysCooperate,
        "always_defect": AlwaysDefect,
        "random": RandomStrategy,
        "tit_for_tat": TitForTat,
        "tit_for_two_tats": TitForTwoTats,
        "generous_tit_for_tat": GenerousTitForTat,
        "grudger": Grudger,
        "pavlov": Pavlov,
    }


STRATEGY_REGISTRY = _factory_registry()

CLASSIC_STRATEGIES = [
    "always_cooperate",
    "always_defect",
    "random",
    "tit_for_tat",
    "grudger",
]

ADVANCED_STRATEGIES = CLASSIC_STRATEGIES + [
    "pavlov",
    "tit_for_two_tats",
    "generous_tit_for_tat",
]


def make_strategy(name: str, **kwargs) -> Strategy:
    """Instantiate a strategy by its registry name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)


def make_strategy_set(names: List[str]) -> dict:
    """Instantiate a dict of {name: Strategy} for a list of registry names."""
    return {name: make_strategy(name) for name in names}
