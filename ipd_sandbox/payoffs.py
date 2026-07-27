"""
Payoff matrix for the (Iterated) Prisoner's Dilemma.

Standard labeling:
    R = Reward for mutual cooperation
    S = Sucker's payoff (cooperated while opponent defected)
    T = Temptation to defect (defected while opponent cooperated)
    P = Punishment for mutual defection

A valid Prisoner's Dilemma requires:  T > R > P > S
and, for the *iterated* game to reward cooperation over alternating
exploitation:  2R > T + S
"""

from typing import Tuple

COOPERATE = 0
DEFECT = 1

ACTION_NAMES = {COOPERATE: "C", DEFECT: "D"}

# Classic Axelrod-tournament values.
DEFAULT_R = 3.0
DEFAULT_S = 0.0
DEFAULT_T = 5.0
DEFAULT_P = 1.0


def validate_payoffs(R: float, S: float, T: float, P: float) -> None:
    if not (T > R > P > S):
        raise ValueError(
            f"Payoffs must satisfy T > R > P > S, got T={T}, R={R}, P={P}, S={S}"
        )
    if not (2 * R > T + S):
        raise ValueError(
            f"Payoffs must satisfy 2R > T + S for a well-formed IPD, "
            f"got 2R={2 * R}, T+S={T + S}"
        )


def payoff(
    my_action: int,
    opp_action: int,
    R: float = DEFAULT_R,
    S: float = DEFAULT_S,
    T: float = DEFAULT_T,
    P: float = DEFAULT_P,
) -> Tuple[float, float]:
    """Return (my_reward, opponent_reward) for a single round."""
    if my_action == COOPERATE and opp_action == COOPERATE:
        return R, R
    if my_action == COOPERATE and opp_action == DEFECT:
        return S, T
    if my_action == DEFECT and opp_action == COOPERATE:
        return T, S
    if my_action == DEFECT and opp_action == DEFECT:
        return P, P
    raise ValueError(f"Invalid actions: {my_action}, {opp_action}")
