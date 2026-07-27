"""
ipd_sandbox
===========

A sandbox for training and studying Stable-Baselines3 reinforcement-learning
agents playing the Iterated Prisoner's Dilemma (IPD) against a variety of
classic and advanced game-theoretic strategies.
"""

from .payoffs import COOPERATE, DEFECT, payoff
from . import strategies

__all__ = ["COOPERATE", "DEFECT", "payoff", "strategies"]

try:
    # IPDEnv requires gymnasium; keep the pure-python parts of this package
    # (payoffs, strategies) importable/testable even if gymnasium/SB3 aren't
    # installed yet.
    from .env import IPDEnv  # noqa: F401

    __all__.append("IPDEnv")
except ImportError:
    pass
