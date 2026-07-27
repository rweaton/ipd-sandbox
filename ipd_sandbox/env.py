"""
Gymnasium environment wrapping the Iterated Prisoner's Dilemma.

The RL agent plays a fixed number of rounds against one opponent strategy
per episode. If a *list* of opponent names is supplied, a new opponent is
sampled uniformly at random at the start of every episode -- this is how
you train a single agent to be robust against a whole population of
strategies rather than overfitting to one.

Observation
-----------
A flat vector encoding the last `history_length` rounds:
    [agent_action_t-k, opp_action_t-k, ..., agent_action_t-1, opp_action_t-1]
Actions are encoded as 0.0 (cooperate) / 1.0 (defect). Rounds before the
start of the episode are padded with -1.0 (a value never taken by an actual
action), so the agent can distinguish "no history yet" from "opponent
cooperated".

Action
------
Discrete(2): 0 = cooperate, 1 = defect.

Reward
------
The agent's payoff for the round, per the configurable payoff matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .payoffs import COOPERATE, DEFECT, DEFAULT_R, DEFAULT_S, DEFAULT_T, DEFAULT_P, payoff, validate_payoffs
from .strategies import Strategy, make_strategy


class IPDEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponents: Union[str, List[str], Dict[str, Strategy]] = "tit_for_tat",
        n_rounds: int = 200,
        history_length: int = 5,
        R: float = DEFAULT_R,
        S: float = DEFAULT_S,
        T: float = DEFAULT_T,
        P: float = DEFAULT_P,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        validate_payoffs(R, S, T, P)
        self.R, self.S, self.T, self.P = R, S, T, P
        self.n_rounds = n_rounds
        self.history_length = history_length
        self.render_mode = render_mode

        # Normalize `opponents` into a dict[name -> Strategy instance].
        if isinstance(opponents, str):
            self._opponent_pool = {opponents: make_strategy(opponents)}
        elif isinstance(opponents, dict):
            self._opponent_pool = dict(opponents)
        else:  # list of names
            self._opponent_pool = {name: make_strategy(name) for name in opponents}
        self._opponent_names = list(self._opponent_pool.keys())

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * history_length,), dtype=np.float32
        )

        self._agent_history: List[int] = []
        self._opp_history: List[int] = []
        self._current_opponent: Optional[Strategy] = None
        self._current_opponent_name: Optional[str] = None
        self._round = 0

    # -- gymnasium API -------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random  # gymnasium-provided RNG, seeded reproducibly

        idx = int(rng.integers(0, len(self._opponent_names)))
        self._current_opponent_name = self._opponent_names[idx]
        self._current_opponent = self._opponent_pool[self._current_opponent_name]
        self._current_opponent.reset()

        self._agent_history = []
        self._opp_history = []
        self._round = 0

        obs = self._build_observation()
        info = {"opponent": self._current_opponent_name}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        opp_action = self._current_opponent.act(self._opp_history, self._agent_history)
        agent_reward, opp_reward = payoff(
            action, opp_action, R=self.R, S=self.S, T=self.T, P=self.P
        )

        self._agent_history.append(action)
        self._opp_history.append(opp_action)
        self._round += 1

        terminated = False
        truncated = self._round >= self.n_rounds

        obs = self._build_observation()
        info = {
            "opponent": self._current_opponent_name,
            "opponent_action": opp_action,
            "agent_action": action,
            "opponent_reward": opp_reward,
            "round": self._round,
        }
        return obs, agent_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            a = self._agent_history[-1] if self._agent_history else None
            o = self._opp_history[-1] if self._opp_history else None
            print(f"round {self._round:3d}  agent={a}  opp({self._current_opponent_name})={o}")

    # -- helpers ---------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        k = self.history_length
        agent_hist = self._agent_history[-k:]
        opp_hist = self._opp_history[-k:]
        pad = k - len(agent_hist)

        obs = np.full(2 * k, -1.0, dtype=np.float32)
        for i in range(len(agent_hist)):
            obs[2 * (pad + i)] = float(agent_hist[i])
            obs[2 * (pad + i) + 1] = float(opp_hist[i])
        return obs
