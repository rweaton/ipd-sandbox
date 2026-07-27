"""
Training entry points for RL agents in the IPD sandbox.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .agents import make_agent
from .env import IPDEnv
from .payoffs import COOPERATE


class EpisodeStatsCallback(BaseCallback):
    """Records per-episode mean reward and cooperation rate during training,
    so you can plot a learning curve afterwards."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_coop_rates: List[float] = []
        self._reward_buffer: List[float] = []
        self._action_buffer: List[int] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        actions = self.locals.get("actions")
        dones = self.locals.get("dones")
        if rewards is None or actions is None or dones is None:
            return True

        # SB3 vectorized envs: take the (single) env's values.
        self._reward_buffer.append(float(rewards[0]))
        self._action_buffer.append(int(np.asarray(actions).flatten()[0]))

        if dones[0]:
            self.episode_rewards.append(float(np.mean(self._reward_buffer)))
            coop_rate = float(np.mean([a == COOPERATE for a in self._action_buffer]))
            self.episode_coop_rates.append(coop_rate)
            self._reward_buffer = []
            self._action_buffer = []
        return True


def train_agent(
    algo_name: str,
    opponents: Union[str, List[str]],
    total_timesteps: int = 100_000,
    n_rounds: int = 200,
    history_length: int = 5,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """Train an SB3 agent against one opponent or a mixture of opponents.

    Args:
        algo_name: "ppo", "dqn", or "a2c".
        opponents: a single strategy name, or a list of strategy names to
            sample from (uniformly) at the start of each training episode.
        total_timesteps: total environment steps to train for.
        n_rounds: number of IPD rounds per episode.
        history_length: number of past rounds encoded in the observation.
        hyperparams: optional SB3 hyperparameter overrides.
        seed: RNG seed for reproducibility.
        save_path: if given, the trained model is saved here (SB3 appends .zip).

    Returns:
        (model, callback) -- the trained SB3 model and the stats callback,
        the latter holding `.episode_rewards` and `.episode_coop_rates` for
        plotting a learning curve.
    """
    env = IPDEnv(
        opponents=opponents,
        n_rounds=n_rounds,
        history_length=history_length,
    )
    if seed is not None:
        env.reset(seed=seed)

    model = make_agent(algo_name, env, hyperparams=hyperparams, seed=seed)
    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if save_path:
        model.save(save_path)

    return model, callback
