"""
Factory for creating Stable-Baselines3 agents to play the IPD.

Supports PPO, DQN, and A2C out of the box, with sensible default
hyperparameters for a small discrete-observation task like this one. All
three share the same Gymnasium `Env`/`Discrete(2)` interface, so switching
algorithms is a one-line change.
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

ALGO_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
}

# Reasonable starting hyperparameters for a tiny, low-dimensional task.
# These are intentionally conservative; tune freely for your experiments.
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "ppo": dict(
        policy="MlpPolicy",
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
    ),
    "dqn": dict(
        policy="MlpPolicy",
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=0,
    ),
    "a2c": dict(
        policy="MlpPolicy",
        n_steps=8,
        learning_rate=7e-4,
        gamma=0.99,
        verbose=0,
    ),
}


def make_agent(
    algo_name: str,
    env: gym.Env,
    hyperparams: Dict[str, Any] | None = None,
    seed: int | None = None,
) -> BaseAlgorithm:
    """Create an untrained SB3 model for the given algorithm name.

    Args:
        algo_name: one of "ppo", "dqn", "a2c".
        env: the environment (a single Gymnasium env; SB3 auto-vectorizes it).
        hyperparams: overrides merged on top of DEFAULT_HYPERPARAMS[algo_name].
        seed: RNG seed for reproducibility.
    """
    algo_name = algo_name.lower()
    if algo_name not in ALGO_REGISTRY:
        raise KeyError(f"Unknown algorithm '{algo_name}'. Available: {sorted(ALGO_REGISTRY)}")

    algo_cls = ALGO_REGISTRY[algo_name]
    kwargs = dict(DEFAULT_HYPERPARAMS[algo_name])
    if hyperparams:
        kwargs.update(hyperparams)

    return algo_cls(env=env, seed=seed, **kwargs)


def load_agent(algo_name: str, path: str) -> BaseAlgorithm:
    """Load a previously saved model. `algo_name` must match how it was trained."""
    algo_name = algo_name.lower()
    if algo_name not in ALGO_REGISTRY:
        raise KeyError(f"Unknown algorithm '{algo_name}'. Available: {sorted(ALGO_REGISTRY)}")
    return ALGO_REGISTRY[algo_name].load(path)
