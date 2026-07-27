"""
Evaluation and tournament utilities: pit a trained agent against each
opponent strategy and collect round-by-round and summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from .env import IPDEnv
from .payoffs import COOPERATE
from .strategies import CLASSIC_STRATEGIES


@dataclass
class MatchResult:
    opponent: str
    agent_actions: List[int] = field(default_factory=list)
    opp_actions: List[int] = field(default_factory=list)
    agent_rewards: List[float] = field(default_factory=list)
    opp_rewards: List[float] = field(default_factory=list)

    @property
    def agent_total_score(self) -> float:
        return float(np.sum(self.agent_rewards))

    @property
    def opp_total_score(self) -> float:
        return float(np.sum(self.opp_rewards))

    @property
    def agent_coop_rate(self) -> float:
        return float(np.mean([a == COOPERATE for a in self.agent_actions]))

    @property
    def opp_coop_rate(self) -> float:
        return float(np.mean([a == COOPERATE for a in self.opp_actions]))

    def rolling_coop_rate(self, window: int = 10) -> np.ndarray:
        """Rolling cooperation rate of the agent, for plotting over time."""
        arr = np.array([a == COOPERATE for a in self.agent_actions], dtype=float)
        if len(arr) < window:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="valid")


def play_match(
    model: BaseAlgorithm,
    opponent_name: str,
    n_rounds: int = 200,
    history_length: int = 5,
    deterministic: bool = True,
    seed: int | None = None,
) -> MatchResult:
    """Play a single deterministic (or stochastic) match of the trained
    agent against one named opponent strategy."""
    env = IPDEnv(opponents=[opponent_name], n_rounds=n_rounds, history_length=history_length)
    obs, info = env.reset(seed=seed)
    result = MatchResult(opponent=opponent_name)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        result.agent_actions.append(info["agent_action"])
        result.opp_actions.append(info["opponent_action"])
        result.agent_rewards.append(reward)
        result.opp_rewards.append(info["opponent_reward"])
        done = terminated or truncated

    return result


def evaluate_agent(
    model: BaseAlgorithm,
    opponent_names: List[str] | None = None,
    n_rounds: int = 200,
    history_length: int = 5,
    n_episodes: int = 5,
    deterministic: bool = True,
    seed: int | None = None,
) -> Dict[str, List[MatchResult]]:
    """Play `n_episodes` matches against each opponent strategy.

    Returns a dict {opponent_name: [MatchResult, ...]}.
    """
    opponent_names = opponent_names or CLASSIC_STRATEGIES
    results: Dict[str, List[MatchResult]] = {name: [] for name in opponent_names}

    for name in opponent_names:
        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            match = play_match(
                model,
                name,
                n_rounds=n_rounds,
                history_length=history_length,
                deterministic=deterministic,
                seed=ep_seed,
            )
            results[name].append(match)

    return results


def summarize_results(results: Dict[str, List[MatchResult]]) -> Dict[str, Dict[str, float]]:
    """Collapse per-episode MatchResults into mean summary stats per opponent."""
    summary = {}
    for name, matches in results.items():
        summary[name] = {
            "mean_agent_score": float(np.mean([m.agent_total_score for m in matches])),
            "mean_opp_score": float(np.mean([m.opp_total_score for m in matches])),
            "mean_agent_coop_rate": float(np.mean([m.agent_coop_rate for m in matches])),
            "mean_opp_coop_rate": float(np.mean([m.opp_coop_rate for m in matches])),
        }
    return summary


def score_matrix(
    models: Dict[str, BaseAlgorithm],
    opponent_names: List[str],
    n_rounds: int = 200,
    history_length: int = 5,
    n_episodes: int = 5,
    seed: int | None = None,
) -> "tuple[np.ndarray, list[str], list[str]]":
    """Build a (n_agents x n_opponents) matrix of mean agent score, for
    comparing multiple trained agents (e.g. PPO vs DQN vs A2C) against the
    same opponent pool in one heatmap.
    """
    agent_names = list(models.keys())
    matrix = np.zeros((len(agent_names), len(opponent_names)))

    for i, agent_name in enumerate(agent_names):
        results = evaluate_agent(
            models[agent_name],
            opponent_names=opponent_names,
            n_rounds=n_rounds,
            history_length=history_length,
            n_episodes=n_episodes,
            seed=seed,
        )
        summary = summarize_results(results)
        for j, opp_name in enumerate(opponent_names):
            matrix[i, j] = summary[opp_name]["mean_agent_score"]

    return matrix, agent_names, opponent_names
