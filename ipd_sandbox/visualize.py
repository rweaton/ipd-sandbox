"""
Matplotlib plotting helpers for the IPD sandbox.

All functions take/return a matplotlib Figure and don't call plt.show(),
so callers can save, embed, or display them as needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .evaluate import MatchResult


def plot_training_curve(
    episode_rewards: List[float],
    episode_coop_rates: List[float],
    title: str = "Training progress",
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(episode_rewards, color="tab:blue", linewidth=1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean reward per round")
    ax1.set_title("Reward over training")
    ax1.grid(alpha=0.3)

    ax2.plot(episode_coop_rates, color="tab:green", linewidth=1)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cooperation rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Agent cooperation rate over training")
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cooperation_over_time(
    match: MatchResult,
    window: int = 10,
    title: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    rolling = match.rolling_coop_rate(window=window)
    ax.plot(rolling, color="tab:blue", label=f"Agent (rolling, w={window})")

    opp_arr = np.array([a == 0 for a in match.opp_actions], dtype=float)
    if len(opp_arr) >= window:
        kernel = np.ones(window) / window
        opp_rolling = np.convolve(opp_arr, kernel, mode="valid")
        ax.plot(opp_rolling, color="tab:orange", alpha=0.7, label="Opponent (rolling)")

    ax.set_xlabel("Round")
    ax.set_ylabel("Cooperation rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or f"Cooperation over time vs {match.opponent}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_action_timeline(match: MatchResult, max_rounds: int = 60) -> plt.Figure:
    """Show the raw sequence of C/D moves for both players (first `max_rounds` rounds)."""
    n = min(max_rounds, len(match.agent_actions))
    fig, ax = plt.subplots(figsize=(max(8, n * 0.15), 2.5))

    agent = match.agent_actions[:n]
    opp = match.opp_actions[:n]

    ax.scatter(range(n), [1] * n, c=["tab:red" if a else "tab:blue" for a in agent], marker="s", s=60)
    ax.scatter(range(n), [0] * n, c=["tab:red" if a else "tab:blue" for a in opp], marker="s", s=60)

    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"Opp ({match.opponent})", "Agent"])
    ax.set_xlabel("Round")
    ax.set_title("Move timeline (blue = cooperate, red = defect)")
    ax.set_xlim(-0.5, n - 0.5)
    fig.tight_layout()
    return fig


def plot_score_matrix(
    matrix: np.ndarray,
    agent_names: List[str],
    opponent_names: List[str],
    title: str = "Mean agent score by opponent",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.2 * len(opponent_names) + 2, 1 * len(agent_names) + 2))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(opponent_names)))
    ax.set_xticklabels(opponent_names, rotation=45, ha="right")
    ax.set_yticks(range(len(agent_names)))
    ax.set_yticklabels(agent_names)

    for i in range(len(agent_names)):
        for j in range(len(opponent_names)):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean total score")
    fig.tight_layout()
    return fig


def plot_summary_bars(summary: Dict[str, Dict[str, float]], title: str = "Agent performance by opponent") -> plt.Figure:
    names = list(summary.keys())
    scores = [summary[n]["mean_agent_score"] for n in names]
    coop = [summary[n]["mean_agent_coop_rate"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(names) * 1.2), 4))

    ax1.bar(names, scores, color="tab:blue")
    ax1.set_ylabel("Mean total score")
    ax1.set_title("Score vs each strategy")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(names, coop, color="tab:green")
    ax2.set_ylabel("Cooperation rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Agent cooperation rate vs each strategy")
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
