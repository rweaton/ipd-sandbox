#!/usr/bin/env python3
"""
Train an RL agent to play the Iterated Prisoner's Dilemma.

Examples
--------
Train PPO against Tit-for-Tat only:
    python scripts/train_agent.py --algo ppo --opponents tit_for_tat --timesteps 100000

Train DQN against a mixture of the classic strategy set:
    python scripts/train_agent.py --algo dqn --opponents classic --timesteps 200000

Train A2C against every advanced strategy, save model + training-curve plot:
    python scripts/train_agent.py --algo a2c --opponents advanced --timesteps 300000 \\
        --save-path outputs/models/a2c_advanced --plot outputs/plots/a2c_advanced_training.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ipd_sandbox.strategies import STRATEGY_REGISTRY, CLASSIC_STRATEGIES, ADVANCED_STRATEGIES
from ipd_sandbox.train import train_agent
from ipd_sandbox.visualize import plot_training_curve


def resolve_opponents(spec: str):
    if spec == "classic":
        return CLASSIC_STRATEGIES
    if spec == "advanced":
        return ADVANCED_STRATEGIES
    names = [s.strip() for s in spec.split(",")]
    for n in names:
        if n not in STRATEGY_REGISTRY:
            raise SystemExit(f"Unknown strategy '{n}'. Available: {sorted(STRATEGY_REGISTRY)}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algo", choices=["ppo", "dqn", "a2c"], default="ppo")
    parser.add_argument(
        "--opponents",
        default="tit_for_tat",
        help="'classic', 'advanced', or a comma-separated list of strategy names",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-rounds", type=int, default=200, help="IPD rounds per episode")
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-path", default=None, help="Where to save the trained model (no extension)")
    parser.add_argument("--plot", default=None, help="Where to save the training-curve PNG")
    args = parser.parse_args()

    opponents = resolve_opponents(args.opponents)
    print(f"Training {args.algo.upper()} against: {opponents}")

    model, callback = train_agent(
        algo_name=args.algo,
        opponents=opponents,
        total_timesteps=args.timesteps,
        n_rounds=args.n_rounds,
        history_length=args.history_length,
        seed=args.seed,
        save_path=args.save_path,
    )

    if args.save_path:
        print(f"Saved model to {args.save_path}.zip")

    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig = plot_training_curve(
            callback.episode_rewards,
            callback.episode_coop_rates,
            title=f"{args.algo.upper()} vs {args.opponents}",
        )
        fig.savefig(args.plot, dpi=150)
        print(f"Saved training curve to {args.plot}")

    if callback.episode_rewards:
        print(f"Final ~20-episode mean reward/round: {sum(callback.episode_rewards[-20:]) / min(20, len(callback.episode_rewards)):.3f}")
        print(f"Final ~20-episode mean cooperation rate: {sum(callback.episode_coop_rates[-20:]) / min(20, len(callback.episode_coop_rates)):.3f}")


if __name__ == "__main__":
    main()
