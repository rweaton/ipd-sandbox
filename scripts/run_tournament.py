#!/usr/bin/env python3
"""
Evaluate one or more trained agents against the full strategy pool and
produce summary tables + plots.

Examples
--------
Evaluate a single saved PPO model against the advanced strategy set:
    python scripts/run_tournament.py --models ppo:outputs/models/ppo_advanced \\
        --opponents advanced --out-dir outputs/tournament

Compare PPO, DQN, and A2C models saved earlier:
    python scripts/run_tournament.py \\
        --models ppo:outputs/models/ppo_advanced dqn:outputs/models/dqn_advanced a2c:outputs/models/a2c_advanced \\
        --opponents advanced --out-dir outputs/tournament
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ipd_sandbox.agents import load_agent
from ipd_sandbox.strategies import STRATEGY_REGISTRY, CLASSIC_STRATEGIES, ADVANCED_STRATEGIES
from ipd_sandbox.evaluate import evaluate_agent, summarize_results, score_matrix
from ipd_sandbox.visualize import (
    plot_cooperation_over_time,
    plot_action_timeline,
    plot_summary_bars,
    plot_score_matrix,
)


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


def parse_model_spec(spec: str):
    """'ppo:outputs/models/ppo_advanced' -> ('ppo', 'outputs/models/ppo_advanced')"""
    if ":" not in spec:
        raise SystemExit(f"Model spec '{spec}' must be ALGO:PATH, e.g. ppo:outputs/models/my_model")
    algo, path = spec.split(":", 1)
    return algo.lower(), path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="One or more ALGO:PATH specs")
    parser.add_argument("--opponents", default="advanced")
    parser.add_argument("--n-rounds", type=int, default=200)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/tournament")
    args = parser.parse_args()

    opponents = resolve_opponents(args.opponents)
    os.makedirs(args.out_dir, exist_ok=True)

    model_specs = [parse_model_spec(m) for m in args.models]
    models = {}
    for algo, path in model_specs:
        label = f"{algo}_{os.path.basename(path)}"
        models[label] = load_agent(algo, path)

    # Per-model detailed evaluation.
    for label, model in models.items():
        print(f"\n=== {label} ===")
        results = evaluate_agent(
            model,
            opponent_names=opponents,
            n_rounds=args.n_rounds,
            history_length=args.history_length,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        summary = summarize_results(results)
        for name, stats in summary.items():
            print(
                f"  vs {name:24s} score={stats['mean_agent_score']:7.1f}  "
                f"coop={stats['mean_agent_coop_rate']:.2f}  "
                f"opp_score={stats['mean_opp_score']:7.1f}  opp_coop={stats['mean_opp_coop_rate']:.2f}"
            )

        fig = plot_summary_bars(summary, title=f"{label} performance")
        fig.savefig(os.path.join(args.out_dir, f"{label}_summary.png"), dpi=150)

        # Cooperation-over-time and move timeline for the first episode vs each opponent.
        for name in opponents:
            match = results[name][0]
            fig = plot_cooperation_over_time(match, title=f"{label} vs {name}")
            fig.savefig(os.path.join(args.out_dir, f"{label}_vs_{name}_coop.png"), dpi=150)

            fig = plot_action_timeline(match)
            fig.savefig(os.path.join(args.out_dir, f"{label}_vs_{name}_timeline.png"), dpi=150)

    # Cross-model comparison, if more than one model was supplied.
    if len(models) > 1:
        matrix, agent_names, opp_names = score_matrix(
            models,
            opponent_names=opponents,
            n_rounds=args.n_rounds,
            history_length=args.history_length,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        fig = plot_score_matrix(matrix, agent_names, opp_names)
        fig.savefig(os.path.join(args.out_dir, "score_matrix.png"), dpi=150)
        print(f"\nSaved cross-model score matrix to {args.out_dir}/score_matrix.png")

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
