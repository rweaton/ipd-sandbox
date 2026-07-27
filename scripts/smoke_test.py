#!/usr/bin/env python3
"""
Fast end-to-end sanity check: trains each algorithm for a tiny number of
timesteps and runs a short evaluation, to confirm the environment, agent
factory, training loop, and evaluation pipeline all work together correctly
in your installed environment.

This is NOT meant to produce a good agent (timesteps are far too low for
that) -- it's meant to run in well under a minute and catch integration
bugs. Run the real training via scripts/train_agent.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ipd_sandbox.env import IPDEnv
from ipd_sandbox.train import train_agent
from ipd_sandbox.evaluate import evaluate_agent, summarize_results


def check_env_basics():
    env = IPDEnv(opponents=["tit_for_tat", "always_defect"], n_rounds=20, history_length=3)
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,), obs.shape
    assert (obs == -1.0).all(), "fresh reset should be all padding"

    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        steps += 1
    assert steps == 20, steps
    print(f"  IPDEnv: {steps} rounds played vs {info['opponent']}, total reward={total_reward:.1f}  OK")


def check_algo(algo_name: str):
    print(f"  training {algo_name.upper()} for 2,000 timesteps...")
    model, callback = train_agent(
        algo_name=algo_name,
        opponents=["tit_for_tat", "always_defect", "grudger"],
        total_timesteps=2_000,
        n_rounds=20,
        history_length=3,
        seed=0,
    )
    assert len(callback.episode_rewards) > 0, "callback recorded no episodes"

    results = evaluate_agent(
        model,
        opponent_names=["tit_for_tat", "always_defect"],
        n_rounds=20,
        n_episodes=2,
        history_length=3
    )
    summary = summarize_results(results)
    for name, stats in summary.items():
        print(f"    vs {name}: score={stats['mean_agent_score']:.1f} coop_rate={stats['mean_agent_coop_rate']:.2f}")
    print(f"  {algo_name.upper()}: train+eval pipeline OK")


def main():
    print("Checking IPDEnv basics...")
    check_env_basics()

    for algo in ["ppo", "dqn", "a2c"]:
        print(f"\nChecking {algo.upper()} pipeline...")
        check_algo(algo)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
