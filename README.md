# IPD Sandbox

A sandbox for training Stable-Baselines3 reinforcement-learning agents to
play the **Iterated Prisoner's Dilemma (IPD)** against a population of
classic and advanced game-theoretic strategies, and for inspecting how they
behave.

## Install

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

## Project layout

```
ipd_sandbox/
  payoffs.py      # payoff matrix (R, S, T, P) + validation
  strategies.py    # opponent strategies (TFT, Grudger, Pavlov, ...)
  env.py           # Gymnasium environment (Discrete(2) actions)
  agents.py        # SB3 agent factory: PPO / DQN / A2C
  train.py         # training loop + episode-stats callback
  evaluate.py      # head-to-head matches, summaries, score matrices
  visualize.py     # matplotlib plotting helpers
scripts/
  train_agent.py       # CLI: train and save a model
  run_tournament.py    # CLI: evaluate saved model(s) and plot results
```

## Opponent strategies

| Name | Behavior |
|---|---|
| `always_cooperate` | Always cooperates |
| `always_defect` | Always defects |
| `random` | Cooperates/defects with fixed probability |
| `tit_for_tat` | Cooperates first, then mirrors opponent's last move |
| `grudger` | Cooperates until first betrayal, then defects forever |
| `pavlov` | Win-Stay-Lose-Shift |
| `tit_for_two_tats` | Only defects after two consecutive opponent defections |
| `generous_tit_for_tat` | TFT that probabilistically forgives a defection |

`classic` = `always_cooperate, always_defect, random, tit_for_tat, grudger`
`advanced` = `classic` + `pavlov, tit_for_two_tats, generous_tit_for_tat`

## Train an agent

```bash
# PPO vs a single strategy
python scripts/train_agent.py --algo ppo --opponents tit_for_tat \
    --timesteps 100000 --save-path outputs/models/ppo_tft \
    --plot outputs/plots/ppo_tft_training.png

# DQN vs the full "advanced" mixture (opponent resampled every episode)
python scripts/train_agent.py --algo dqn --opponents advanced \
    --timesteps 200000 --save-path outputs/models/dqn_advanced \
    --plot outputs/plots/dqn_advanced_training.png

# A2C vs a custom subset
python scripts/train_agent.py --algo a2c --opponents grudger,pavlov,random \
    --timesteps 150000 --save-path outputs/models/a2c_custom
```

## Evaluate / run a tournament

```bash
# Single model vs every advanced strategy
python scripts/run_tournament.py --models ppo:outputs/models/ppo_tft \
    --opponents advanced --out-dir outputs/tournament

# Compare multiple algorithms head-to-head (adds a score-matrix heatmap)
python scripts/run_tournament.py \
    --models ppo:outputs/models/ppo_advanced dqn:outputs/models/dqn_advanced a2c:outputs/models/a2c_advanced \
    --opponents advanced --out-dir outputs/tournament
```

This prints per-opponent score/cooperation stats to the console and saves,
per model and opponent:
- a rolling cooperation-rate-over-time plot
- a move timeline (first ~60 rounds, blue=cooperate/red=defect)
- a bar-chart summary across all opponents

and, when comparing multiple models, a score-matrix heatmap.

## Using it as a library

```python
from ipd_sandbox.train import train_agent
from ipd_sandbox.evaluate import evaluate_agent, summarize_results
from ipd_sandbox.visualize import plot_cooperation_over_time

model, callback = train_agent("ppo", opponents=["tit_for_tat", "grudger"], total_timesteps=50_000)

results = evaluate_agent(model, opponent_names=["tit_for_tat", "always_defect"], n_episodes=3)
print(summarize_results(results))

fig = plot_cooperation_over_time(results["tit_for_tat"][0])
fig.savefig("coop_vs_tft.png")
```

## Customizing the payoff matrix

`IPDEnv` accepts `R`, `S`, `T`, `P` directly (defaults are the classic
Axelrod values 3/0/5/1) and validates `T > R > P > S` and `2R > T + S`:

```python
from ipd_sandbox.env import IPDEnv
env = IPDEnv(opponents="tit_for_tat", R=4, S=-1, T=6, P=0)
```

## Adding a new strategy

Subclass `Strategy` in `strategies.py` and implement `act(own_history,
opp_history)` (and `reset()` if it needs internal state), then add it to
`STRATEGY_REGISTRY`. It will automatically become available via
`--opponents your_strategy_name` in both CLI scripts.

## Notes

- Observations encode the last `history_length` rounds as
  `[agent_t-k, opp_t-k, ..., agent_t-1, opp_t-1]`, padded with `-1` before
  the episode has enough history.
- Training against a *list* of opponents resamples one uniformly at the
  start of every episode, which is how you get an agent that generalizes
  across strategies rather than overfitting to one.
- All three algorithms (PPO, DQN, A2C) work with the same environment
  since actions are `Discrete(2)`.
