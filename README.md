# Battleship-PPO-Agent

A reinforcement learning agent for the game **Battleship**, trained with **PPO** and augmented with a lightweight **rule-based prior** and a **parallel Monte-Carlo (MCMC) probability map**. The agent plays on a 10×10 board and learns to sink all ships with as few shots as possible.

---

## Overview

- **Environment**: `gymnasium`-style env with masked discrete actions (no re-shooting the same cell).
- **Algorithm**: PPO with a shared policy–value MLP.
- **Inductive bias**:  
  - **Rule prior** scores cells using ship-length feasibility & “hit-cluster” guidance.  
  - **MCMC probability map** estimates cell-hit likelihood given current observations and remaining ship lengths.
- **Training signal** (dense & shaped):  
  - Step penalty −1  
  - **HIT**: +4 (in addition to step penalty)  
  - **SINK**: +12 (in addition to step penalty)  
  - **WIN**: +20 bonus at episode end

---

## Repository Structure

- **`battleship_env.py`** – RL environment
  - `reset()`: places opponent ships randomly (uses `battleship_original.py`)
  - `step(a)`: applies shot, returns reward & `event ∈ {MISS, HIT, ('SINK', length)}`  
  - **Action mask** prevents selecting already-shot cells
- **`battleship_original.py`** – Board utilities & opponent ship placement (random, valid layouts)
- **`rules.py`** – Rule-based components
  - `length_constraint(obs, remaining)`: feasible placements by ship length (row/col runs)
  - `hit_technique(obs)`: guidance around hit clusters (ends/adjacent)
  - `RuleAgent`: maintains remaining lengths, toggles “guide mode,” updates from env events
- **`monte_carlo_search.py`** – Parallel sampling
  - `monte_carlo_probability_map_parallel(...)`: multi-core sampler that proposes valid full placements consistent with `obs` & remaining ships; returns cell-wise hit probabilities
- **`train_PPO_with_rule.py`** – PPO training & evaluation
  - **Model**: `PolicyValueNet` (256-hidden MLP, Tanh) → logits (100 actions) & value
  - **Rollout features**: concatenates flattened observation (100) with MCMC map (100) → 200-D input
  - **Action selection**: policy logits **+ κ·(rule prior)**, optional **α boost** on the RuleAgent’s pick
  - **Updates**: standard PPO (clipping, GAE, entropy bonus, value loss)

---

## Observation & Action Space

- **Observation**: 10×10 grid with
  - `0` **UNKNOWN**, `1` **HIT**, `2` **MISS**
- **Action**: Discrete **100** (pick a cell).  
  Masking enforces legality (can’t shoot a known cell).

---

## Training Setup (the run you reported)

- **Steps**: `200,000`
- **Horizon (rollout length)**: `2048`
- **Optimizer / LR**: Adam, `3e-4`
- **Mini-batch**: `512`, **Epochs**: `10`
- **Clipping**: `0.2`, **Entropy coef**: `0.01`, **Value coef**: `0.5`, **Grad-clip**: `0.5`
- **Board / Episode**: `board_size=10`, `max_steps=100`
- **Rule / Prior mixing**: `κ = 0.5`, **Rule boost** `α = 1.5`
- **MCMC (training)**: `samples=512`, **refresh every** `12` steps, **cores used**: `8`  
  *(code default supports up to env var `MCMC_CORES`; you ran with 8 cores)*  
- **MCMC (evaluation)**: `samples=2048`, **cores used**: `8`  
- **Device**: CPU/GPU (auto-detect); `torch.set_num_threads(1)`

---

## How Learning Works (end-to-end)

1. **Reset**: env builds a valid ship layout; observation grid is all `UNKNOWN`.
2. **Feature build** each step:
   - Flatten obs (100)  
   - If due (every 12 steps or after hits change), run **parallel MCMC** to get a 10×10 hit-prob map (100)  
   - Concatenate → 200-D input to the network
3. **Policy shaping**:
   - Compute policy logits from the network
   - Add **κ × rule_prior_scores** (length feasibility + hit-cluster guidance)
   - Optionally add **α** to the RuleAgent’s suggested index (if legal)
4. **Interact**: sample an action under the mask, step the env, get shaped reward & event
5. **Storage**: keep transitions (obs_features, action, logp, value, reward, done, mask)
6. **PPO Update**: after a horizon, compute **GAE**, normalize advantages, and update with clipping & entropy
7. **Repeat** until reaching total steps; save `ppo_battleship_torch.pt`

---

## Results (your reported run)

> Episodes were evaluated with the same board rules and max episode length of 100 steps.

| Agent / Setting                 | Avg. Moves ↓ | Avg. Return ↑ |
|---------------------------------|--------------:|--------------:|
| **PPO + Rule prior + MCMC**     | **54.8**      | **71**        |
| **PPO only (no rule, no MCMC)** | 96.0          | 48            |

- **Interpretation**: the inductive bias (rule prior + MCMC map) substantially reduces shots needed and increases shaped return compared to PPO alone.

> *(If you later log win rate, median moves, or learning curves, we can add those plots under a “Results” sub-section.)*

---

## Reproducibility Notes

- The MCMC map respects **length constraints** and existing **HIT/MISS** cells, sampling full consistent boards; the aggregated frequency acts as a probabilistic prior over actions.
- The **RuleAgent** reacts to **HIT** and **SINK** events:
  - switches into “guide mode” around clusters,
  - infers orientation for multi-cell streaks,
  - removes sunk length from the remaining set.
- **Action masking** and shaped rewards make the task learnable with dense feedback, while priors speed up exploration.

---

## Future Work

- Train with a larger number of MCMC samples and recompute the probability map at every step to improve model accuracy.

---

## Files / Entry Points

- Train: `python train_PPO_with_rule.py --train_steps 200000 --horizon 2048 --board_size 10 --max_steps 100`
- Evaluate existing checkpoint:  
  `python train_PPO_with_rule.py --eval_only --model ppo_battleship_torch.pt --episodes_eval 50`

*(Adjust `MCMC_CORES` via environment variable, e.g., `export MCMC_CORES=8`.)*
