# One-System Estimation / Oh Hell (RL + Distributional Bidding)

Train an agent to play the card game **Estimation (Oh Hell)** with an integrated system:

* **Distributional Bidder (PyTorch):** predicts $P(\text{tricks}=k\mid \text{hand}, \text{trump})$ and chooses the bid that **maximizes expected Estimation score**.
* **Playing Agent (MaskablePPO):** learns to realize the chosen bid (controls **seat 0**).
* **Opponents (seats 1–3):** follow suit and play a high legal card with slight randomness (±2 ranks).
* **Alternating training:** RL player improves → collect realized trick counts → supervised bidder calibration → repeat.
* **Evaluation:** deterministic rollouts, plus a **13-deal match** under standard Oh Hell scoring.
* **Logging:** TensorBoard + CSV with periodic evaluation; optional performance plots.

> Single entrypoint: `end_to_end_estimation.py`

---

## Quickstart

```bash
# 1) Create & activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install "stable-baselines3>=2.3.0" "sb3-contrib>=2.3.0" torch gymnasium numpy pandas matplotlib

# 3) Run a full experiment (example)
python end_to_end_estimation.py \
  --cycles 5 \
  --player_steps 2000000 \
  --bidder_rollouts 120000 \
  --bidder_epochs 8 \
  --eval_episodes 300 \
  --log_dir runs/exp1 \
  --eval_freq 20000 \
  --n_envs 8
```

Open TensorBoard to monitor training:

```bash
tensorboard --logdir runs/exp1
```

---

## What’s inside?

### Environment

A custom **4-player trick-taking** environment (Gymnasium) with:

* 52-card deck, 13 cards per player, follow-suit enforced.
* Action masking for legal moves (via `ActionMasker`).
* Shaping reward nudging the agent toward **reducing |won − bid|** during play.
* Terminal reward = Oh Hell score: **$10+b$** if exact, else **$-|\,\text{won} - \text{bid}\,|$**.

### Bidder (PyTorch)

* Input: one-hot **52 cards** in hand + one-hot **4 trumps** → 56-D vector.
* Architecture: `Linear(56→256)→ReLU→Dropout→Linear(256→256)→ReLU→Dropout→Linear(256→14)`
* Output: logits over tricks $k\in\{0..13\}$ → softmax distribution $p(k)$.
* **Bid selection**: argmax over expected Estimation score

  $$
  \arg\max_b\ p(b)(10+b) + \sum_k p(k)\cdot(-|k-b|).
  $$

### Training Loop (Alternating)

1. **Train player (MaskablePPO)** for `player_steps`.
2. **Collect bidder dataset**: run episodes with current player, label = realized tricks (seat 0).
3. **Supervised train bidder** for `bidder_epochs`.
4. **Rebind environments** to the updated bidder and repeat.

### Evaluation

* **Periodic eval** during training (`--eval_freq`) logs:

  * Mean return ± std
  * Exact-hit rate (won == bid)
  * Within ±1 rate (|won − bid| ≤ 1)
* **Final eval** with `--eval_episodes`.
* **13-deal match**: prints per-deal bids/wins/points and totals; also writes a CSV.

---

##  CLI Arguments

| Flag                | Type |       Default | Description                                           |
| ------------------- | ---: | ------------: | ----------------------------------------------------- |
| `--cycles`          |  int |           `3` | Alternating training cycles (player → bidder).        |
| `--player_steps`    |  int |      `600000` | PPO timesteps per cycle.                              |
| `--bidder_rollouts` |  int |       `30000` | Episodes to collect for bidder supervision per cycle. |
| `--bidder_epochs`   |  int |           `6` | Supervised epochs for bidder per cycle.               |
| `--seed`            |  int |          `42` | Global seed.                                          |
| `--save_dir`        |  str |       `ckpts` | Directory for model checkpoints.                      |
| `--eval_episodes`   |  int |         `100` | Episodes for final evaluation.                        |
| `--log_dir`         |  str |    `runs/exp` | TensorBoard/CSV log directory.                        |
| `--eval_freq`       |  int |       `50000` | Eval frequency (trainer steps).                       |
| `--tb_log_name`     |  str | `MaskablePPO` | TensorBoard run name prefix.                          |
| `--n_envs`          |  int |           `8` | Parallel envs (SubprocVecEnv).                        |

> **Note:** Rollout size is kept roughly constant (target 1024) by adjusting `n_steps = 1024 / n_envs`.

---

##  Outputs

* `ckpts/`

  * `player_cycle{N}.zip`, `bidder_cycle{N}.pt` after each cycle
  * `player_final.zip`, `bidder_final.pt` at the end
* `runs/exp*/`

  * `progress.csv`, TensorBoard logs, per-cycle CSVs (`cycle_{N}.csv`)
  * `final_eval.csv` with summary metrics
  * `match13_deals.csv` with per-deal table + totals
  * `plots/` with PNG/PDF learning curves (mean return, exact-hit rate, within ±1)

---

## Plots & Tracking

At the end of training, the script tries to generate summary plots under `runs/.../plots/`:

* **Mean Return vs Timesteps**
* **Exact-Hit Rate vs Timesteps**
* **Within ±1 Rate vs Timesteps**

You can also call `plot_learning_curves(log_dir=..., out_dir=...)` directly if needed.

---

##  Tips for Good Performance

* **Time steps help**: More PPO timesteps generally improve mean return and exact-hit rate.
* **Learning rate stability**: Too large LR can destabilize PPO; the default `3e-4` is a safe choice.
* **Parallelism**: Increase `--n_envs` for faster wall-clock (the code adjusts steps per env).
* **Deterministic eval**: The evaluation and 13-deal match use deterministic actions for the agent.
* **Reproducibility**: Use `--seed` and keep dependency versions consistent.

---

##  Dependencies

* Python 3.9+ (recommended)
* PyTorch (for Bidder)
* Gymnasium
* Stable-Baselines3 ≥ 2.3.0
* sb3-contrib ≥ 2.3.0 (for MaskablePPO + ActionMasker)
* NumPy, Pandas, Matplotlib

Install in one line:

```bash
pip install "stable-baselines3>=2.3.0" "sb3-contrib>=2.3.0" torch gymnasium numpy pandas matplotlib
```

---

## How it Works (High-Level)

1. **Bidder** encodes a hand + trump as a 56-D vector and outputs a **distribution over tricks**.
2. It selects the **payoff-aware bid** maximizing expected Oh Hell score.
3. **Player** (MaskablePPO) receives masked actions that enforce follow-suit, and a shaped reward to track the bid.
4. After an RL phase, the system collects **realized trick counts** and **re-trains** the bidder, aligning bids with what the player can actually achieve.
5. Repeat to **co-adapt** bidder and player.

---

##  13-Deal Match

After final eval, the script runs a full **13-deal match** vs heuristic opponents, printing:

* Bids per seat
* Tricks won per seat
* Points per deal and cumulative totals
* Final ranking

A CSV is saved to `runs/.../match13_deals.csv`.



