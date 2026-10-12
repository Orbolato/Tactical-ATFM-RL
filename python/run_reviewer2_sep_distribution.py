"""Separation-policy occurrence analysis for the Reviewer 2 second-round revision.

For each ablation variant evaluated on the nine severe SBGR days, this script
re-simulates the deterministic policy and records the full per-time-step
separation trajectory. It then tabulates the number and percentage of
occurrences of each MIT separation level (5, 10, 15, 20, 25, 30 NM) per variant.

All required models already exist (saved 100,000-step FB policies in
models/Results_2, and the reduced-budget no-curriculum / direct-separation /
10-NM fine-tuned models in models/ablation), so no retraining is
performed. The rule-based heuristic and the 5-NM references are computed
analytically.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from run_reviewer2_ablation import (
    ALPHA,
    BETA,
    DELTA,
    MODEL_DIR,
    ORIGINAL_RESULTS_DIR,
    OUT_DIR,
    SEP_MAX,
    SEP_MIN,
    SEP_STEP,
    SEVERE_DAYS_9,
    DirectSeparationFBEnv,
    NoSmoothingFBEnv,
    load_saved_full_model,
    load_setup,
    make_direct_env,
    make_fb_env,
)
from run_reviewer2_cap_sensitivity import capped_valid_bool

import improved_gym_tdcboptimizer_specific_2 as tdcb_sp

SEP_LEVELS = list(range(SEP_MIN, SEP_MAX + 1, SEP_STEP))


def _sim_and_get_seps(model: MaskablePPO, env) -> List[int]:
    obs, _ = env.reset(training=False)
    done = False
    while not done:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, done, _, _ = env.step(int(action), training=False, evaluate=False)
    return [int(round(s)) for s in env.results[env.day]["separations"]]


def _rule_get_seps(setup, day: str) -> List[int]:
    env = make_fb_env(setup, day)
    obs, _ = env.reset(training=False)
    done = False
    prev_delta = 0
    while not done:
        a_i = int(obs[1])
        d_i = int(obs[2])
        wc = int(obs[3])
        sep = int(obs[5])
        f100 = float(obs[6])
        desired_delta = 0
        if (a_i >= 10 or (wc >= 1 and a_i >= 6) or f100 >= 10) and sep < SEP_MAX:
            desired_delta = 1
        elif a_i <= 3 and sep > SEP_MIN:
            desired_delta = -1
        if desired_delta * prev_delta < 0:
            desired_delta = 0
        mask = env.action_masks()
        candidate_idxs = np.flatnonzero(mask)
        preferred = [idx for idx in candidate_idxs if int(env.possible_actions[idx][3]) == desired_delta + 1]
        if preferred:
            candidate_idxs = np.array(preferred, dtype=int)
        best_idx = int(candidate_idxs[0])
        best_score = -np.inf
        for idx in candidate_idxs:
            action = env.possible_actions[int(idx)]
            rc_f = int(action[0])
            arr_served = int(action[1])
            dep_served = int(action[2])
            s_t = sep + SEP_STEP * (int(action[3]) - 1)
            a_f = a_i - arr_served
            d_f = d_i - dep_served
            switch = 1 if rc_f != int(obs[0]) else 0
            score = -(ALPHA * a_f ** 2 + d_f ** 2) - DELTA * switch * (ALPHA * a_i ** 2 + d_i ** 2)
            score -= BETA * ((s_t - SEP_MIN) / (SEP_MAX - SEP_MIN)) * f100 / max(env.max_f_100, 1)
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        obs, _, done, _, _ = env.step(best_idx, training=False, evaluate=False)
        new_delta = int(env.possible_actions[best_idx][3]) - 1
        if new_delta != 0:
            prev_delta = new_delta
    return [int(round(s)) for s in env.results[env.day]["separations"]]


def collect_variant_seps(setup, day: str, variant: str) -> List[int]:
    if variant == "Full FB saved":
        model, env = load_saved_full_model(setup, day)
        return _sim_and_get_seps(model, env)
    if variant == "FB, no smoothing":
        model, env = load_saved_full_model(setup, day, env_cls=NoSmoothingFBEnv)
        return _sim_and_get_seps(model, env)
    if variant == "FB no curric.":
        env = make_fb_env(setup, day)
        path = MODEL_DIR / f"no_curr_50000_{day}.zip"
        model = MaskablePPO.load(str(path), env=env)
        return _sim_and_get_seps(model, env)
    if variant == "Direct sep. RL":
        env = make_direct_env(setup, day)
        path = MODEL_DIR / f"direct_sep_50000_{day}.zip"
        model = MaskablePPO.load(str(path), env=env)
        return _sim_and_get_seps(model, env)
    if variant == "Rule heuristic":
        return _rule_get_seps(setup, day)
    if variant == "FB saved, 10-NM cap":
        cap_mask = capped_valid_bool(setup, 10)
        env = make_fb_env(setup, day, valid_bool=cap_mask)
        saved_path = ORIGINAL_RESULTS_DIR / f"dcb_ppo_fb_2_{day}.zip"
        model = MaskablePPO.load(str(saved_path), env=env)
        return _sim_and_get_seps(model, env)
    if variant == "FB tuned, 10-NM cap":
        cap_mask = capped_valid_bool(setup, 10)
        env = make_fb_env(setup, day, valid_bool=cap_mask)
        path = MODEL_DIR / f"fb_cap10_finetune_50000_{day}.zip"
        model = MaskablePPO.load(str(path), env=env)
        return _sim_and_get_seps(model, env)
    raise ValueError(f"Unknown variant: {variant}")


VARIANTS = [
    "Full FB saved",
    "FB, no smoothing",
    "FB no curric.",
    "Direct sep. RL",
    "Rule heuristic",
    "FB saved, 10-NM cap",
    "FB tuned, 10-NM cap",
]


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    setup = load_setup(verbose=True)

    # Determine the number of decision steps per day (length of any trajectory).
    long_rows: List[Dict[str, object]] = []
    counts: Dict[str, np.ndarray] = {v: np.zeros(len(SEP_LEVELS), dtype=int) for v in VARIANTS}

    for day in SEVERE_DAYS_9:
        print(f"\n=== {day} ===")
        for variant in VARIANTS:
            seps = collect_variant_seps(setup, day, variant)
            for s in seps:
                if s in SEP_LEVELS:
                    counts[variant][SEP_LEVELS.index(s)] += 1
                else:
                    # Clamp any off-grid value to nearest level (should not happen).
                    nearest = min(SEP_LEVELS, key=lambda L: abs(L - s))
                    counts[variant][SEP_LEVELS.index(nearest)] += 1
            row = {"day": day, "variant": variant, "n_steps": len(seps)}
            for lvl in SEP_LEVELS:
                row[f"n_{lvl}"] = sum(1 for s in seps if s == lvl)
            long_rows.append(row)
            print(f"  {variant:22s} steps={len(seps):3d} mean={np.mean(seps):.2f}")

    # Per-day long table.
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_DIR / "reviewer2_sep_distribution_byday.csv", index=False)

    # Pooled distribution table (counts + percentages) across all nine days.
    pooled_rows = []
    for variant in VARIANTS:
        total = int(counts[variant].sum())
        row = {"variant": variant, "total_steps": total}
        for i, lvl in enumerate(SEP_LEVELS):
            c = int(counts[variant][i])
            row[f"n_{lvl}"] = c
            row[f"pct_{lvl}"] = 100.0 * c / total if total else 0.0
        # Share of steps at or below 10 NM.
        le10 = int(counts[variant][0] + counts[variant][1])
        row["pct_le10"] = 100.0 * le10 / total if total else 0.0
        pooled_rows.append(row)

    # Append the trivial 5-NM references for completeness.
    ref_total = int(long_df.loc[long_df.variant == "Full FB saved", "n_steps"].sum())
    for ref_name in ["Baseline/CM 5-NM", "No-separation LB (5 NM)"]:
        row = {"variant": ref_name, "total_steps": ref_total}
        for lvl in SEP_LEVELS:
            row[f"n_{lvl}"] = ref_total if lvl == 5 else 0
            row[f"pct_{lvl}"] = 100.0 if lvl == 5 else 0.0
        row["pct_le10"] = 100.0
        pooled_rows.append(row)

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(OUT_DIR / "reviewer2_sep_distribution_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("\n=== Pooled separation distribution (counts) ===")
    print(pooled_df[["variant", "total_steps"] + [f"n_{l}" for l in SEP_LEVELS]].to_string(index=False))
    print("\n=== Pooled separation distribution (percentages) ===")
    pct_cols = [f"pct_{l}" for l in SEP_LEVELS] + ["pct_le10"]
    print(pooled_df[["variant"] + pct_cols].round(1).to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'reviewer2_sep_distribution_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'reviewer2_sep_distribution_byday.csv'}")


if __name__ == "__main__":
    main()
