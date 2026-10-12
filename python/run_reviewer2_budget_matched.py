"""Budget-matched control for the Reviewer 2 component ablation.

The original reduced ablation trained the no-curriculum and direct-action
variants for 50,000 steps, whereas the saved curriculum FB policy used a
100,000-step FB stage. To remove this training-budget confound, this script
retrains the no-curriculum and direct-action variants for the SAME 100,000-step
budget and compares them against the saved curriculum FB policy on the nine
severe SBGR days. It also records the separation-policy occurrence distribution
for the 100,000-step variants.
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
    MODEL_DIR,
    OUT_DIR,
    SEP_MAX,
    SEP_MIN,
    SEP_STEP,
    SEVERE_DAYS_9,
    load_saved_full_model,
    load_setup,
    make_direct_env,
    make_fb_env,
    metrics_from_solution,
    simulate_model,
    summarize,
    train_or_load_model,
)

SEP_LEVELS = list(range(SEP_MIN, SEP_MAX + 1, SEP_STEP))
TIMESTEPS = 100000
SEED = 42


def seps_from_env(env) -> List[int]:
    return [int(round(s)) for s in env.results[env.day]["separations"]]


def run() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    setup = load_setup(verbose=True)

    rows: List[Dict[str, object]] = []
    counts: Dict[str, np.ndarray] = {
        v: np.zeros(len(SEP_LEVELS), dtype=int)
        for v in ["Full FB saved (100k)", "FB no curric. (100k)", "Direct sep. RL (100k)"]
    }

    def add(day, variant, metrics, train_seconds, loaded):
        row = {"day": day, "variant": variant, "train_seconds": train_seconds,
               "model_loaded": loaded, "train_timesteps": TIMESTEPS}
        row.update(metrics)
        rows.append(row)

    def tally(variant, seps):
        for s in seps:
            nearest = min(SEP_LEVELS, key=lambda L: abs(L - s))
            counts[variant][SEP_LEVELS.index(nearest)] += 1

    for day in SEVERE_DAYS_9:
        print(f"\n=== {day} ===", flush=True)

        # Reference: saved curriculum FB policy (100k FB stage + warm start).
        model, env = load_saved_full_model(setup, day)
        m = simulate_model(model, env)
        add(day, "Full FB saved (100k)", m, 0.0, True)
        tally("Full FB saved (100k)", seps_from_env(env))
        print(f"  Full FB saved (100k)   obj={m['objective']:.1f}", flush=True)

        # No-curriculum FB trained from scratch at the SAME 100k budget.
        env_nc = make_fb_env(setup, day)
        path = MODEL_DIR / f"no_curr_{TIMESTEPS}_{day}.zip"
        model_nc, elapsed, loaded = train_or_load_model(path, env_nc, TIMESTEPS, SEED)
        eval_env_nc = make_fb_env(setup, day)
        model_nc.set_env(eval_env_nc)
        m = simulate_model(model_nc, eval_env_nc)
        add(day, "FB no curric. (100k)", m, elapsed, loaded)
        tally("FB no curric. (100k)", seps_from_env(eval_env_nc))
        print(f"  FB no curric. (100k)   obj={m['objective']:.1f} ({elapsed:.0f}s, loaded={loaded})", flush=True)

        # Direct absolute-separation RL at the SAME 100k budget.
        env_d = make_direct_env(setup, day)
        path = MODEL_DIR / f"direct_sep_{TIMESTEPS}_{day}.zip"
        model_d, elapsed, loaded = train_or_load_model(path, env_d, TIMESTEPS, SEED)
        eval_env_d = make_direct_env(setup, day)
        model_d.set_env(eval_env_d)
        m = simulate_model(model_d, eval_env_d)
        add(day, "Direct sep. RL (100k)", m, elapsed, loaded)
        tally("Direct sep. RL (100k)", seps_from_env(eval_env_d))
        print(f"  Direct sep. RL (100k)  obj={m['objective']:.1f} ({elapsed:.0f}s, loaded={loaded})", flush=True)

        partial = pd.DataFrame(rows)
        partial.to_csv(OUT_DIR / "reviewer2_budget_matched_results_partial.csv", index=False)
        summarize(partial).to_csv(OUT_DIR / "reviewer2_budget_matched_summary_partial.csv", index=False)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "reviewer2_budget_matched_results.csv", index=False)
    summary.to_csv(OUT_DIR / "reviewer2_budget_matched_summary.csv", index=False)

    # Separation-policy occurrence distribution for the 100k variants.
    dist_rows = []
    for variant, c in counts.items():
        total = int(c.sum())
        row = {"variant": variant, "total_steps": total}
        for i, lvl in enumerate(SEP_LEVELS):
            row[f"pct_{lvl}"] = 100.0 * int(c[i]) / total if total else 0.0
        row["pct_le10"] = 100.0 * int(c[0] + c[1]) / total if total else 0.0
        dist_rows.append(row)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(OUT_DIR / "reviewer2_budget_matched_sep_distribution.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Budget-matched summary (100k) ===")
    print(summary[["variant", "days", "objective", "queue_cost", "total_delay_min",
                   "mean_sep_nm", "sep_changes", "config_changes", "train_seconds"]].to_string(index=False))
    print("\n=== Budget-matched separation distribution (100k) ===")
    print(dist.round(1).to_string(index=False))


if __name__ == "__main__":
    run()
