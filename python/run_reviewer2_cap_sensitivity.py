"""MIT cap sensitivity for Reviewer 2 ablation.

This script evaluates whether adding an operational cap to the FB policy reduces
excessive spacing. It reuses the preprocessing and environments from
run_reviewer2_ablation.py.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from run_reviewer2_ablation import (
    MODEL_DIR,
    ORIGINAL_RESULTS_DIR,
    OUT_DIR,
    SEP_MAX,
    SEP_STEP,
    SEVERE_DAYS_9,
    load_setup,
    make_fb_env,
    simulate_model,
    summarize,
)


def capped_valid_bool(setup, cap_nm: int) -> np.ndarray:
    capped = setup.valid_bool.copy()
    for sep in range(capped.shape[4]):
        if sep < 0 or sep > SEP_MAX:
            continue
        for action_idx, action in enumerate(setup.possible_actions):
            final_sep = sep + SEP_STEP * (int(action[3]) - 1)
            if final_sep > cap_nm:
                capped[:, :, :, :, sep, action_idx] = False
    # Guard against empty masks: keep original valid actions if a corner state has no capped action.
    empty = capped.sum(axis=-1) == 0
    if np.any(empty):
        capped[empty] = setup.valid_bool[empty]
    return capped


def add_result(rows: List[Dict[str, object]], day: str, variant: str, metrics: Dict[str, object],
               train_seconds: float, loaded: bool, timesteps: int, cap_nm: int) -> None:
    row = {"day": day, "variant": variant, "cap_nm": cap_nm, "train_seconds": train_seconds,
           "model_loaded": loaded, "train_timesteps": timesteps}
    row.update(metrics)
    rows.append(row)


def train_or_load_from_saved(path: Path, saved_path: Path, env, timesteps: int) -> tuple[MaskablePPO, float, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return MaskablePPO.load(str(path), env=env), 0.0, True
    model = MaskablePPO.load(str(saved_path), env=env)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    elapsed = time.perf_counter() - start
    model.save(str(path))
    return model, elapsed, False


def run(days: List[str], caps: List[int], timesteps: int, fine_tune: bool, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    setup = load_setup(verbose=True)
    rows: List[Dict[str, object]] = []

    cap_masks = {cap: capped_valid_bool(setup, cap) for cap in caps}
    for day in days:
        print(f"\n=== {day} ===")
        saved_path = ORIGINAL_RESULTS_DIR / f"dcb_ppo_fb_2_{day}.zip"
        for cap in caps:
            eval_env = make_fb_env(setup, day, valid_bool=cap_masks[cap])
            model = MaskablePPO.load(str(saved_path), env=eval_env)
            add_result(rows, day, f"FB saved with {cap}-NM cap", simulate_model(model, eval_env),
                       0.0, True, 100000, cap)
            print(f"Saved FB with {cap}-NM cap evaluated.")

            if fine_tune:
                train_env = make_fb_env(setup, day, valid_bool=cap_masks[cap])
                path = MODEL_DIR / f"fb_cap{cap}_finetune_{timesteps}_{day}.zip"
                tuned, elapsed, loaded = train_or_load_from_saved(path, saved_path, train_env, timesteps)
                tuned_env = make_fb_env(setup, day, valid_bool=cap_masks[cap])
                tuned.set_env(tuned_env)
                add_result(rows, day, f"FB fine-tuned with {cap}-NM cap", simulate_model(tuned, tuned_env),
                           elapsed, loaded, timesteps, cap)
                print(f"Fine-tuned FB with {cap}-NM cap evaluated ({elapsed:.1f}s, loaded={loaded}).")

        partial = pd.DataFrame(rows)
        partial.to_csv(OUT_DIR / "reviewer2_cap_sensitivity_results_partial.csv", index=False)
        summarize(partial).to_csv(OUT_DIR / "reviewer2_cap_sensitivity_summary_partial.csv", index=False)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "reviewer2_cap_sensitivity_results.csv", index=False)
    summary.to_csv(OUT_DIR / "reviewer2_cap_sensitivity_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate capped FB policy variants.")
    parser.add_argument("--days", default=",".join(SEVERE_DAYS_9))
    parser.add_argument("--caps", default="15,20,25")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run([d.strip() for d in args.days.split(",") if d.strip()],
        [int(c.strip()) for c in args.caps.split(",") if c.strip()],
        args.timesteps, args.fine_tune, args.seed)
