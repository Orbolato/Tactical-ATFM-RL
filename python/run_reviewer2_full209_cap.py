"""Full-209-day 10-NM execution-cap robustness check for Reviewer 2.

Evaluates the saved day-specific FB policies across all 209 test days both
without a cap (reproducing the paper's full-sample FB numbers) and with a
10-NM execution-level cap (action mask), without any retraining. The cap is a
post-processing constraint applied at deterministic execution only.

The canonical paper FB models are the evaluated policies in models/Results_eval
(saved there by maskable_ppo_eval.py). This script
first prints an uncapped validation against the reported Table 2 values
(mean queue-related cost 2433.81 euros, mean total delay 559.70 minutes) so the
result can be trusted before the capped comparison is used.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from run_reviewer2_ablation import (
    OUT_DIR,
    ROOT,
    SEP_MIN,
    TEST_DAYS_209,
    load_setup,
    make_fb_env,
    simulate_model,
)
from run_reviewer2_cap_sensitivity import capped_valid_bool

MODELS_DIR = ROOT / "models" / "Results_eval"
CAP_NM = 10
LAMBDA_STAR = 0.676  # break-even delay valuation (euros/min) from Table 3
REPORTED_COST = 2433.81
REPORTED_DELAY = 559.70


def seps_le10_share(env) -> float:
    seps = np.array(env.results[env.day]["separations"], dtype=float)
    return 100.0 * float(np.mean(seps <= 10.0)) if len(seps) else 0.0


def run() -> None:
    random.seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup = load_setup(verbose=True)
    cap_mask = capped_valid_bool(setup, CAP_NM)

    rows: List[Dict[str, object]] = []
    missing = 0
    for i, day in enumerate(TEST_DAYS_209):
        model_path = MODELS_DIR / f"dcb_ppo_fb_2_{day}.zip"
        if not model_path.exists():
            missing += 1
            continue

        env_u = make_fb_env(setup, day)
        model_u = MaskablePPO.load(str(model_path), env=env_u)
        mu = simulate_model(model_u, env_u)
        le10_u = seps_le10_share(env_u)

        env_c = make_fb_env(setup, day, valid_bool=cap_mask)
        model_c = MaskablePPO.load(str(model_path), env=env_c)
        mc = simulate_model(model_c, env_c)

        rows.append({
            "day": day,
            "cost_uncapped": mu["operating_cost_eur"], "cost_capped": mc["operating_cost_eur"],
            "delay_uncapped": mu["total_delay_min"], "delay_capped": mc["total_delay_min"],
            "queue_cost_uncapped": mu["queue_cost"], "queue_cost_capped": mc["queue_cost"],
            "mean_sep_uncapped": mu["mean_sep_nm"], "mean_sep_capped": mc["mean_sep_nm"],
            "le10_uncapped": le10_u,
            "obj_uncapped": mu["objective"], "obj_capped": mc["objective"],
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(TEST_DAYS_209)} days", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "reviewer2_full209_cap_results.csv", index=False)

    n = len(df)
    cu, cc = df.cost_uncapped.mean(), df.cost_capped.mean()
    du, dc = df.delay_uncapped.mean(), df.delay_capped.mean()
    qu, qc = df.queue_cost_uncapped.mean(), df.queue_cost_capped.mean()
    su, sc = df.mean_sep_uncapped.mean(), df.mean_sep_capped.mean()
    le10 = df.le10_uncapped.mean()
    # Generalized cost G = euro cost + lambda * delay (euros), at break-even lambda.
    gu = cu + LAMBDA_STAR * du
    gc = cc + LAMBDA_STAR * dc

    print(f"\n=== Full-209 evaluation ({n} days, {missing} missing) ===")
    print(f"{'metric':32s} {'uncapped':>12s} {'10-NM capped':>12s}")
    print(f"{'Mean queue-related cost (eur)':32s} {cu:12.2f} {cc:12.2f}")
    print(f"{'Mean total delay (min)':32s} {du:12.2f} {dc:12.2f}")
    print(f"{'Mean raw queue cost':32s} {qu:12.2f} {qc:12.2f}")
    print(f"{'Mean separation (NM)':32s} {su:12.2f} {sc:12.2f}")
    print(f"{'Gen. cost at lambda*=0.676':32s} {gu:12.2f} {gc:12.2f}")
    print(f"\nUncapped share of steps <= 10 NM: {le10:.1f}%")
    print(f"\n--- Validation vs reported Table 2 (uncapped FB) ---")
    print(f"  cost:  computed {cu:.2f}  vs reported {REPORTED_COST}  (diff {cu - REPORTED_COST:+.2f})")
    print(f"  delay: computed {du:.2f}  vs reported {REPORTED_DELAY}  (diff {du - REPORTED_DELAY:+.2f})")
    print(f"\nCap effect: delay {du:.1f} -> {dc:.1f} min ({100*(dc-du)/du:+.1f}%); "
          f"queue-related cost {cu:.1f} -> {cc:.1f} eur ({100*(cc-cu)/cu:+.1f}%)")
    print(f"Wrote {OUT_DIR / 'reviewer2_full209_cap_results.csv'}")


if __name__ == "__main__":
    run()
