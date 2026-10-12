# Archived results

The `ablation/` directory preserves outputs from the reviewer-response experiments and sensitivity analyses associated with the final Transport Policy study package.

The files include:

- curriculum and direct-separation ablation results and summaries;
- a capacity-management/5-NM baseline comparison;
- budget-matched experiments;
- separation-cap sensitivity results;
- a full 209-day separation-cap evaluation;
- severe-day lower-bound results;
- separation-distribution summaries and by-day outputs;
- retained run logs for selected analyses.

The corresponding experiment drivers are under `../python/`, including `run_reviewer2_ablation.py`, `run_reviewer2_budget_matched.py`, `run_reviewer2_cap_sensitivity.py`, `run_reviewer2_full209_cap.py`, and `run_reviewer2_sep_distribution.py`.

These CSV/log files are archived study outputs. They are not regenerated in CI because doing so requires trained-policy evaluation and, for some experiments, additional training. CI instead verifies their presence and pins the principal summary tables by exact byte identity.

Because the RL training process was stochastic, rerunning training is not expected to recreate these archived results exactly. The retained policy archives under `../models/` support evaluation of the saved policies without making an unsupported deterministic-retraining claim.

See [`../PROVENANCE.md`](../PROVENANCE.md) for the complete reproducibility boundary.
