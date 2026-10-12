# Model-free RL for tactical air traffic flow and capacity management under uncertainty

[![reproducibility](https://github.com/Orbolato/Tactical-ATFM-RL/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/Orbolato/Tactical-ATFM-RL/actions/workflows/reproducibility.yml)

Code, processed data, retained Maskable PPO policies, and archived reviewer-response results for:

> Carvalho, L.O., Murça, M.C.R., 2026. **An optimized model-free reinforcement learning approach for tactical air traffic flow and capacity management under uncertainty.** *Transport Policy* 188, 104380. https://doi.org/10.1016/j.tranpol.2026.104380

Project page: https://orbolato.github.io/publications/tactical-atfm-reinforcement-learning/

The study extends the runway-capacity management framework of Carvalho and Murça (2025, *Transportation Research Part A* 200, 104620) with tactical arrival-separation control. A Maskable PPO agent jointly selects the runway configuration, the arrival–departure balance on the runway configuration capacity envelope (RCCE), and an incremental Miles-in-Trail (MIT) action (−5, 0, or +5 NM, bounded from 5 to 30 NM) with an anti-reversal smoothing rule. Training uses a curriculum combining historical data (data-based, DB) and forecast-based stochastic scenarios (FB). The case study is São Paulo/Guarulhos International Airport (SBGR), 2022–2023.

## Reproducibility scope

This repository distinguishes three levels of reproducibility:

1. **Artifact inspection and verification.** CI checks the exact bytes of principal data, RCCE, model, and result artifacts; validates the processed-data structure and model inventories; compiles the Python sources; validates citation metadata; and confirms that excluded raw source records are absent.
2. **Policy re-evaluation and experiment rerunning.** The retained Maskable PPO archives and Python implementation support loading saved policies and rerunning/evaluating the study workflow from the committed processed inputs. The original RL training was stochastic, so retraining is not expected to recreate the saved model bytes or published numerical results exactly.
3. **Raw-to-processed reconstruction.** The historical R preprocessing code is included for methodological transparency, but the underlying source traffic records are not redistributed. In addition, the current `R/generate_sbgr_data.R` is not asserted to recreate the committed `data/data.csv` byte-for-byte; see the April 2023 provenance note in [`PROVENANCE.md`](PROVENANCE.md).

CI deliberately does **not** retrain the curriculum or execute the complete 209-policy evaluation. Those are computational experiments rather than lightweight repository-integrity checks.

## Repository layout

```text
.
├── .github/workflows/        Lightweight reproducibility/integrity CI
├── data/                     Public processed/aggregated study artifacts
│   ├── data.csv              Archived 15-min SBGR operational input (2022–2023)
│   ├── new_taf_list.csv      Decoded probabilistic TAF scenarios
│   ├── std_taf.csv           Standard/deterministic TAF scenario
│   ├── atfm/                 Processed CGNA ATFM measures for MIT characterization
│   ├── movimentos_sbgr.csv   Annual/monthly SBGR aircraft movements
│   └── all_proc.rda          Processed STAR-procedure artifact used by the map workflow
├── curves/                   RCCE breakpoints by weather, configuration, and percentile
├── python/                   Environments, training, evaluation, figures, GA, and ablations
├── R/                        Historical preprocessing, TAF decoding, RCCE estimation, and map
├── optimization-models/      AMPL quantile-regression models for the RCCEs
├── models/                   Retained Maskable PPO policy archives
│   ├── Results_eval/         Canonical evaluation policies (DB/LB/UB + FB × 209 test days)
│   ├── Results_2/            Severe-day/curriculum policy collection
│   └── ablation/             Reviewer-response ablation/sensitivity policies
├── results/ablation/         Archived reviewer-response outputs
├── tests/                    Public-package integrity tests
├── CITATION.cff              Machine-readable citation metadata
└── PROVENANCE.md             Data/model/result provenance and reproducibility boundary
```

## Installation

Python 3.12 was used for the prepared experiment environment. The retained policies were saved with Stable-Baselines3 2.4.0, SB3-Contrib 2.4.0, PyTorch 2.6.0 (CPU), Gymnasium 1.0.0, and NumPy 1.26.4. The remaining packages in `requirements.txt` retain the compatibility bounds documented in the prepared study package rather than inventing exact versions that were not recorded.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The historical R preprocessing uses R 4.4 with `plyr`, `dplyr`, `ggplot2`, `nloptr`, `lubridate`, and `RCurl`. RCCE estimation also needs AMPL with Gurobi and `rAMPL`; set the AMPL installation folder in `R/capacity_management.R`. The map workflow additionally uses `leaflet`, `mapview`, `magick`, `webshot`, and `conflicted`.

For the lightweight repository checks only, install `requirements-ci.txt` instead of the full experiment environment.

## Running the study code

The Python scripts were developed as cell-based Spyder scripts and read inputs relative to the repository root. Run them from `python/`. Generated models, PDFs, CSVs, and figures written under the working directory are ignored by Git unless they are part of the archived public package.

```bash
cd python
python maskable_ppo_test_2.py        # curriculum training (Section 3.5)
python maskable_ppo_eval.py          # canonical-policy evaluation (Section 4.5)
python plots.py                      # result figures (Figs. 8–11, 14–27)
python mit_characterization.py       # MIT characterization (Figs. 12–13)
```

`maskable_ppo_test_2.py` reuses a policy from `models/Results_2/` when it exists and trains it otherwise. DB LB is trained in 500,000-timestep blocks until the validation reward stops improving, DB UB and DB for 1,000,000 timesteps, and FB LB and FB for 100,000 timesteps each per test day. A complete retraining/evaluation over 209 test days is therefore substantially more expensive than the CI verification path.

### Reviewer-response ablation and separation-policy analyses

```bash
cd python
python run_reviewer2_ablation.py --timesteps 50000
python run_reviewer2_cap_sensitivity.py --caps 10 --fine-tune
python run_reviewer2_budget_matched.py
python run_reviewer2_sep_distribution.py
python run_reviewer2_full209_cap.py
```

These scripts load retained policies where available and train only when a required policy is missing. Their archived outputs are under `results/ablation/`; see [`results/README.md`](results/README.md).

## Code overview

### Environments (`python/`)

| File | Role |
|---|---|
| `improved_gym_capoptimizer.py` | Capacity-management environment inherited from the 2025 runway-capacity study. |
| `improved_gym_tdcboptimizer.py` | DB tactical demand-capacity-balancing environment using historical traffic and METAR data. |
| `improved_gym_tdcboptimizer_specific.py` | FB environment for a specific test day, including stochastic traffic/weather scenarios from schedules, TAFs, error models, and weather likelihoods. |
| `improved_gym_tdcboptimizer_2.py`, `improved_gym_tdcboptimizer_specific_2.py` | Reviewer-response variants using the same state/action/masking dynamics with the alternative cost accounting used by the targeted analyses. |

The CM/LB, UB, DB, and FB policies differ in their action constraints. The tactical environments represent separation as a decision state/action bounded from 5 to 30 NM; the archived operational input itself is not a record of the learned MIT decisions.

### Scripts (`python/`)

| File | Purpose |
|---|---|
| `maskable_ppo_test_2.py` | Data preparation, forecast/error models, train/test split, greedy baseline, and curriculum DB LB → {DB UB, DB, FB LB → FB}. |
| `maskable_ppo_eval.py` | Loads the canonical policies from `models/Results_eval/` and evaluates them on the test days. |
| `plots.py` | Study result figures; archived per-day arrays used by the plotting workflow are retained in the script. |
| `mit_characterization.py` | Processes the public CGNA ATFM-measure artifact and plots the MIT characterization. |
| `runway_optimization_ga.py` | DEAP genetic-algorithm benchmark under the study action constraints and reward. |
| `calculations.py` | Interactive/cell-based GA driver that uses variables created by `maskable_ppo_test_2.py`. |
| `run_reviewer2_*.py` | Targeted reviewer-response ablation, cap-sensitivity, budget-matched, full-sample, and separation-occurrence analyses. |

### Historical R preprocessing (`R/`)

| File | Purpose |
|---|---|
| `generate_sbgr_data.R` | Historical construction of the 15-min SBGR series from source traffic and METAR records. |
| `capacity_management.R` | RCCE quantile-regression estimation and export of processed capacity inputs. |
| `taf_decoder.R`, `taf_decoder_period.R`, `forecast_data_decoder_mod.R` | TAF decoding into the processed forecast files. |
| `fixes.R` | Map workflow for the terminal-area rings and STAR procedures. |

The source traffic records required by `generate_sbgr_data.R` are intentionally not redistributed. The committed processed files are the public study artifacts used by the Python workflow. See [`data/README.md`](data/README.md).

## Data and provenance

The principal archived input, `data/data.csv`, is byte-for-byte identical to the processed SBGR input distributed with the preceding 2025 runway-capacity repository (Git blob `5660ea374ddd186dff838907a8b76ec20c0f0472`). It contains 66,816 15-minute periods across 696 operational days. The tactical study extends the decision process and training/evaluation workflow without silently rewriting that shared archived input.

The file includes traffic-demand/service variables, runway configuration, and meteorological features. Historical `sep_*` fields in this shared input are constant placeholders and are not the learned tactical MIT decisions; tactical separation is represented by the environment/action logic in the Python code.

The current `R/generate_sbgr_data.R` excludes operational days 24–28 April 2023, whereas the archived `data/data.csv` contains those days. Both are preserved, and no unsupported claim is made that the current raw-to-processed R pipeline exactly recreates the archived study input. [`PROVENANCE.md`](PROVENANCE.md) documents this distinction and the immutable artifact anchors.

Included processed/aggregated artifacts are intentionally public research materials for this repository. Underlying VRA/SIROS, Flightradar24-based trajectory, and DECEA/TATIC source records are not redistributed; source-provider terms remain applicable.

## Retained policies

`models/Results_eval/` contains the principal DB policy and lower/upper-bound variants together with **209** forecast-based policy archives, one for each evaluation day. `models/Results_2/` preserves the severe-day/curriculum collection, while `models/ablation/` contains **45** reviewer-response ablation/sensitivity policies. See [`models/README.md`](models/README.md).

A retained policy can be loaded without retraining, for example:

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("models/Results_eval/dcb_ppo_fb_2_2022-4-23.zip")
```

Loading a retained policy supports re-evaluation of that saved policy. This repository does **not** claim that retraining from scratch will recreate the same policy or numerical outputs exactly.

## Paper items and associated sources

| Paper item | Associated source/artifact |
|---|---|
| Fig. 1 | `data/movimentos_sbgr.csv` |
| Fig. 3 | `R/fixes.R`, `data/all_proc.rda`; full historical map reconstruction may require excluded source tracks |
| Figs. 8–11, 14 | Data-preparation sections of `maskable_ppo_test_2.py` / `plots.py` |
| Figs. 12–13 | `mit_characterization.py`, processed CGNA ATFM measures |
| Figs. 15–24, 26–27; Tables 2–3 | `plots.py`, canonical-policy evaluation/GA workflow |
| Fig. 25 | `plots.py` / tactical-environment visualization for the selected example day |
| Table 4 | Reviewer-response ablation/cap-sensitivity scripts and archived CSV outputs |
| Table 5 | `run_reviewer2_sep_distribution.py` and archived separation-distribution outputs |
| Budget-matched control | `run_reviewer2_budget_matched.py` and archived outputs |

`results/ablation/reviewer2_baseline_cm_5nm_*.csv` and `reviewer2_lb_severe_*.csv` are retained historical outputs without a standalone generation script. They are therefore documented as archived artifacts rather than advertised as automatically regenerated by CI.

## Quick repository verification

This path is deliberately lightweight and does not retrain the agents:

```bash
pip install -r requirements-ci.txt
python -m compileall -q python tests
python -m unittest discover -s tests -p 'test_*.py' -v
```

The same checks run automatically in GitHub Actions. They verify principal artifact byte identity, the processed-data shape, the April preprocessing provenance distinction, CGNA/SBGR supporting-data schemas, canonical and ablation policy inventories, representative policy ZIP integrity, archived reviewer-response outputs, the public/raw-data boundary, and `CITATION.cff` metadata.

## Historical implementation notes

Only maintenance-oriented changes were made relative to the prepared working copies: local absolute paths were made repository-relative, and environment module filenames imported historically as `improved_gym_dcboptimizer*` were normalized to `improved_gym_tdcboptimizer*` after the experiments. `mit_characterization.py` is the MIT-processing section of the study workflow separated into a standalone script. The repository preserves this historical implementation rather than presenting a clean-room rewrite.

PPO uses the Stable-Baselines3 defaults with `gamma = 1`. Training is stochastic; use the retained policies for policy re-evaluation rather than interpreting a new training run as an exact numerical reproduction.

## Citation

GitHub also exposes [`CITATION.cff`](CITATION.cff). Please cite the published article:

```bibtex
@article{Carvalho2026,
  author  = {Carvalho, Lucas Orbolato and Mur{\c{c}}a, Mayara Cond{\'e} Rocha},
  title   = {An optimized model-free reinforcement learning approach for tactical air traffic flow and capacity management under uncertainty},
  journal = {Transport Policy},
  volume  = {188},
  pages   = {104380},
  year    = {2026},
  doi     = {10.1016/j.tranpol.2026.104380}
}
```

## License

No software or data license is asserted by this repository. Please follow the publication citation and the provenance/access conditions documented for the included research materials and their underlying sources.

## Contact

Lucas Orbolato Carvalho (orbolatoloc@fab.mil.br) and Mayara Condé Rocha Murça (mayara@ita.br), Aeronautics Institute of Technology (ITA), São José dos Campos, Brazil.
