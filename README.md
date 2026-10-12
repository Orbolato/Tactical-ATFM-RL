# Model-free RL for tactical air traffic flow and capacity management under uncertainty

Code, processed data and trained policies for:

> Carvalho, L.O., Murça, M.C.R., 2026. **An optimized model-free reinforcement learning approach for tactical air traffic flow and capacity management under uncertainty.** *Transport Policy* 188, 104380. https://doi.org/10.1016/j.tranpol.2026.104380

Project page: https://orbolato.github.io/publications/tactical-atfm-reinforcement-learning/

The study extends the runway capacity management (CM) framework of Carvalho and Murça (2025, *Transp. Res. Part A* 200, 104620) with arrival separation control (miles-in-trail, MIT). A Maskable PPO agent jointly selects the runway configuration, the arrival–departure balance on the runway configuration capacity envelope (RCCE), and an incremental MIT action (−5, 0 or +5 NM, from 5 to 30 NM) with an anti-reversal smoothing rule. The agent is trained with a curriculum that combines historical data (data-based, DB) and forecast-based stochastic scenarios (FB). The case study is São Paulo/Guarulhos International Airport (SBGR), 2022–2023.

## Repository layout

```
.
├── data/                    Processed SBGR data used by all Python scripts
│   ├── data.csv             15-min traffic, weather and MIT series (2022–2023)
│   ├── new_taf_list.csv     Decoded TAF scenarios (with probabilities)
│   ├── std_taf.csv          Standard (most likely) TAF scenario
│   ├── atfm/                CGNA ATFM measures (MIT characterization, Figs. 12–13)
│   ├── movimentos_sbgr.csv  Annual/monthly SBGR aircraft movements (Fig. 1)
│   └── all_proc.rda         STAR procedures (map of Fig. 3)
├── curves/                  RCCE breakpoints per weather condition, runway configuration and percentile
├── python/                  Environments, training, evaluation, figures, GA benchmark and ablations
├── R/                       Data preprocessing, TAF decoding, RCCE estimation and map
├── optimization-models/     AMPL quantile-regression models for the RCCEs (Eqs. 1–2)
├── models/                  Trained Maskable PPO policies (Stable-Baselines3 .zip)
│   ├── Results_eval/        Canonical policies behind the main results (LB, UB, DB, FB × 209 test days)
│   ├── Results_2/           Curriculum run for the severe test days, used by the ablation
│   └── ablation/            Ablation policies (no curriculum, direct separation, 10-NM fine-tuned)
└── results/ablation/        Outputs of the ablation and separation-occurrence analyses (Tables 4–5)
```

## Installation

Python 3.12 was used. The trained policies were saved with Stable-Baselines3 2.4.0, PyTorch 2.6.0 (CPU), Gymnasium 1.0.0 and NumPy 1.26.4.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The R preprocessing scripts additionally need R (4.4 was used) with `plyr`, `dplyr`, `ggplot2`, `nloptr`, `lubridate` and `RCurl`. RCCE estimation needs AMPL with Gurobi and the `rAMPL` package; set the AMPL installation folder in `R/capacity_management.R` (`new(Environment, "...")`). The map script needs `leaflet`, `mapview`, `magick`, `webshot` and `conflicted`.

## Running the code

The Python scripts were developed as cell-based (Spyder) scripts and read all inputs relative to the repository root. Run them from `python/`. Outputs (models, PDFs) are written to the working directory, in `Results_2/`, `Results_eval/` and `Figures/`.

```bash
cd python
python maskable_ppo_test_2.py        # curriculum training (Section 3.5)
python maskable_ppo_eval.py          # evaluation of the canonical policies (Section 4.5)
python plots.py                      # result figures (Figs. 8–11, 14–27)
python mit_characterization.py       # MIT characterization (Figs. 12–13)
```

`maskable_ppo_test_2.py` reuses a policy from `models/Results_2/` when it exists and trains it otherwise. DB LB is trained in 500,000-timestep blocks until the validation reward stops improving, DB UB and DB for 1,000,000 timesteps, and FB LB and FB for 100,000 timesteps each per test day. A full retraining over the 209 test days takes many hours on a CPU.

### Ablation and separation-policy analyses (Section 4.5, Tables 4–5)

```bash
cd python
python run_reviewer2_ablation.py --timesteps 50000          # no smoothing, no curriculum, direct separation, rule heuristic
python run_reviewer2_cap_sensitivity.py --caps 10 --fine-tune
python run_reviewer2_budget_matched.py                      # 100,000-timestep budget-matched control
python run_reviewer2_sep_distribution.py                    # separation occurrence distribution (Table 5)
python run_reviewer2_full209_cap.py                         # 10-NM execution cap over all 209 test days
```

These scripts load existing policies from `models/` and only train when a policy is missing. They write their outputs to `results/ablation/`.

## Code overview

### Environments (`python/`)

| File | Role |
|---|---|
| `improved_gym_capoptimizer.py` | Capacity-management (CM) environment of Carvalho and Murça (2025). |
| `improved_gym_tdcboptimizer.py` | DB demand-capacity balancing environment, using historical traffic and METAR data. Training reward of Eqs. 9 and 19; with `evaluate=True` it accumulates the delay accounting of Eq. 21. |
| `improved_gym_tdcboptimizer_specific.py` | FB environment for a specific test day: stochastic traffic and weather scenarios from schedules, TAFs, the error models (Eqs. 24–26) and the confusion and transition likelihoods (Eq. 27). Same reward as above. Also provides `visualize_results` (Fig. 25). |
| `improved_gym_tdcboptimizer_2.py`, `improved_gym_tdcboptimizer_specific_2.py` | Same state, action, masking and transition dynamics. Queue terms are accumulated in the cost units of Eq. 20 (multiplied by *P* = 45 €/min downstream), and the separation term counts the arrivals displaced by the applied MIT. Used by the `run_reviewer2_*.py` analyses, which reproduce the full-sample FB queue-related cost of Table 2 (2433.81 €). |

The CM/LB, UB, DB and FB models differ only in the action mask: LB fixes Δsep = 0 at 5 NM, UB always increases separation up to 30 NM, and DB/FB use all incremental actions.

### Scripts (`python/`)

| File | Purpose |
|---|---|
| `maskable_ppo_test_2.py` | Data preparation, error models, likelihood matrices, train/test split (209 test days), greedy baseline, and the curriculum DB LB → {DB UB, DB, FB LB → FB} (Fig. 6). |
| `maskable_ppo_eval.py` | Loads the canonical policies from `models/Results_eval/` and evaluates them deterministically on the test days. |
| `plots.py` | Result figures. Per-day rewards, costs and delays of every method (GA-CM, GA-DCB, LB, UB, DB, FB and the operational baseline) are stored as arrays at the top of the script. |
| `mit_characterization.py` | Processes the CGNA ATFM measures and plots Figs. 12 and 13. |
| `runway_optimization_ga.py` | Genetic algorithm benchmark (DEAP) under the same action constraints and reward. |
| `calculations.py` | Cell script that runs the GA over the test days. It uses variables created by `maskable_ppo_test_2.py` (`data`, `u`, `v`, `possible_actions`, `valid_bool`, `wind_states`, `test_days`, `greedy_list`), so run it in the same interactive session. |
| `run_reviewer2_*.py` | Self-contained re-implementation of the preprocessing plus the targeted ablation, cap sensitivity, budget-matched control and separation-occurrence analyses. |

### R preprocessing (`R/`)

| File | Purpose |
|---|---|
| `generate_sbgr_data.R` | Builds the 15-min SBGR series from the raw traffic data (VRA, SIROS, Flightradar24, TATIC) and METARs. |
| `capacity_management.R` | Estimates the RCCEs with the quantile regression of Ramanujam and Balakrishnan (2009) (`optimization-models/linear_quantreg.mod`), writes `curves/` and the final `data/data.csv`. |
| `taf_decoder.R`, `taf_decoder_period.R`, `forecast_data_decoder_mod.R` | Decode the TAFs into `new_taf_list.csv` and `std_taf.csv`. |
| `fixes.R` | Map of the 40/100-NM rings and STARs (Fig. 3). |

The raw inputs of these scripts (`data/raw-data/...`) are not redistributed (see *Data*). The processed outputs needed by the Python code are included.

## Data

| File | Content |
|---|---|
| `data/data.csv` | One row per 15-min time step from 2022-01-02 06:00 to 2023-12-30 05:45 (UTC). Scheduled, observed (`*_demand_*`), served (`base_*`) and actual arrivals at 40 and 100 NM, per STAR entry fix (EDMUS, VUNOX, ZARES, BUXUK) and in total; the same for departures; runway configuration; MIT separation per fix (`sep_*`); METAR visibility, ceiling, IFR/LIFR flags, wind, head/crosswind and wind state; operating `day` (06:00 to 05:45 GMT). |
| `data/new_taf_list.csv` | TAF-derived weather scenarios per 15-min step, with probability, wind, visibility, ceiling, weather condition and wind state. |
| `data/std_taf.csv` | Standard TAF scenario with the same columns. |
| `curves/<WC>_<RC>_RCCE_<p>.csv` | RCCE breakpoints (arrivals, departures) for weather condition `VMC`/`IMC`/`LIMC`, configuration `10R_10L`/`28L_28R` and percentile `p` (e.g. `9900` = 0.99). The paper uses 0.99 (VMC), 0.95 (IMC) and 0.90 (LIMC). |
| `data/atfm/CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv` | ATFM measures published by CGNA (DECEA), used for the MIT characterization. |
| `data/movimentos_sbgr.csv` | Annual and monthly aircraft movements at SBGR (Fig. 1). |
| `data/all_proc.rda` | Standard terminal arrival route (STAR) procedures for the map. |

Sources: ANAC (VRA and SIROS), Flightradar24, TATIC (Saipher/DECEA), REDEMET (DECEA), Iowa Environmental Mesonet, CGNA (DECEA) and AISWEB (DECEA). The raw data are subject to the terms of their providers.

## Trained policies

| Folder | Files | Used for |
|---|---|---|
| `models/Results_eval/` | `dcb_ppo_db_1_lb` (LB/CM), `dcb_ppo_db_1_ub` (UB), `dcb_ppo_db_1` (DB), `dcb_ppo_fb_2_<day>` (FB, 209 test days) | Table 2, Table 3, Figs. 15–27, `run_reviewer2_full209_cap.py` |
| `models/Results_2/` | DB LB, DB UB, DB, and FB LB/FB for the severe test days | "FB saved" policy in Tables 4–5 |
| `models/ablation/` | `no_curr_*`, `direct_sep_*` (50,000 and 100,000 timesteps), `fb_cap10_finetune_50000_*` | Tables 4–5 and the budget-matched control |

Load a policy with:

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("models/Results_eval/dcb_ppo_fb_2_2022-4-23")
```

## Reproducing figures and tables

| Paper item | Source |
|---|---|
| Fig. 1 | `data/movimentos_sbgr.csv` |
| Fig. 3 | `R/fixes.R` (needs the raw Flightradar24 tracks) |
| Figs. 8–11, 14 | Data-preparation sections of `maskable_ppo_test_2.py` / `plots.py` |
| Figs. 12–13 | `mit_characterization.py` |
| Figs. 15–24, 26–27, Tables 2–3 | `plots.py` (per-day results produced by `maskable_ppo_eval.py` and the GA) |
| Fig. 25 | `plots.py` (`visualize_results` for day 2022-4-23) |
| Table 4 | `run_reviewer2_ablation.py`, `run_reviewer2_cap_sensitivity.py`; operational-baseline row in `results/ablation/reviewer2_baseline_cm_5nm_*.csv` |
| Table 5 | `run_reviewer2_sep_distribution.py` |
| Budget-matched control (Section 4.5) | `run_reviewer2_budget_matched.py` |

`results/ablation/reviewer2_baseline_cm_5nm_*.csv` and `reviewer2_lb_severe_*.csv` were produced interactively and have no standalone script.

## Notes

- Only paths were changed relative to the working copies used in the study: absolute local paths are now repository-relative, and the environment modules imported as `improved_gym_dcboptimizer*` were renamed to `improved_gym_tdcboptimizer*` after the experiments (same files). `mit_characterization.py` is the MIT-processing section of the study's training script, copied unchanged into a standalone file.
- PPO uses the Stable-Baselines3 defaults with `gamma = 1`. Training is stochastic, so retrained policies will not match the saved ones exactly. Use the policies in `models/` to reproduce the reported numbers.

## Citation

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

## Contact

Lucas Orbolato Carvalho (orbolatoloc@fab.mil.br) and Mayara Condé Rocha Murça (mayara@ita.br), Aeronautics Institute of Technology (ITA), São José dos Campos, Brazil.
