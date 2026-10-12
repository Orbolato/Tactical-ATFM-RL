# Provenance and reproducibility notes

This repository is the public research package associated with:

> Lucas Orbolato Carvalho and Mayara Condé Rocha Murça (2026), “An optimized model-free reinforcement learning approach for tactical air traffic flow and capacity management under uncertainty,” *Transport Policy*, 188, 104380. DOI: 10.1016/j.tranpol.2026.104380.

The initial prepared public package is preserved in repository history at commit `3ab789bd3943829580014e261557974a408a9a14`. The hardening pass adds documentation, integrity tests, and continuous integration without replacing the archived study data, trained policies, RCCE files, or ablation outputs.

## Public package boundary

The repository includes processed or aggregated research artifacts needed to inspect and rerun the study implementation:

- the archived 15-minute SBGR operational dataset (`data/data.csv`);
- decoded TAF scenario files (`data/new_taf_list.csv`, `data/std_taf.csv`);
- the processed CGNA ATFM-measure file used for the miles-in-trail characterization;
- SBGR movement totals and STAR-procedure material used for figures;
- RCCE breakpoints under `curves/`;
- retained Maskable PPO policies under `models/`;
- archived ablation and sensitivity outputs under `results/ablation/`.

The underlying source traffic records used by the historical R preprocessing pipeline are **not** redistributed. In particular, the VRA/SIROS source records, Flightradar24-based trajectory extracts, and DECEA/TATIC source records referenced by `R/generate_sbgr_data.R` are absent. Defensive `.gitignore` rules and CI tests preserve this boundary.

The inclusion of processed artifacts here does not imply a separate license for the underlying third-party source records. Source-provider terms remain applicable. No software or data license is asserted by this repository.

## Relationship to the 2025 runway-capacity package

`data/data.csv` is byte-for-byte the same archived processed SBGR input used in the preceding 2025 Transportation Research Part A runway-capacity repository (Git blob `5660ea374ddd186dff838907a8b76ec20c0f0472`). The tactical ATFM study extends the decision problem and training/evaluation workflow with arrival separation control, curriculum training, and the additional ATFM/ablation materials in this repository; it does not require silently rewriting that shared archived input.

## Reproducibility levels

The repository distinguishes three levels of reproducibility:

1. **Artifact inspection and verification.** CI checks immutable study anchors, dataset structure, policy archives, ablation outputs, citation metadata, and the absence of excluded raw source records.
2. **Policy re-evaluation and experiment rerunning.** The saved Maskable PPO policies and experiment code permit the study workflow to be re-evaluated or rerun from the committed processed inputs. The original training was stochastic, so retraining is not expected to reproduce saved policies or reported numerical results exactly.
3. **Raw-to-processed reconstruction.** The historical R preprocessing code is provided for methodological transparency, but the excluded source records are required for a full reconstruction and the current script is not asserted to recreate the archived processed dataset byte-for-byte.

CI intentionally does not retrain the full curriculum or evaluate all 209 forecast-based policies. Those computations are substantially more expensive than a repository-integrity check and would not establish deterministic retraining because the original training was stochastic.

## April 2023 preprocessing provenance

The archived `data/data.csv` is the same immutable processed input used by the predecessor runway-capacity package. It contains operational days 24–28 April 2023, while the currently committed `R/generate_sbgr_data.R` explicitly excludes those five days. Repository history does not provide an immutable raw-data run manifest establishing when that exclusion was introduced relative to the archived processed input.

Accordingly, the repository preserves both facts and makes no unsupported claim that the current R script exactly reconstructs `data/data.csv`. CI verifies the archived data structure and the presence of the exclusion statements so this distinction cannot be silently erased.

## Principal integrity anchors

The following Git blob IDs identify exact bytes in the initial prepared package:

| Artifact | Git blob ID |
|---|---|
| `data/data.csv` | `5660ea374ddd186dff838907a8b76ec20c0f0472` |
| `data/new_taf_list.csv` | `61b4ead4316413a0d939700632657b6a7d02c8cd` |
| `data/std_taf.csv` | `f747e78347f78ab230dd593b58f4812e42a60fff` |
| `data/atfm/CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv` | `77859f14d90854e2f9114cecc694b1afeba39fa4` |
| `data/movimentos_sbgr.csv` | `8ea036a731b80abffe6dbc96f789c2a9cf10e544` |
| `data/all_proc.rda` | `66b51e36ba46af3232a535665628e240a0889b1d` |
| `results/ablation/reviewer2_ablation_summary.csv` | `f7e226fcebc65ce1a2dd7f9ae30d44140708719c` |
| `results/ablation/reviewer2_budget_matched_summary.csv` | `37796cd2730ff47657b2f63c3ed25c1dc473bc57` |
| `results/ablation/reviewer2_cap_sensitivity_summary.csv` | `075b65a92d3739869675a8ed00c0930b2374750f` |
| `results/ablation/reviewer2_sep_distribution_summary.csv` | `8d4285199c58410c931963c0428ef8a163beae11` |
| `models/Results_eval/dcb_ppo_db_1.zip` | `027127ee93c49e4f4e98e6046bdd4c6ad061ee43` |
| `models/Results_eval/dcb_ppo_db_1_lb.zip` | `4f5b2bae76091ea5283e7bdd45dec533d4092010` |
| `models/Results_eval/dcb_ppo_db_1_ub.zip` | `9f3b4fda3a7eb948e6a24ed496c0a1ba10ac08f2` |
| `models/Results_eval/dcb_ppo_fb_2_2022-4-23.zip` | `65ef17b3836d028e16c5aeefc63849413d39453f` |

The integrity test also pins the six RCCE files used by the main Python workflow: VMC p=0.99, IMC p=0.95, and LIMC p=0.90 for both runway configurations.

## Historical implementation notes

The repository retains the study implementation rather than presenting a clean-room rewrite. Repository-relative paths replace local absolute paths, and environment-module filenames were normalized after the experiments as documented in the README. These maintenance changes should not be interpreted as evidence that every historical run can be recreated bit-for-bit.
