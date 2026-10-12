# Data

This directory contains the processed and aggregated research artifacts used by the tactical ATFM study at São Paulo/Guarulhos International Airport (SBGR). The public package is intended to support inspection and experiment rerunning without redistributing the underlying restricted or third-party source traffic records.

## Included artifacts

| Path | Role |
|---|---|
| `data.csv` | Archived 15-minute SBGR operational input used by the Python experiments |
| `new_taf_list.csv` | Decoded probabilistic TAF scenarios |
| `std_taf.csv` | Deterministic/standard TAF scenario |
| `atfm/CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv` | Processed CGNA ATFM-measure records used for Miles-in-Trail characterization |
| `movimentos_sbgr.csv` | SBGR movement totals used by the study plotting/characterization workflow |
| `all_proc.rda` | Processed STAR-procedure artifact used by the R/Python analysis workflow |

The RCCE files used by the tactical decision model are stored separately under `../curves/`.

## Shared archived SBGR input

`data.csv` is byte-for-byte the same archived processed SBGR input distributed with the preceding 2025 runway-capacity study repository. Its Git blob ID is:

`5660ea374ddd186dff838907a8b76ec20c0f0472`

The tactical ATFM paper extends the decision problem with arrival-separation control and additional training/evaluation materials; it does not require altering this shared archived operational input.

The committed file contains **66,816 15-minute periods over 696 operational days**, with 96 periods per operational day. CI verifies those structural properties.

## Historical preprocessing note

The archived `data.csv` contains operational days 24–28 April 2023. The currently committed historical preprocessing script `../R/generate_sbgr_data.R`, however, explicitly excludes those five days. Repository history does not contain a run manifest that establishes when that exclusion was introduced relative to the archived processed input.

For that reason, the archived `data.csv` should be treated as the authoritative processed input for rerunning the Python study workflow. The repository does **not** claim that the current R source-data pipeline recreates this file byte-for-byte. See [`../PROVENANCE.md`](../PROVENANCE.md).

## Source records not redistributed

`R/generate_sbgr_data.R` references source traffic records including ANAC VRA/SIROS material, Flightradar24-based trajectory extracts, and DECEA/TATIC operational records. Those source records are intentionally absent from this repository and are defensively ignored under `data/raw-data/`.

The included processed artifacts do not confer redistribution rights for the excluded underlying source records. Source-provider terms remain applicable. No separate software or data license is asserted by this repository.

## Integrity

The CI test suite pins the exact bytes of the principal processed artifacts, checks the 696-day/66,816-period structure of `data.csv`, validates the processed CGNA schema and SBGR movement years, confirms the April provenance distinction, and verifies that the excluded raw-data directory is absent.
