import csv
import hashlib
from pathlib import Path
import unittest
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Git blob IDs from the initial prepared public package. Computing a Git blob ID
# directly from the checked-out bytes makes accidental changes detectable without
# relying on Git metadata.
EXPECTED_BLOBS = {
    "data/data.csv": "5660ea374ddd186dff838907a8b76ec20c0f0472",
    "data/new_taf_list.csv": "61b4ead4316413a0d939700632657b6a7d02c8cd",
    "data/std_taf.csv": "f747e78347f78ab230dd593b58f4812e42a60fff",
    "data/all_proc.rda": "66b51e36ba46af3232a535665628e240a0889b1d",
    "data/atfm/CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv": "77859f14d90854e2f9114cecc694b1afeba39fa4",
    "data/movimentos_sbgr.csv": "8ea036a731b80abffe6dbc96f789c2a9cf10e544",
    "curves/VMC_10R_10L_RCCE_9900.csv": "86bf2aceb16f5a0be22d82a00b2cedcf9b334a6c",
    "curves/VMC_28L_28R_RCCE_9900.csv": "a92c23ca7a38debfd41b3ba2f0c3b643d5721466",
    "curves/IMC_10R_10L_RCCE_9500.csv": "1ed3217222502e98a3138a951043ca2127e66027",
    "curves/IMC_28L_28R_RCCE_9500.csv": "713280ef3400fcf338cc2579bd705adac123c425",
    "curves/LIMC_10R_10L_RCCE_9000.csv": "4019d6f642544f9a24ed240d440bebff71ec5c1e",
    "curves/LIMC_28L_28R_RCCE_9000.csv": "ec347fc3c44b7c76b211a7f551c7fee2947c8065",
    "models/Results_eval/dcb_ppo_db_1.zip": "027127ee93c49e4f4e98e6046bdd4c6ad061ee43",
    "models/Results_eval/dcb_ppo_db_1_lb.zip": "4f5b2bae76091ea5283e7bdd45dec533d4092010",
    "models/Results_eval/dcb_ppo_db_1_ub.zip": "9f3b4fda3a7eb948e6a24ed496c0a1ba10ac08f2",
    "models/Results_eval/dcb_ppo_fb_2_2022-4-23.zip": "65ef17b3836d028e16c5aeefc63849413d39453f",
    "results/ablation/reviewer2_ablation_summary.csv": "f7e226fcebc65ce1a2dd7f9ae30d44140708719c",
    "results/ablation/reviewer2_budget_matched_summary.csv": "37796cd2730ff47657b2f63c3ed25c1dc473bc57",
    "results/ablation/reviewer2_cap_sensitivity_summary.csv": "075b65a92d3739869675a8ed00c0930b2374750f",
    "results/ablation/reviewer2_sep_distribution_summary.csv": "8d4285199c58410c931963c0428ef8a163beae11",
}


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def normalize_day(value: str):
    y, m, d = (int(x) for x in value.strip().split("-"))
    return y, m, d


def load_processed_day_counts():
    rows = 0
    day_counts = {}
    with (ROOT / "data" / "data.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "day" not in (reader.fieldnames or []):
            raise AssertionError("data/data.csv has no 'day' column")
        for row in reader:
            rows += 1
            day = normalize_day(row["day"])
            day_counts[day] = day_counts.get(day, 0) + 1
    return rows, day_counts


class RepositoryIntegrityTests(unittest.TestCase):
    def test_immutable_artifact_blob_ids(self):
        for rel, expected in EXPECTED_BLOBS.items():
            path = ROOT / rel
            self.assertTrue(path.is_file(), rel)
            self.assertEqual(git_blob_sha(path), expected, rel)

    def test_processed_dataset_shape(self):
        rows, day_counts = load_processed_day_counts()
        self.assertEqual(rows, 66816)
        self.assertEqual(len(day_counts), 696)
        self.assertEqual(set(day_counts.values()), {96})

    def test_archived_dataset_contains_april_24_to_28(self):
        _, day_counts = load_processed_day_counts()
        for d in range(24, 29):
            self.assertIn((2023, 4, d), day_counts)

    def test_current_preprocessing_script_preserves_april_exclusion(self):
        text = (ROOT / "R" / "generate_sbgr_data.R").read_text(encoding="utf-8")
        for d in range(24, 29):
            self.assertIn(f'2023-04-{d:02d}', text)

    def test_cgna_processed_atfm_schema(self):
        path = ROOT / "data" / "atfm" / "CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv"
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            fields = set(reader.fieldnames or [])
            self.assertTrue({"year", "regulated_element", "TMI", "MIT", "rate"} <= fields)
            first = next(reader)
            self.assertTrue(first["year"])

    def test_sbgr_movements_cover_2022_to_2024(self):
        with (ROOT / "data" / "movimentos_sbgr.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        years = {row["Ano"] for row in rows}
        self.assertEqual(years, {"2022", "2023", "2024"})

    def test_canonical_policy_inventory(self):
        folder = ROOT / "models" / "Results_eval"
        forecast_models = sorted(folder.glob("dcb_ppo_fb_2_*.zip"))
        self.assertEqual(len(forecast_models), 209)
        for name in ("dcb_ppo_db_1.zip", "dcb_ppo_db_1_lb.zip", "dcb_ppo_db_1_ub.zip"):
            self.assertTrue((folder / name).is_file(), name)
        self.assertTrue((folder / "dcb_ppo_fb_2_2023-4-28.zip").is_file())

    def test_ablation_policy_inventory(self):
        models = sorted((ROOT / "models" / "ablation").glob("*.zip"))
        self.assertEqual(len(models), 45)

    def test_policy_archives_are_valid_zip_files(self):
        # Validate the three DB anchors, one canonical FB policy, and one archive
        # from each ablation family. Inventory tests cover the full collections.
        paths = [
            ROOT / "models" / "Results_eval" / "dcb_ppo_db_1.zip",
            ROOT / "models" / "Results_eval" / "dcb_ppo_db_1_lb.zip",
            ROOT / "models" / "Results_eval" / "dcb_ppo_db_1_ub.zip",
            ROOT / "models" / "Results_eval" / "dcb_ppo_fb_2_2022-4-23.zip",
        ]
        ablation = ROOT / "models" / "ablation"
        for prefix in (
            "direct_sep_50000_",
            "direct_sep_100000_",
            "no_curr_50000_",
            "no_curr_100000_",
            "fb_cap10_finetune_50000_",
        ):
            matches = sorted(ablation.glob(prefix + "*.zip"))
            self.assertTrue(matches, prefix)
            paths.append(matches[0])
        for path in paths:
            self.assertTrue(zipfile.is_zipfile(path), str(path))
            with zipfile.ZipFile(path) as zf:
                self.assertTrue(zf.namelist(), str(path))

    def test_archived_ablation_outputs_present(self):
        folder = ROOT / "results" / "ablation"
        expected = {
            "reviewer2_ablation_results.csv",
            "reviewer2_ablation_summary.csv",
            "reviewer2_baseline_cm_5nm_results.csv",
            "reviewer2_baseline_cm_5nm_summary.csv",
            "reviewer2_budget_matched_results.csv",
            "reviewer2_budget_matched_summary.csv",
            "reviewer2_cap_sensitivity_results.csv",
            "reviewer2_cap_sensitivity_summary.csv",
            "reviewer2_full209_cap_results.csv",
            "reviewer2_lb_severe_results.csv",
            "reviewer2_lb_severe_summary.csv",
            "reviewer2_sep_distribution_byday.csv",
            "reviewer2_sep_distribution_summary.csv",
        }
        self.assertTrue(expected <= {p.name for p in folder.glob("*.csv")})
        for name in expected:
            self.assertGreater((folder / name).stat().st_size, 0, name)

    def test_restricted_raw_source_records_are_absent(self):
        self.assertFalse((ROOT / "data" / "raw-data").exists())
        self.assertFalse((ROOT / "data" / "taf_workspace.RData").exists())

    def test_citation_metadata(self):
        with (ROOT / "CITATION.cff").open(encoding="utf-8") as f:
            cff = yaml.safe_load(f)
        self.assertEqual(cff["preferred-citation"]["doi"], "10.1016/j.tranpol.2026.104380")
        self.assertEqual(cff["repository-code"], "https://github.com/Orbolato/Tactical-ATFM-RL")
        self.assertEqual(cff["authors"][0]["orcid"], "https://orcid.org/0000-0002-8402-5832")


if __name__ == "__main__":
    unittest.main()
