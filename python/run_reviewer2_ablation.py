"""Targeted reduced experiments for Reviewer 2 second-round revision.

This script is intentionally separate from the original monolithic training
scripts. It reuses the original environments and preprocessing logic, but runs a
compact set of ablations on selected severe SBGR days.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from itertools import product
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sklearn.linear_model import LinearRegression

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import improved_gym_tdcboptimizer_2 as tdcb
import improved_gym_tdcboptimizer_specific_2 as tdcb_sp

try:
    from torch.distributions import Distribution
    Distribution.set_default_validate_args(False)
except Exception:
    pass

ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
CURVES_DIR = ROOT / "curves"
ORIGINAL_RESULTS_DIR = ROOT / "models" / "Results_2"
OUT_DIR = ROOT / "results" / "ablation"
MODEL_DIR = ROOT / "models" / "ablation"

P_GROUND = 45.0
ALPHA = 2
BETA = 1
DELTA = 0.3
SEP_MIN = 5
SEP_MAX = 30
SEP_STEP = 5

SEVERE_DAYS_9 = [
    "2023-9-10", "2023-12-6", "2022-9-19", "2022-5-29", "2023-6-1",
    "2023-6-30", "2023-1-29", "2023-11-15", "2022-8-2",
]

TEST_DAYS_209 = [
 '2023-9-10','2023-12-6','2022-9-19','2022-5-29','2023-6-1','2023-6-30','2023-1-29','2023-11-15','2022-8-2','2023-11-1','2023-1-30','2022-8-17','2023-12-7','2023-1-5','2022-3-18','2023-11-25','2023-3-9','2022-10-14','2023-3-7','2022-9-2','2023-10-7','2023-2-6','2022-5-23','2023-3-23','2023-5-19','2022-2-17','2023-5-3','2023-11-4','2022-1-16','2022-7-29','2023-2-1','2023-3-21','2023-8-8','2023-5-24','2022-8-20','2023-8-19','2023-4-28','2023-8-10','2022-8-1','2023-10-28','2022-10-4','2023-2-15','2022-11-24','2022-11-4','2022-10-29','2022-10-30','2022-11-20','2023-8-11','2022-12-7','2023-9-23','2023-6-29','2023-2-7','2022-2-28','2023-6-21','2022-9-13','2022-10-19','2023-9-18','2023-1-1','2022-6-12','2022-1-8','2023-7-9','2023-1-3','2022-12-2','2023-10-18','2022-12-22','2022-12-9','2023-2-10','2023-4-30','2022-11-19','2022-8-6','2023-8-3','2022-12-30','2022-5-10','2023-12-2','2023-7-17','2023-10-27','2022-11-12','2023-8-24','2023-7-21','2022-11-26','2023-11-23','2023-12-5','2023-4-12','2022-7-14','2022-5-27','2023-6-11','2023-5-12','2022-11-10','2022-7-9','2022-5-4','2022-3-22','2023-5-9','2022-10-8','2023-9-1','2023-10-8','2023-11-18','2023-12-13','2023-3-25','2022-10-17','2023-12-11','2022-8-24','2022-6-17','2022-9-26','2022-12-6','2022-11-1','2022-11-3','2023-10-15','2023-9-4','2022-11-16','2023-10-30','2022-3-30','2022-1-6','2023-8-16','2023-3-12','2022-2-16','2023-6-14','2022-8-3','2022-4-17','2022-11-7','2022-10-1','2023-3-31','2022-11-25','2023-9-16','2023-9-15','2023-1-18','2023-12-21','2023-1-12','2023-10-22','2023-6-17','2022-1-4','2023-10-20','2022-10-28','2022-6-4','2022-3-25','2023-4-8','2022-11-22','2022-4-30','2022-3-24','2022-2-26','2022-12-28','2022-4-25','2022-7-15','2023-11-30','2022-3-12','2022-1-28','2023-2-13','2023-5-23','2023-2-9','2022-6-9','2022-5-26','2022-4-23','2023-8-14','2022-11-27','2023-8-6','2022-5-16','2023-12-22','2022-2-11','2022-4-29','2022-5-19','2022-9-17','2022-7-24','2023-7-6','2022-10-23','2022-4-21','2023-9-25','2022-3-11','2022-10-21','2023-11-21','2022-7-2','2023-12-17','2023-1-24','2023-8-5','2022-5-3','2022-12-29','2023-4-26','2022-3-15','2022-5-28','2023-2-28','2022-6-15','2022-6-10','2022-4-14','2023-1-14','2022-1-24','2022-9-30','2022-12-10','2022-1-13','2023-12-12','2022-9-24','2023-3-14','2022-6-13','2023-6-8','2022-4-4','2023-3-13','2022-7-7','2022-1-2','2022-9-6','2022-8-11','2023-2-5','2023-11-8','2023-8-20','2023-5-25','2022-4-1','2023-5-30','2023-5-28','2023-9-22','2023-5-8','2022-9-27','2022-6-29','2023-11-5'
]


@dataclass
class Setup:
    data: pd.DataFrame
    taf_data: pd.DataFrame
    std_taf_data: pd.DataFrame
    days: List[str]
    congested_days: List[str]
    training_days: List[str]
    u: Dict[Tuple[int, int], np.ndarray]
    v: Dict[Tuple[int, int], np.ndarray]
    u_max: int
    v_max: int
    possible_actions: np.ndarray
    valid_bool: np.ndarray
    valid_bool_lb: np.ndarray
    possible_actions_direct: np.ndarray
    valid_bool_direct: np.ndarray
    wind_states: Dict[int, List[int]]
    observation_low: List[int]
    observation_high: List[int]
    arr_mu_40: float
    arr_sigma_40: float
    arr_mu_100: float
    arr_sigma_100: float
    dep_mu: float
    dep_sigma: float
    arr_mean_40: np.ndarray
    arr_std_40: np.ndarray
    arr_mean_100: np.ndarray
    arr_std_100: np.ndarray
    dep_mean: np.ndarray
    dep_std: np.ndarray
    p_wc_trans: np.ndarray
    p_ws_trans: np.ndarray
    p_wc_error: np.ndarray
    p_ws_error: np.ndarray


def previous_day(day_string: str) -> str:
    year, month, day = [int(x) for x in day_string.split("-")]
    month_days = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                  31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if day == 1:
        if month == 1:
            return f"{year - 1}-12-31"
        return f"{year}-{month - 1}-{month_days[month - 2]}"
    return f"{year}-{month}-{day - 1}"


def f_curve(x: np.ndarray, y: np.ndarray, u_values: np.ndarray) -> np.ndarray:
    v_ub = []
    for u_i in u_values:
        appended = False
        for j in range(len(x) - 1):
            if x[j] <= u_i < x[j + 1]:
                dx = x[j + 1] - x[j]
                v_ub.append(y[j] if dx == 0 else (y[j + 1] - y[j]) / dx * (u_i - x[j]) + y[j])
                appended = True
                break
        if not appended:
            v_ub.append(y[-2] if u_i == x[-1] and x[-1] == x[-2] else 0)
    return np.array(v_ub, dtype=np.int64)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    out = matrix.astype(float).copy()
    for i in range(out.shape[0]):
        row_sum = out[i].sum()
        out[i] = out[i] / row_sum if row_sum > 0 else np.full(out.shape[1], 1 / out.shape[1])
    return out


def linear_error_model(data: pd.DataFrame, sched_col: str, error_col: str) -> Tuple[np.ndarray, np.ndarray]:
    means, stds, x = [], [], []
    fallback_std = max(float(np.std(data[error_col])), 1e-3)
    for value in np.sort(data[sched_col].unique()):
        errors = data.loc[data[sched_col] == value, error_col]
        means.append(float(np.mean(errors)))
        stds.append(float(np.std(errors)) or fallback_std)
        x.append([value])
    grid = np.array([[i] for i in range(100)])
    mean_model = LinearRegression().fit(x, np.array(means))
    std_model = LinearRegression().fit(x, np.array(stds))
    return mean_model.predict(grid), np.maximum(std_model.predict(grid), 1e-3)


def capacity_actions_for_state(setup_u: Dict[Tuple[int, int], np.ndarray],
                               setup_v: Dict[Tuple[int, int], np.ndarray],
                               wind_states: Dict[int, List[int]], ws: int, wc: int,
                               a_i: int, d_i: int, sep_vars: List[int]) -> List[List[int]]:
    actions: List[List[int]] = []
    for rc in wind_states[ws]:
        u_values = setup_u[wc, rc]
        v_values = setup_v[wc, rc]
        arr_lim = min(a_i, int(u_values[-1]))
        arr_min = min(int(u_values[-1]) - list(v_values[::-1]).index(v_values[0]), arr_lim)
        for arr in range(arr_min, arr_lim):
            if d_i >= v_values[arr]:
                for sep_var in sep_vars:
                    actions.append([rc, arr, int(v_values[arr]), sep_var])
        dep = min(d_i, int(v_values[arr_lim]))
        for sep_var in sep_vars:
            actions.append([rc, arr_lim, dep, sep_var])
    return actions


def build_valid_bool(possible_actions: np.ndarray, u: Dict[Tuple[int, int], np.ndarray],
                     v: Dict[Tuple[int, int], np.ndarray], wind_states: Dict[int, List[int]],
                     u_max: int, v_max: int, mode: str) -> np.ndarray:
    action_to_idx = {tuple(action): i for i, action in enumerate(possible_actions.tolist())}
    mask = np.zeros((len(wind_states), 3, u_max + 1, v_max + 1, SEP_MAX + 1, len(possible_actions)), dtype=bool)
    for ws in wind_states:
        for wc in range(3):
            for a_i in range(u_max + 1):
                for d_i in range(v_max + 1):
                    for sep in range(SEP_MIN, SEP_MAX + 1, SEP_STEP):
                        if mode == "lb":
                            sep_vars = [1]
                        elif sep == SEP_MIN:
                            sep_vars = [1, 2]
                        elif sep == SEP_MAX:
                            sep_vars = [0, 1]
                        else:
                            sep_vars = [0, 1, 2]
                        for action in capacity_actions_for_state(u, v, wind_states, ws, wc, a_i, d_i, sep_vars):
                            idx = action_to_idx.get(tuple(action))
                            if idx is not None:
                                mask[ws, wc, a_i, d_i, sep, idx] = True
    return mask


def build_direct_actions_and_mask(u: Dict[Tuple[int, int], np.ndarray], v: Dict[Tuple[int, int], np.ndarray],
                                  wind_states: Dict[int, List[int]], u_max: int, v_max: int) -> Tuple[np.ndarray, np.ndarray]:
    actions = []
    for rc in [0, 1, 2]:
        if rc == 0:
            for sep_abs in range(SEP_MIN, SEP_MAX + 1, SEP_STEP):
                actions.append([rc, 0, 0, sep_abs])
        else:
            for arr in u[0, rc]:
                for dep in range(int(v[0, rc][arr]) + 1):
                    for sep_abs in range(SEP_MIN, SEP_MAX + 1, SEP_STEP):
                        actions.append([rc, int(arr), int(dep), sep_abs])
    possible_actions = np.array(actions, dtype=np.int64)
    action_to_idx = {tuple(action): i for i, action in enumerate(possible_actions.tolist())}
    mask = np.zeros((len(wind_states), 3, u_max + 1, v_max + 1, SEP_MAX + 1, len(possible_actions)), dtype=bool)
    for ws in wind_states:
        for wc in range(3):
            for a_i in range(u_max + 1):
                for d_i in range(v_max + 1):
                    for sep in range(SEP_MIN, SEP_MAX + 1, SEP_STEP):
                        for sep_abs in range(SEP_MIN, SEP_MAX + 1, SEP_STEP):
                            for action in capacity_actions_for_state(u, v, wind_states, ws, wc, a_i, d_i, [1]):
                                direct = [action[0], action[1], action[2], sep_abs]
                                idx = action_to_idx.get(tuple(direct))
                                if idx is not None:
                                    mask[ws, wc, a_i, d_i, sep, idx] = True
    return possible_actions, mask


def load_setup(verbose: bool = True) -> Setup:
    if verbose:
        print("Loading data and rebuilding original preprocessing...")
    data = pd.read_csv(DATA_DIR / "data.csv")
    taf_data = pd.read_csv(DATA_DIR / "new_taf_list.csv")
    std_taf_data = pd.read_csv(DATA_DIR / "std_taf.csv")

    for i in range(len(data) - 1):
        data.loc[i, "base_arrivals_40"] = data.loc[i + 1, "actual_arrivals"]
        data.loc[i, "base_departures"] = data.loc[i + 1, "actual_departures"]
        data.loc[i, "base_configuration"] = data.loc[i + 1, "configuration"]

    while data.loc[0, "time"] != data.loc[0, "time"].replace(data.loc[0, "time"][-8:], "06:00:00"):
        data.drop(0, axis="rows", inplace=True)
        data.reset_index(drop=True, inplace=True)
    while data.loc[len(data) - 1, "time"] != data.loc[len(data) - 1, "time"].replace(data.loc[len(data) - 1, "time"][-8:], "05:45:00"):
        data.drop(len(data) - 1, axis="rows", inplace=True)
        data.reset_index(drop=True, inplace=True)

    days_col = []
    for time_string in data["time"]:
        parts = time_string.split(" ")
        y, m, d = [int(x) for x in parts[0].split("-")]
        day_string = f"{y}-{m}-{d}"
        hour = int(parts[1][0:2]) if len(parts) > 1 else 0
        days_col.append(previous_day(day_string) if hour < 6 else day_string)
    data["day"] = days_col

    data["arrival_error_40"] = (data["arrival_demand_40"] - data["scheduled_arrivals_40"]).astype(int)
    data["arrival_error_100"] = (data["arrival_demand_100"] - data["scheduled_arrivals_100"]).astype(int)
    data["departure_error"] = (data["departure_demand"] - data["scheduled_departures"]).astype(int)
    arr_mean_40, arr_std_40 = linear_error_model(data, "scheduled_arrivals_40", "arrival_error_40")
    arr_mean_100, arr_std_100 = linear_error_model(data, "scheduled_arrivals_100", "arrival_error_100")
    dep_mean, dep_std = linear_error_model(data, "scheduled_departures", "departure_error")
    arr_mu_40, arr_sigma_40 = float(np.mean(data["arrival_error_40"])), float(np.std(data["arrival_error_40"]))
    arr_mu_100, arr_sigma_100 = float(np.mean(data["arrival_error_100"])), float(np.std(data["arrival_error_100"]))
    dep_mu, dep_sigma = float(np.mean(data["departure_error"])), float(np.std(data["departure_error"]))

    while taf_data.loc[0, "time"] != data.loc[0, "time"]:
        taf_data.drop(0, axis="rows", inplace=True)
        taf_data.reset_index(drop=True, inplace=True)
        std_taf_data.drop(0, axis="rows", inplace=True)
        std_taf_data.reset_index(drop=True, inplace=True)
    while taf_data.loc[taf_data.shape[0] - 1, "time"] != data.loc[data.shape[0] - 1, "time"]:
        taf_data.drop(taf_data.shape[0] - 1, axis="rows", inplace=True)
        taf_data.reset_index(drop=True, inplace=True)
        std_taf_data.drop(std_taf_data.shape[0] - 1, axis="rows", inplace=True)
        std_taf_data.reset_index(drop=True, inplace=True)

    demand_pairs = [
        ("arrival_demand_40_edmus", "scheduled_arrivals_40_edmus"),
        ("arrival_demand_40_vunox", "scheduled_arrivals_40_vunox"),
        ("arrival_demand_40_zares", "scheduled_arrivals_40_zares"),
        ("arrival_demand_40_buxuk", "scheduled_arrivals_40_buxuk"),
        ("arrival_demand_40", "scheduled_arrivals_40"),
        ("arrival_demand_100_edmus", "scheduled_arrivals_100_edmus"),
        ("arrival_demand_100_vunox", "scheduled_arrivals_100_vunox"),
        ("arrival_demand_100_zares", "scheduled_arrivals_100_zares"),
        ("arrival_demand_100_buxuk", "scheduled_arrivals_100_buxuk"),
        ("arrival_demand_100", "scheduled_arrivals_100"),
        ("departure_demand", "scheduled_departures"),
    ]
    for target, source in demand_pairs:
        taf_data[target] = data[source].to_numpy()
        std_taf_data[target] = data[source].to_numpy()

    data["weather_condition"] = (data["ifr"] + data["lifr"]).astype(int)
    data["10R_10L"] = 0
    data["28L_28R"] = 0
    cfg_map = {"10R|10L": 1, "28L|28R": 2}
    for i in range(len(data)):
        data.loc[i, "configuration"] = cfg_map.get(data.loc[i, "configuration"], 0)
        data.loc[i, "base_configuration"] = cfg_map.get(data.loc[i, "base_configuration"], 0)
        data.loc[i, "10R_10L"] = 1 if data.loc[i, "configuration"] == 1 else 0
        data.loc[i, "28L_28R"] = 1 if data.loc[i, "configuration"] == 2 else 0
    data["configuration"] = data["configuration"].astype(int)
    data["base_configuration"] = data["base_configuration"].astype(int)

    data["wind_state"] = 0
    data["10R_10L_OK"] = 0
    data["28L_28R_OK"] = 0
    for i in range(len(data)):
        if abs(data.loc[i, "crosswind_10"]) > 20:
            ws, ok10, ok28 = 3, 0, 0
        elif data.loc[i, "headwind_10"] < -5:
            ws, ok10, ok28 = 2, 0, 1
        elif data.loc[i, "headwind_10"] > 5:
            ws, ok10, ok28 = 1, 1, 0
        else:
            ws, ok10, ok28 = 0, 1, 1
        data.loc[i, "wind_state"] = ws
        data.loc[i, "10R_10L_OK"] = ok10
        data.loc[i, "28L_28R_OK"] = ok28
    data["wind_state"] = data["wind_state"].astype(int)

    points: Dict[int, Dict[int, pd.DataFrame]] = {0: {}, 1: {}, 2: {}}
    for file in [f for f in listdir(CURVES_DIR) if isfile(join(CURVES_DIR, f))]:
        if file.startswith("VMC"):
            wc, p = 0, 0.99
        elif file.startswith("IMC"):
            wc, p = 1, 0.95
        elif file.startswith("LIMC"):
            wc, p = 2, 0.9
        else:
            continue
        suffix = "{:.4f}".format(p).replace("0.", "") + ".csv"
        if not file.endswith(suffix):
            continue
        name = file.replace("_RCCE_" + suffix, "")
        rc = 1 if name.endswith("10R_10L") else 2 if name.endswith("28L_28R") else None
        if rc is not None:
            points[wc][rc] = pd.read_csv(CURVES_DIR / file)

    wind_states = {0: [1, 2], 1: [1], 2: [2], 3: [0]}
    u: Dict[Tuple[int, int], np.ndarray] = {}
    v: Dict[Tuple[int, int], np.ndarray] = {}
    u_max = v_max = 0
    for wc in [0, 1, 2]:
        for rc in [0, 1, 2]:
            if rc == 0:
                u[wc, rc] = np.array([0], dtype=np.int64)
                v[wc, rc] = np.array([0], dtype=np.int64)
            else:
                curve = points[wc][rc]
                u[wc, rc] = np.array(range(int(np.max(curve["x"])) + 1), dtype=np.int64)
                v[wc, rc] = f_curve(np.array(curve["x"]), np.array(curve["y"]), u[wc, rc])
            u_max = max(u_max, int(np.max(u[wc, rc])))
            v_max = max(v_max, int(np.max(v[wc, rc])))

    actions = []
    for rc in [0, 1, 2]:
        if rc == 0:
            for sep_var in range(3):
                actions.append([rc, 0, 0, sep_var])
        else:
            for arr in u[0, rc]:
                for dep in range(int(v[0, rc][arr]) + 1):
                    for sep_var in range(3):
                        actions.append([rc, int(arr), int(dep), sep_var])
    possible_actions = np.array(actions, dtype=np.int64)
    valid_bool = build_valid_bool(possible_actions, u, v, wind_states, u_max, v_max, "all")
    valid_bool_lb = build_valid_bool(possible_actions, u, v, wind_states, u_max, v_max, "lb")
    possible_actions_direct, valid_bool_direct = build_direct_actions_and_mask(u, v, wind_states, u_max, v_max)

    p_wc_trans = np.zeros((3, 3))
    p_ws_trans = np.zeros((4, 4))
    p_wc_error = np.zeros((3, 3))
    p_ws_error = np.zeros((4, 4))
    prev_data = data.loc[: data.shape[0] - 2, :].reset_index(drop=True)
    curr_data = data.loc[1:, :].reset_index(drop=True)
    for i in range(prev_data.shape[0]):
        p_wc_trans[int(prev_data.loc[i, "weather_condition"]), int(curr_data.loc[i, "weather_condition"])] += 1
        p_ws_trans[int(prev_data.loc[i, "wind_state"]), int(curr_data.loc[i, "wind_state"])] += 1
    for i in range(taf_data.shape[0]):
        p_wc_error[int(taf_data.loc[i, "weather_condition"]), int(data.loc[i, "weather_condition"])] += 1
        p_ws_error[int(taf_data.loc[i, "wind_state"]), int(data.loc[i, "wind_state"])] += 1
    p_wc_trans = normalize_rows(p_wc_trans)
    p_ws_trans = normalize_rows(p_ws_trans)
    p_wc_error = normalize_rows(p_wc_error)
    p_ws_error = normalize_rows(p_ws_error)

    days = []
    for day in data["day"]:
        if day not in days:
            days.append(day)
    observation_low = [0, 0, 0, 0, 0, SEP_MIN, 0]
    observation_high = [2, 50, 50, 2, 3, SEP_MAX, int(np.max(data["arrival_demand_100"]))]

    # The original script identifies congested days by simulating a naive baseline
    # over the full two-year set. For this reduced ablation, avoid recomputing that
    # expensive filter and retain all available non-test days as the stochastic
    # training pool while forcing the reviewer-selected severe days into scope.
    congested_days = days
    data = data[data.day.isin(congested_days)].reset_index(drop=True)
    training_days = [day for day in congested_days if day not in TEST_DAYS_209]
    if verbose:
        print(f"Prepared {len(congested_days)} available days; {len(training_days)} non-test training days.")

    return Setup(data, taf_data, std_taf_data, days, congested_days, training_days, u, v, u_max, v_max,
                 possible_actions, valid_bool, valid_bool_lb, possible_actions_direct, valid_bool_direct,
                 wind_states, observation_low, observation_high, arr_mu_40, arr_sigma_40,
                 arr_mu_100, arr_sigma_100, dep_mu, dep_sigma, arr_mean_40, arr_std_40,
                 arr_mean_100, arr_std_100, dep_mean, dep_std, p_wc_trans, p_ws_trans,
                 p_wc_error, p_ws_error)


def make_fb_env(setup: Setup, day: str, env_cls=tdcb_sp.RunwayEnv, valid_bool: Optional[np.ndarray] = None):
    return env_cls(alpha=ALPHA, beta=BETA, delta=DELTA,
                   taf_data=setup.taf_data.copy(), std_taf_data=setup.std_taf_data.copy(), data=setup.data.copy(),
                   congested_days=setup.training_days, u=setup.u, u_max=setup.u_max, v=setup.v, v_max=setup.v_max,
                   possible_actions=setup.possible_actions, valid_bool=valid_bool if valid_bool is not None else setup.valid_bool,
                   valid_bool_lb=setup.valid_bool_lb, sep_min=SEP_MIN, sep_max=SEP_MAX, sep_step=SEP_STEP,
                   wind_states=setup.wind_states, observation_low=setup.observation_low,
                   observation_high=setup.observation_high, arr_mu_40=setup.arr_mu_40,
                   arr_sigma_40=setup.arr_sigma_40, arr_mu_100=setup.arr_mu_100,
                   arr_sigma_100=setup.arr_sigma_100, dep_mu=setup.dep_mu, dep_sigma=setup.dep_sigma,
                   arr_mean_40=setup.arr_mean_40, arr_std_40=setup.arr_std_40,
                   arr_mean_100=setup.arr_mean_100, arr_std_100=setup.arr_std_100,
                   dep_mean=setup.dep_mean, dep_std=setup.dep_std,
                   p_wc_trans=setup.p_wc_trans, p_ws_trans=setup.p_ws_trans,
                   p_wc_error=setup.p_wc_error, p_ws_error=setup.p_ws_error,
                   chosen_day=day)


class NoSmoothingFBEnv(tdcb_sp.RunwayEnv):
    def step(self, action_index, training=True, evaluate=False):
        action = self.possible_actions[action_index]
        RC_i = int(self.obs[0])
        a_i = int(self.obs[1])
        d_i = int(self.obs[2])
        self.prev_prev_sep = self.prev_sep
        self.prev_sep = int(self.obs[5])
        RC_f = int(action[0])
        a_served = int(action[1])
        d_served = int(action[2])
        s_t = self.prev_sep + self.sep_step * (int(action[3]) - 1)
        a_f = a_i - a_served
        d_f = d_i - d_served
        f_100_t = self.day_data.loc[self.time_step, "arrival_demand_100"]
        reward = self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, f_100_t, s_t, training, evaluate)
        self.configs.append(RC_f)
        self.arrival_queue.append(a_f)
        self.departure_queue.append(d_f)
        self.arrivals_served.append(a_served)
        self.departures_served.append(d_served)
        self.separations.append(s_t)
        self.objective -= reward
        terminated = self.time_step == len(self.day_data) - 1
        if not terminated:
            self.time_step += 1
            self.obs = self._generate_next_state(RC_f, a_f, d_f, s_t, training)
        else:
            self.results[self.day] = {
                "objective": self.objective, "queue_cost": self.queue_cost,
                "switch_cost": self.switch_cost, "sep_cost": self.sep_cost,
                "configs": self.configs, "arrival_queue": self.arrival_queue,
                "departure_queue": self.departure_queue, "arrivals_served": self.arrivals_served,
                "departures_served": self.departures_served, "separations": self.separations,
            }
        return self._get_obs(), reward, terminated, False, self._get_info()


class DirectSeparationFBEnv(tdcb_sp.RunwayEnv):
    def set_of_actions(self) -> np.ndarray:
        arrivals_queue = min(int(self.obs[1]), self.u_max)
        departures_queue = min(int(self.obs[2]), self.v_max)
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        separation = int(self.obs[5])
        return self.possible_actions[self.valid_bool[wind_state, weather_condition, arrivals_queue, departures_queue, separation]]

    def action_masks(self) -> np.ndarray:
        arrivals_queue = min(int(self.obs[1]), self.u_max)
        departures_queue = min(int(self.obs[2]), self.v_max)
        weather_condition = int(self.obs[3])
        wind_state = int(self.obs[4])
        separation = int(self.obs[5])
        return self.valid_bool[wind_state, weather_condition, arrivals_queue, departures_queue, separation]

    def step(self, action_index, training=True, evaluate=False):
        action = self.possible_actions[action_index]
        RC_i = int(self.obs[0])
        a_i = int(self.obs[1])
        d_i = int(self.obs[2])
        self.prev_prev_sep = self.prev_sep
        self.prev_sep = int(self.obs[5])
        RC_f = int(action[0])
        a_served = int(action[1])
        d_served = int(action[2])
        s_t = int(action[3])
        a_f = a_i - a_served
        d_f = d_i - d_served
        f_100_t = self.day_data.loc[self.time_step, "arrival_demand_100"]
        reward = self.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f, f_100_t, s_t, training, evaluate)
        self.configs.append(RC_f)
        self.arrival_queue.append(a_f)
        self.departure_queue.append(d_f)
        self.arrivals_served.append(a_served)
        self.departures_served.append(d_served)
        self.separations.append(s_t)
        self.objective -= reward
        terminated = self.time_step == len(self.day_data) - 1
        if not terminated:
            self.time_step += 1
            self.obs = self._generate_next_state(RC_f, a_f, d_f, s_t, training)
        else:
            self.results[self.day] = {
                "objective": self.objective, "queue_cost": self.queue_cost,
                "switch_cost": self.switch_cost, "sep_cost": self.sep_cost,
                "configs": self.configs, "arrival_queue": self.arrival_queue,
                "departure_queue": self.departure_queue, "arrivals_served": self.arrivals_served,
                "departures_served": self.departures_served, "separations": self.separations,
            }
        return self._get_obs(), reward, terminated, False, self._get_info()


def make_direct_env(setup: Setup, day: str):
    env = make_fb_env(setup, day, env_cls=DirectSeparationFBEnv, valid_bool=setup.valid_bool_direct)
    env.possible_actions = setup.possible_actions_direct
    env.action_space = spaces.Discrete(len(setup.possible_actions_direct))
    return env


def train_or_load_model(path: Path, env, timesteps: int, seed: int) -> Tuple[MaskablePPO, float, bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return MaskablePPO.load(str(path), env=env), 0.0, True
    start = time.perf_counter()
    model = MaskablePPO("MlpPolicy", env, gamma=1, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)
    elapsed = time.perf_counter() - start
    model.save(str(path))
    return model, elapsed, False


def simulate_model(model: MaskablePPO, env) -> Dict[str, object]:
    obs, _ = env.reset(training=False)
    done = False
    total_reward = 0.0
    while not done:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action), training=False, evaluate=False)
        total_reward += float(reward)
    sol = env.results[env.day]
    metrics = metrics_from_solution(env, sol)
    metrics["reward"] = total_reward
    return metrics


def metrics_from_solution(env, sol: Dict[str, object]) -> Dict[str, object]:
    arr_q = np.array(sol["arrival_queue"], dtype=float)
    dep_q = np.array(sol["departure_queue"], dtype=float)
    seps = np.array(sol["separations"], dtype=float)
    configs = np.array(sol["configs"], dtype=float)
    f100 = np.array(env.day_data["arrival_demand_100"].iloc[: len(seps)], dtype=float)
    queue_cost = float(sol["queue_cost"])
    switch_cost = float(sol["switch_cost"])
    sep_cost = float(sol["sep_cost"])
    queue_delay_min = 15.0 * float(arr_q.sum() + dep_q.sum())
    spacing_delay_min = float(np.sum(0.15 * np.maximum(seps - SEP_MIN, 0) * f100))
    return {
        "objective": float(sol["objective"]),
        "queue_cost": queue_cost,
        "switch_cost": switch_cost,
        "sep_cost": sep_cost,
        "operating_cost_eur": P_GROUND * (queue_cost + switch_cost),
        "queue_delay_min": queue_delay_min,
        "spacing_delay_min": spacing_delay_min,
        "total_delay_min": queue_delay_min + spacing_delay_min,
        "mean_sep_nm": float(np.mean(seps)),
        "max_sep_nm": float(np.max(seps)),
        "sep_changes": int(np.sum(np.diff(seps) != 0)) if len(seps) > 1 else 0,
        "config_changes": int(np.sum(np.diff(configs) != 0)) if len(configs) > 1 else 0,
    }


def load_saved_full_model(setup: Setup, day: str, env_cls=tdcb_sp.RunwayEnv):
    env = make_fb_env(setup, day, env_cls=env_cls)
    model_path = ORIGINAL_RESULTS_DIR / f"dcb_ppo_fb_2_{day}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved full FB model: {model_path}")
    return MaskablePPO.load(str(model_path), env=env), env


def simulate_rule_exact(setup: Setup, day: str) -> Dict[str, object]:
    env = make_fb_env(setup, day)
    obs, _ = env.reset(training=False)
    done = False
    total_reward = 0.0
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
        obs, reward, done, _, _ = env.step(best_idx, training=False, evaluate=False)
        new_delta = int(env.possible_actions[best_idx][3]) - 1
        if new_delta != 0:
            prev_delta = new_delta
        total_reward += float(reward)
    sol = env.results[env.day]
    metrics = metrics_from_solution(env, sol)
    metrics["reward"] = total_reward
    return metrics


def add_result(rows: List[Dict[str, object]], day: str, variant: str, metrics: Dict[str, object],
               train_seconds: float = 0.0, loaded: bool = True, timesteps: int = 0) -> None:
    row = {"day": day, "variant": variant, "train_seconds": train_seconds,
           "model_loaded": loaded, "train_timesteps": timesteps}
    row.update(metrics)
    rows.append(row)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "objective", "reward", "queue_cost", "switch_cost", "sep_cost", "operating_cost_eur",
        "queue_delay_min", "spacing_delay_min", "total_delay_min", "mean_sep_nm", "max_sep_nm",
        "sep_changes", "config_changes", "train_seconds",
    ]
    grouped = results.groupby("variant")
    summary = grouped[numeric_cols].mean().reset_index()
    summary.insert(1, "days", grouped.size().to_numpy())
    return summary


def run(days: List[str], timesteps: int, seed: int, include_no_curr: bool, include_direct: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    setup = load_setup(verbose=True)
    rows: List[Dict[str, object]] = []

    for day in days:
        print(f"\n=== {day} ===")
        model, env = load_saved_full_model(setup, day)
        add_result(rows, day, "Full FB saved policy", simulate_model(model, env), 0.0, True, 100000)
        print("Saved full policy evaluated.")

        model_ns, env_ns = load_saved_full_model(setup, day, env_cls=NoSmoothingFBEnv)
        add_result(rows, day, "Full FB without smoothing at execution", simulate_model(model_ns, env_ns), 0.0, True, 100000)
        print("No-smoothing execution evaluated.")

        add_result(rows, day, "Rule-based MIT heuristic", simulate_rule_exact(setup, day), 0.0, True, 0)
        print("Rule heuristic evaluated.")

        if include_no_curr:
            env_nc = make_fb_env(setup, day)
            path = MODEL_DIR / f"no_curr_{timesteps}_{day}.zip"
            model_nc, elapsed, loaded = train_or_load_model(path, env_nc, timesteps, seed)
            eval_env_nc = make_fb_env(setup, day)
            model_nc.set_env(eval_env_nc)
            add_result(rows, day, "FB RL without curriculum", simulate_model(model_nc, eval_env_nc), elapsed, loaded, timesteps)
            print(f"No-curriculum RL evaluated ({elapsed:.1f}s training, loaded={loaded}).")

        if include_direct:
            env_direct = make_direct_env(setup, day)
            path = MODEL_DIR / f"direct_sep_{timesteps}_{day}.zip"
            model_direct, elapsed, loaded = train_or_load_model(path, env_direct, timesteps, seed)
            eval_env_direct = make_direct_env(setup, day)
            model_direct.set_env(eval_env_direct)
            add_result(rows, day, "Direct absolute-separation RL", simulate_model(model_direct, eval_env_direct), elapsed, loaded, timesteps)
            print(f"Direct-separation RL evaluated ({elapsed:.1f}s training, loaded={loaded}).")

        partial = pd.DataFrame(rows)
        partial.to_csv(OUT_DIR / "reviewer2_ablation_results_partial.csv", index=False)
        summarize(partial).to_csv(OUT_DIR / "reviewer2_ablation_summary_partial.csv", index=False)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(OUT_DIR / "reviewer2_ablation_results.csv", index=False)
    summary.to_csv(OUT_DIR / "reviewer2_ablation_summary.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'reviewer2_ablation_results.csv'}")
    print(f"Wrote {OUT_DIR / 'reviewer2_ablation_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reduced Reviewer 2 ablations.")
    parser.add_argument("--days", default=",".join(SEVERE_DAYS_9), help="Comma-separated day list.")
    parser.add_argument("--timesteps", "--episodes", dest="timesteps", type=int, default=50000,
                        help="Training timesteps for reduced ablation models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-no-curr", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only the first day with a tiny training budget.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_days = [d.strip() for d in args.days.split(",") if d.strip()]
    if args.smoke:
        selected_days = selected_days[:1]
        args.timesteps = min(args.timesteps, 1000)
    run(selected_days, args.timesteps, args.seed,
        include_no_curr=not args.skip_no_curr,
        include_direct=not args.skip_direct)




