"""
exp_05_real_data_hapt.py

Experiment 5: Real-data validation on the UCI HAPT smartphone-activity dataset.

This experiment reuses the same questions studied in the synthetic experiments:

1. Does q_hat_M approach exhaustive q as M increases?
2. Do Monte Carlo prediction sets reproduce the exhaustive sets?
3. Are decision disagreements concentrated near the threshold q = alpha?
4. When is Monte Carlo faster than exhaustive enumeration?
5. How do raw and Hoeffding-corrected prediction sets behave on real data?

Important design change from the first version
----------------------------------------------
The evaluation points are selected with a STATE-BALANCED design. We target the
same number of future stationary and future moving states. This prevents a
trivial 100% coverage result caused by evaluating only one future state.
Coverage is therefore descriptive for this balanced evaluation sample; it is
not an estimate of the natural HAPT state-frequency-weighted coverage.

The script also:
- scans candidate prediction points before running expensive exact enumeration;
- does not select points based on q-values or prediction-set outcomes;
- uses calibration-only preprocessing for sensor discretization;
- uses nested Monte Carlo samples across M for each candidate, so increasing M
  extends the same random-permutation stream rather than starting from an
  unrelated stream;
- reports overall and state-conditional empirical coverage.

Expected dataset structure
--------------------------
The script searches recursively under project/data for:

    RawData/labels.txt
    RawData/acc_expXX_userYY.txt
    RawData/gyro_expXX_userYY.txt

Outputs
-------
Tables:
    results/tables/exp05_real_data_summary.csv
    results/tables/exp05_real_data_splits.csv
    results/tables/exp05_real_data_candidates.csv
    results/tables/exp05_real_data_sessions.csv

Figures:
    results/figures/exp05_real_q_convergence.png
    results/figures/exp05_real_set_agreement.png
    results/figures/exp05_real_runtime.png
    results/figures/exp05_real_coverage.png
    results/figures/exp05_real_coverage_by_state.png
    results/figures/exp05_real_set_size.png
    results/figures/exp05_real_workload.png
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Project imports
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exact_method import candidate_state_sequences, exact_prediction_set
from src.monte_carlo_method import monte_carlo_q_for_candidate


# ============================================================================
# Configuration
# ============================================================================

ALPHA = 0.20
DELTA = 0.05
T1 = 1
N_STATES = 2

M_GRID = [25, 50, 100, 250, 500, 1000]

# HAPT raw sampling rate is 50 Hz. We aggregate to one-second windows.
WINDOW_SAMPLES = 50

# Number of one-second windows in each calibration sample.
CALIBRATION_WINDOWS = 150

# Balanced real-data evaluation: 25 future stationary + 25 future moving.
TARGET_PER_STATE = 25

# Scan sessions/cuts until both quotas are filled. None means all sessions.
MAX_SESSIONS_TO_SCAN: int | None = None

# Candidate cut points are considered every CUT_STRIDE valid one-second windows.
CUT_STRIDE = 3

# To avoid a calibration sample in which one state is effectively absent.
MIN_STATE_COUNT = 5

# Keep a one-second window when at least this fraction of raw samples is labeled.
MIN_LABELED_FRACTION = 0.90

# We retain majority-labeled windows even near activity changes. The segment
# continuity check below prevents stitching together separated pieces of time.
MIN_STATE_PURITY = 0.50

# Quartile bins for continuous sensor motion score.
QUANTILE_LEVELS = [0.25, 0.50, 0.75]

# Exact-enumeration safety cap per candidate.
MAX_EXACT_PERMUTATIONS_PER_CANDIDATE = 100_000

MASTER_SEED = 20260904


# ============================================================================
# HAPT activity mapping
# ============================================================================

# HAPT basic activity IDs:
# 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS,
# 4 SITTING, 5 STANDING, 6 LAYING.
# IDs 7--12 are postural transitions.

STATIONARY_ACTIVITY_IDS = {4, 5, 6}
MOVING_ACTIVITY_IDS = {1, 2, 3, 7, 8, 9, 10, 11, 12}

STATE_NAMES = {0: "stationary", 1: "moving"}


# ============================================================================
# Small utilities
# ============================================================================


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jaccard_similarity(set_a, set_b) -> float:
    a = set(set_a)
    b = set(set_b)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def hoeffding_epsilon(M: int, delta: float) -> float:
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * M)))


def corrected_threshold(alpha: float, M: int, delta: float) -> float:
    return float(alpha - hoeffding_epsilon(M=M, delta=delta))


def stable_candidate_seed(split_index: int, candidate: tuple[int, ...]) -> int:
    """Deterministic candidate-specific seed, stable across M."""
    candidate_code = 0
    for value in candidate:
        candidate_code = candidate_code * N_STATES + int(value) + 1
    seed_sequence = np.random.SeedSequence(
        [MASTER_SEED, int(split_index), int(candidate_code)]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


# ============================================================================
# Dataset discovery / loading
# ============================================================================


def find_hapt_raw_directory() -> Path:
    data_directory = PROJECT_ROOT / "data"
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_directory}")

    candidates = []
    for labels_path in data_directory.rglob("labels.txt"):
        if labels_path.parent.name.lower() == "rawdata":
            candidates.append(labels_path.parent)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate HAPT RawData/labels.txt.\n"
            "Place the extracted HAPT dataset somewhere under project/data/."
        )

    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def load_hapt_labels(raw_directory: Path) -> np.ndarray:
    labels = np.loadtxt(raw_directory / "labels.txt", dtype=np.int64)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    if labels.shape[1] != 5:
        raise ValueError("HAPT labels.txt should contain exactly five columns.")
    return labels


SESSION_PATTERN = re.compile(r"acc_exp(\d+)_user(\d+)\.txt$", re.IGNORECASE)


def discover_sessions(raw_directory: Path) -> list[dict]:
    sessions: list[dict] = []
    for acc_path in raw_directory.glob("acc_exp*_user*.txt"):
        match = SESSION_PATTERN.match(acc_path.name)
        if match is None:
            continue

        experiment_id = int(match.group(1))
        user_id = int(match.group(2))
        gyro_name = re.sub(r"^acc_", "gyro_", acc_path.name, flags=re.IGNORECASE)
        gyro_path = raw_directory / gyro_name
        if not gyro_path.exists():
            continue

        sessions.append(
            {
                "experiment_id": experiment_id,
                "user_id": user_id,
                "acc_path": acc_path,
                "gyro_path": gyro_path,
            }
        )

    sessions.sort(key=lambda item: (item["experiment_id"], item["user_id"]))
    if not sessions:
        raise FileNotFoundError("No matched accelerometer/gyroscope HAPT sessions found.")
    return sessions


def activity_to_binary_state(activity_id: int) -> int:
    if activity_id in STATIONARY_ACTIVITY_IDS:
        return 0
    if activity_id in MOVING_ACTIVITY_IDS:
        return 1
    return -1


def load_raw_session(session: dict, all_labels: np.ndarray):
    experiment_id = session["experiment_id"]
    user_id = session["user_id"]

    acceleration = np.loadtxt(session["acc_path"], dtype=float)
    gyroscope = np.loadtxt(session["gyro_path"], dtype=float)

    if acceleration.ndim != 2 or acceleration.shape[1] < 3:
        raise ValueError("Accelerometer data must have at least three columns.")
    if gyroscope.ndim != 2 or gyroscope.shape[1] < 3:
        raise ValueError("Gyroscope data must have at least three columns.")

    n_samples = min(len(acceleration), len(gyroscope))
    acceleration = acceleration[:n_samples, :3]
    gyroscope = gyroscope[:n_samples, :3]

    sample_states = np.full(n_samples, -1, dtype=np.int64)

    session_labels = all_labels[
        (all_labels[:, 0] == experiment_id) & (all_labels[:, 1] == user_id)
    ]

    for _, _, activity_id, start_sample, end_sample in session_labels:
        state = activity_to_binary_state(int(activity_id))
        if state < 0:
            continue

        start = max(0, int(start_sample))
        end = min(n_samples - 1, int(end_sample))
        if end >= start:
            sample_states[start : end + 1] = state

    return acceleration, gyroscope, sample_states


# ============================================================================
# Raw signals -> one-second state/features
# ============================================================================


def build_windowed_sequence(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    sample_states: np.ndarray,
):
    n_samples = min(len(acceleration), len(gyroscope), len(sample_states))

    states: list[int] = []
    features: list[list[float]] = []
    raw_start_indices: list[int] = []

    for start in range(0, n_samples - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        end = start + WINDOW_SAMPLES
        state_window = sample_states[start:end]
        valid_mask = state_window >= 0

        labeled_fraction = float(np.mean(valid_mask))
        if labeled_fraction < MIN_LABELED_FRACTION:
            continue

        valid_states = state_window[valid_mask]
        counts = np.bincount(valid_states, minlength=N_STATES)
        majority_state = int(np.argmax(counts))
        state_purity = float(counts[majority_state] / len(valid_states))
        if state_purity < MIN_STATE_PURITY:
            continue

        acc_window = acceleration[start:end]
        gyro_window = gyroscope[start:end]

        # Motion feature 1: RMS magnitude of accelerometer first differences.
        acc_difference = np.diff(acc_window, axis=0)
        acc_change_rms = float(
            np.sqrt(np.mean(np.sum(acc_difference**2, axis=1)))
        )

        # Motion feature 2: RMS angular velocity.
        gyro_rms = float(np.sqrt(np.mean(np.sum(gyro_window**2, axis=1))))

        if not (np.isfinite(acc_change_rms) and np.isfinite(gyro_rms)):
            continue

        states.append(majority_state)
        features.append([acc_change_rms, gyro_rms])
        raw_start_indices.append(start)

    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(features, dtype=float),
        np.asarray(raw_start_indices, dtype=np.int64),
    )


def raw_segment_is_contiguous(raw_indices: np.ndarray) -> bool:
    if len(raw_indices) <= 1:
        return True
    return bool(np.all(np.diff(raw_indices) == WINDOW_SAMPLES))


# ============================================================================
# Calibration-only observation discretization
# ============================================================================


def discretize_sensor_features(
    segment_features: np.ndarray,
    calibration_length: int,
):
    if segment_features.ndim != 2 or segment_features.shape[1] != 2:
        raise ValueError("segment_features must have shape (n, 2).")

    calibration_features = segment_features[:calibration_length]
    means = np.mean(calibration_features, axis=0)
    standard_deviations = np.std(calibration_features, axis=0, ddof=1)
    standard_deviations = np.where(standard_deviations > 1e-12, standard_deviations, 1.0)

    standardized = (segment_features - means) / standard_deviations
    motion_score = np.mean(standardized, axis=1)
    calibration_score = motion_score[:calibration_length]

    raw_edges = np.quantile(calibration_score, QUANTILE_LEVELS)
    edges = np.unique(raw_edges)
    observations = np.digitize(motion_score, edges, right=False).astype(np.int64)
    n_observations = int(len(edges) + 1)

    return observations, n_observations, edges


# ============================================================================
# Split screening
# ============================================================================


def calibration_has_both_states(calibration_states: np.ndarray) -> bool:
    if len(calibration_states) < 2:
        return False

    transition_origins = calibration_states[:-1]
    for state in range(N_STATES):
        if np.sum(transition_origins == state) < MIN_STATE_COUNT:
            return False
    return True


def inspect_split_before_exact(
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    Y_future: np.ndarray,
    n_observations: int,
    seed: int,
):
    """
    Use M=1 only to obtain deterministic preprocessing diagnostics such as the
    exact permutation counts. No exhaustive enumeration is performed here.
    """
    rng = np.random.default_rng(seed)
    diagnostics = []

    for candidate in candidate_state_sequences(n_states=N_STATES, horizon=T1):
        result = monte_carlo_q_for_candidate(
            calibration_states=X_cal,
            calibration_observations=Y_cal,
            future_observations=Y_future,
            candidate=np.asarray(candidate, dtype=np.int64),
            n_states=N_STATES,
            n_observations=n_observations,
            M=1,
            rng=rng,
        )

        if result.q_hat is None or result.used_fallback:
            return None

        permutation_count = int(result.exact_permutation_count)
        if permutation_count > MAX_EXACT_PERMUTATIONS_PER_CANDIDATE:
            return None

        diagnostics.append(
            {
                "candidate": result.candidate,
                "n_blocks": int(result.n_blocks),
                "permutation_count": permutation_count,
            }
        )

    return diagnostics


def make_candidate_split(
    session: dict,
    window_states: np.ndarray,
    window_features: np.ndarray,
    raw_window_indices: np.ndarray,
    cut: int,
):
    cal_start = cut - CALIBRATION_WINDOWS
    future_end = cut + T1

    if cal_start < 0 or future_end > len(window_states):
        return None

    raw_segment = raw_window_indices[cal_start:future_end]
    if not raw_segment_is_contiguous(raw_segment):
        return None

    X_segment = window_states[cal_start:future_end]
    feature_segment = window_features[cal_start:future_end]

    X_cal = X_segment[:CALIBRATION_WINDOWS]
    X_future = X_segment[CALIBRATION_WINDOWS:]

    if not calibration_has_both_states(X_cal):
        return None

    Y_segment, n_observations, edges = discretize_sensor_features(
        segment_features=feature_segment,
        calibration_length=CALIBRATION_WINDOWS,
    )

    if n_observations < 2:
        return None

    Y_cal = Y_segment[:CALIBRATION_WINDOWS]
    Y_future = Y_segment[CALIBRATION_WINDOWS:]

    # Every category produced by calibration quantile binning should normally
    # occur in calibration; keep this defensive check explicit.
    if not set(int(v) for v in Y_future).issubset(set(int(v) for v in Y_cal)):
        return None

    return {
        "experiment_id": int(session["experiment_id"]),
        "user_id": int(session["user_id"]),
        "cut": int(cut),
        "raw_sample_index": int(raw_window_indices[cut]),
        "X_cal": X_cal.copy(),
        "X_future": X_future.copy(),
        "Y_cal": Y_cal.copy(),
        "Y_future": Y_future.copy(),
        "n_observations": int(n_observations),
        "observation_edges": edges.copy(),
        "future_state": int(X_future[0]),
    }


# ============================================================================
# Build a balanced evaluation sample
# ============================================================================


def collect_balanced_splits(
    sessions: list[dict],
    all_labels: np.ndarray,
    rng_master: np.random.Generator,
):
    selected = {0: [], 1: []}
    session_rows: list[dict] = []

    # Randomize session order so quotas are not filled only from the earliest
    # experiments/users.
    session_order = rng_master.permutation(len(sessions))
    if MAX_SESSIONS_TO_SCAN is not None:
        session_order = session_order[:MAX_SESSIONS_TO_SCAN]

    n_preflight_rejected = 0
    n_basic_rejected = 0

    for session_index in session_order:
        if all(len(selected[s]) >= TARGET_PER_STATE for s in range(N_STATES)):
            break

        session = sessions[int(session_index)]
        experiment_id = int(session["experiment_id"])
        user_id = int(session["user_id"])

        try:
            acceleration, gyroscope, sample_states = load_raw_session(
                session=session,
                all_labels=all_labels,
            )
            window_states, window_features, raw_window_indices = build_windowed_sequence(
                acceleration=acceleration,
                gyroscope=gyroscope,
                sample_states=sample_states,
            )
        except Exception as error:
            print(f"Skipping exp {experiment_id}, user {user_id}: {error}")
            continue

        n_windows = len(window_states)
        if n_windows < CALIBRATION_WINDOWS + T1 + 1:
            continue

        first_cut = CALIBRATION_WINDOWS
        last_cut = n_windows - T1
        cuts = np.arange(first_cut, last_cut + 1, CUT_STRIDE, dtype=int)
        rng_master.shuffle(cuts)

        selected_before = {s: len(selected[s]) for s in range(N_STATES)}

        for cut in cuts:
            if all(len(selected[s]) >= TARGET_PER_STATE for s in range(N_STATES)):
                break

            split = make_candidate_split(
                session=session,
                window_states=window_states,
                window_features=window_features,
                raw_window_indices=raw_window_indices,
                cut=int(cut),
            )
            if split is None:
                n_basic_rejected += 1
                continue

            future_state = split["future_state"]
            if len(selected[future_state]) >= TARGET_PER_STATE:
                continue

            preflight_seed = int(
                rng_master.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            )
            preflight = inspect_split_before_exact(
                X_cal=split["X_cal"],
                Y_cal=split["Y_cal"],
                Y_future=split["Y_future"],
                n_observations=split["n_observations"],
                seed=preflight_seed,
            )

            if preflight is None:
                n_preflight_rejected += 1
                continue

            split["preflight"] = preflight
            selected[future_state].append(split)

        added_stationary = len(selected[0]) - selected_before[0]
        added_moving = len(selected[1]) - selected_before[1]

        session_rows.append(
            {
                "experiment_id": experiment_id,
                "user_id": user_id,
                "n_raw_samples": int(len(sample_states)),
                "n_valid_one_second_windows": int(n_windows),
                "stationary_window_fraction": float(np.mean(window_states == 0)),
                "moving_window_fraction": float(np.mean(window_states == 1)),
                "selected_stationary_points": int(added_stationary),
                "selected_moving_points": int(added_moving),
            }
        )

        print(
            f"Scanned exp {experiment_id:02d}, user {user_id:02d} | "
            f"selected totals: stationary={len(selected[0])}/{TARGET_PER_STATE}, "
            f"moving={len(selected[1])}/{TARGET_PER_STATE}"
        )

    # Use the largest balanced sample actually supported by the data and
    # screening rules, up to TARGET_PER_STATE per class. This avoids turning
    # an otherwise valid real-data experiment into an error just because one
    # class is rarer after calibration/preflight filtering.
    achieved_per_state = min(
        TARGET_PER_STATE,
        len(selected[0]),
        len(selected[1]),
    )

    if achieved_per_state == 0:
        raise RuntimeError(
            "No balanced real-data evaluation sample could be formed.\n"
            f"stationary candidates={len(selected[0])}, "
            f"moving candidates={len(selected[1])}."
        )

    if achieved_per_state < TARGET_PER_STATE:
        print()
        print(
            "Requested balanced quota could not be fully reached; "
            "using the largest balanced sample available under the frozen "
            "screening rules."
        )
        print(
            f"Available after screening: stationary={len(selected[0])}, "
            f"moving={len(selected[1])}."
        )
        print(
            f"Final balanced sample: {achieved_per_state} stationary + "
            f"{achieved_per_state} moving = {2 * achieved_per_state}."
        )
        print()

    # Randomize within each state before truncation so selection is not tied
    # to discovery order, then combine and shuffle the final balanced sample.
    rng_master.shuffle(selected[0])
    rng_master.shuffle(selected[1])

    balanced = (
        selected[0][:achieved_per_state]
        + selected[1][:achieved_per_state]
    )
    rng_master.shuffle(balanced)

    return balanced, session_rows, n_basic_rejected, n_preflight_rejected


# ============================================================================
# Evaluate one selected split
# ============================================================================


def evaluate_split(
    split: dict,
    split_index: int,
    total_splits: int,
    rng_master: np.random.Generator,
):
    X_cal = split["X_cal"]
    X_future = split["X_future"]
    Y_cal = split["Y_cal"]
    Y_future = split["Y_future"]
    n_observations = split["n_observations"]

    experiment_id = split["experiment_id"]
    user_id = split["user_id"]
    cut = split["cut"]
    future_state = split["future_state"]

    split_id = f"exp{experiment_id:02d}_user{user_id:02d}_cut{cut}"
    true_sequence = tuple(int(v) for v in X_future)

    # ------------------------------------------------------------------
    # Exact benchmark
    # ------------------------------------------------------------------

    exact_seed = int(
        rng_master.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
    )

    exact_start = perf_counter()
    exact_result = exact_prediction_set(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        alpha=ALPHA,
        n_states=N_STATES,
        n_observations=n_observations,
        seed=exact_seed,
    )
    exact_runtime = perf_counter() - exact_start

    if not all(
        result.q_value is not None and not result.used_fallback
        for result in exact_result.candidate_results
    ):
        raise RuntimeError(f"Exact fallback unexpectedly occurred for {split_id}.")

    exact_set = tuple(exact_result.prediction_set)
    exact_q = {
        result.candidate: float(result.q_value)
        for result in exact_result.candidate_results
    }
    exact_permutation_counts = {
        result.candidate: int(result.n_permutations)
        for result in exact_result.candidate_results
    }

    exact_total_score_evaluations = int(sum(exact_permutation_counts.values()))
    exact_max_candidate_permutations = int(max(exact_permutation_counts.values()))
    exact_covered = bool(true_sequence in exact_set)
    exact_scaled_size = float(len(exact_set) / (N_STATES**T1))

    split_rows: list[dict] = []
    candidate_rows: list[dict] = []

    # Candidate-specific seeds are fixed across M. Thus q_hat at larger M uses
    # a longer prefix of the same random stream.
    candidate_seeds = {
        candidate: stable_candidate_seed(split_index=split_index, candidate=candidate)
        for candidate in candidate_state_sequences(n_states=N_STATES, horizon=T1)
    }

    # ------------------------------------------------------------------
    # Monte Carlo budgets
    # ------------------------------------------------------------------

    for M in M_GRID:
        corrected_t = corrected_threshold(alpha=ALPHA, M=M, delta=DELTA)

        raw_mc_set = []
        corrected_mc_set = []
        mc_candidate_results = []

        mc_start = perf_counter()

        for candidate in candidate_state_sequences(n_states=N_STATES, horizon=T1):
            # Reset to the same candidate-specific seed for every M so the
            # M=25 sample is a prefix of M=50, etc.
            rng = np.random.default_rng(candidate_seeds[candidate])

            result = monte_carlo_q_for_candidate(
                calibration_states=X_cal,
                calibration_observations=Y_cal,
                future_observations=Y_future,
                candidate=np.asarray(candidate, dtype=np.int64),
                n_states=N_STATES,
                n_observations=n_observations,
                M=M,
                rng=rng,
            )

            if result.q_hat is None or result.used_fallback:
                raise RuntimeError(
                    f"Monte Carlo fallback unexpectedly occurred for {split_id}."
                )

            q_hat = float(result.q_hat)
            candidate_tuple = result.candidate

            if q_hat > ALPHA:
                raw_mc_set.append(candidate_tuple)
            if q_hat > corrected_t:
                corrected_mc_set.append(candidate_tuple)

            mc_candidate_results.append(result)

        mc_runtime = perf_counter() - mc_start

        raw_mc_set = tuple(raw_mc_set)
        corrected_mc_set = tuple(corrected_mc_set)

        exact_match = bool(raw_mc_set == exact_set)
        jaccard = float(jaccard_similarity(raw_mc_set, exact_set))
        symmetric_difference = len(set(raw_mc_set) ^ set(exact_set))

        raw_covered = bool(true_sequence in raw_mc_set)
        corrected_covered = bool(true_sequence in corrected_mc_set)

        raw_scaled_size = float(len(raw_mc_set) / (N_STATES**T1))
        corrected_scaled_size = float(len(corrected_mc_set) / (N_STATES**T1))

        mc_score_evaluations = int(M * (N_STATES**T1))
        workload_ratio = float(exact_total_score_evaluations / mc_score_evaluations)
        runtime_ratio = float(exact_runtime / mc_runtime)

        split_rows.append(
            {
                "split_id": split_id,
                "split_index": split_index,
                "experiment_id": experiment_id,
                "user_id": user_id,
                "cut_window": cut,
                "raw_sample_index": split["raw_sample_index"],
                "M": M,
                "alpha": ALPHA,
                "delta": DELTA,
                "corrected_threshold": corrected_t,
                "true_state": future_state,
                "true_state_name": STATE_NAMES[future_state],
                "future_observation": int(Y_future[0]),
                "n_observations": n_observations,
                "exact_prediction_set": str(exact_set),
                "raw_mc_prediction_set": str(raw_mc_set),
                "corrected_mc_prediction_set": str(corrected_mc_set),
                "exact_set_match": exact_match,
                "jaccard": jaccard,
                "symmetric_difference": symmetric_difference,
                "exact_covered": exact_covered,
                "raw_mc_covered": raw_covered,
                "corrected_mc_covered": corrected_covered,
                "exact_scaled_set_size": exact_scaled_size,
                "raw_mc_scaled_set_size": raw_scaled_size,
                "corrected_mc_scaled_set_size": corrected_scaled_size,
                "exact_runtime_seconds": exact_runtime,
                "mc_runtime_seconds": mc_runtime,
                "runtime_ratio_exact_over_mc": runtime_ratio,
                "exact_score_evaluations": exact_total_score_evaluations,
                "exact_max_candidate_permutations": exact_max_candidate_permutations,
                "mc_score_evaluations": mc_score_evaluations,
                "workload_ratio_exact_over_mc": workload_ratio,
            }
        )

        for mc_result in mc_candidate_results:
            candidate = mc_result.candidate
            q_value = exact_q[candidate]
            q_hat = float(mc_result.q_hat)
            error = q_hat - q_value
            margin = abs(q_value - ALPHA)
            exact_decision = bool(q_value > ALPHA)
            mc_decision = bool(q_hat > ALPHA)
            disagreement = bool(exact_decision != mc_decision)
            decision_bound = float(np.exp(-2.0 * M * margin**2))

            candidate_rows.append(
                {
                    "split_id": split_id,
                    "split_index": split_index,
                    "experiment_id": experiment_id,
                    "user_id": user_id,
                    "true_state": future_state,
                    "true_state_name": STATE_NAMES[future_state],
                    "M": M,
                    "candidate": str(candidate),
                    "q_exact": q_value,
                    "q_hat": q_hat,
                    "error": error,
                    "absolute_error": abs(error),
                    "squared_error": error**2,
                    "margin_abs_q_minus_alpha": margin,
                    "exact_decision": exact_decision,
                    "mc_decision": mc_decision,
                    "decision_disagreement": disagreement,
                    "hoeffding_decision_bound": decision_bound,
                    "n_blocks": int(mc_result.n_blocks),
                    "exact_permutation_count": exact_permutation_counts[candidate],
                }
            )

    print(
        f"[{split_index + 1:02d}/{total_splits}] {split_id} | "
        f"true={STATE_NAMES[future_state]} | exact_set={exact_set} | "
        f"exact evals={exact_total_score_evaluations:,} | "
        f"exact runtime={exact_runtime:.4f}s"
    )

    return split_rows, candidate_rows


# ============================================================================
# Aggregate summaries
# ============================================================================


def aggregate_results(split_rows: list[dict], candidate_rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []

    for M in M_GRID:
        M_splits = [row for row in split_rows if row["M"] == M]
        M_candidates = [row for row in candidate_rows if row["M"] == M]

        errors = np.asarray([row["error"] for row in M_candidates], dtype=float)

        row = {
            "M": M,
            "n_splits": len(M_splits),
            "n_stationary_splits": sum(row["true_state"] == 0 for row in M_splits),
            "n_moving_splits": sum(row["true_state"] == 1 for row in M_splits),
            "n_candidate_evaluations": len(M_candidates),
            "alpha": ALPHA,
            "delta": DELTA,
            "corrected_threshold": corrected_threshold(ALPHA, M, DELTA),
            "q_bias": float(np.mean(errors)),
            "q_mae": float(np.mean(np.abs(errors))),
            "q_rmse": float(np.sqrt(np.mean(errors**2))),
            "candidate_decision_disagreement_rate": float(
                np.mean([row["decision_disagreement"] for row in M_candidates])
            ),
            "mean_hoeffding_decision_bound": float(
                np.mean([row["hoeffding_decision_bound"] for row in M_candidates])
            ),
            "exact_set_match_rate": float(
                np.mean([row["exact_set_match"] for row in M_splits])
            ),
            "mean_jaccard": float(np.mean([row["jaccard"] for row in M_splits])),
            "exact_empirical_coverage": float(
                np.mean([row["exact_covered"] for row in M_splits])
            ),
            "raw_mc_empirical_coverage": float(
                np.mean([row["raw_mc_covered"] for row in M_splits])
            ),
            "corrected_mc_empirical_coverage": float(
                np.mean([row["corrected_mc_covered"] for row in M_splits])
            ),
            "exact_mean_scaled_set_size": float(
                np.mean([row["exact_scaled_set_size"] for row in M_splits])
            ),
            "raw_mc_mean_scaled_set_size": float(
                np.mean([row["raw_mc_scaled_set_size"] for row in M_splits])
            ),
            "corrected_mc_mean_scaled_set_size": float(
                np.mean([row["corrected_mc_scaled_set_size"] for row in M_splits])
            ),
            "exact_mean_runtime_seconds": float(
                np.mean([row["exact_runtime_seconds"] for row in M_splits])
            ),
            "mc_mean_runtime_seconds": float(
                np.mean([row["mc_runtime_seconds"] for row in M_splits])
            ),
            "mean_exact_score_evaluations": float(
                np.mean([row["exact_score_evaluations"] for row in M_splits])
            ),
            "mean_mc_score_evaluations": float(
                np.mean([row["mc_score_evaluations"] for row in M_splits])
            ),
        }

        row["runtime_ratio_exact_over_mc"] = (
            row["exact_mean_runtime_seconds"] / row["mc_mean_runtime_seconds"]
        )
        row["workload_ratio_exact_over_mc"] = (
            row["mean_exact_score_evaluations"] / row["mean_mc_score_evaluations"]
        )

        # State-conditional coverage. These are the key diagnostics that were
        # missing in the first version.
        for state in range(N_STATES):
            state_name = STATE_NAMES[state]
            state_rows = [r for r in M_splits if r["true_state"] == state]

            row[f"exact_coverage_{state_name}"] = float(
                np.mean([r["exact_covered"] for r in state_rows])
            )
            row[f"raw_mc_coverage_{state_name}"] = float(
                np.mean([r["raw_mc_covered"] for r in state_rows])
            )
            row[f"corrected_mc_coverage_{state_name}"] = float(
                np.mean([r["corrected_mc_covered"] for r in state_rows])
            )

        summary_rows.append(row)

    return summary_rows


# ============================================================================
# Figures
# ============================================================================


def make_figures(summary_rows: list[dict], figure_directory: Path) -> None:
    M_values = np.asarray([row["M"] for row in summary_rows], dtype=float)

    # 1. q convergence ---------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(M_values, [r["q_mae"] for r in summary_rows], marker="o", label="MAE")
    plt.plot(M_values, [r["q_rmse"] for r in summary_rows], marker="s", label="RMSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Error relative to exact q")
    plt.title("Real-data convergence of Monte Carlo conformal scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_q_convergence.png", dpi=300)
    plt.close()

    # 2. Set agreement ---------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        M_values,
        [r["exact_set_match_rate"] for r in summary_rows],
        marker="o",
        label="Exact-set match rate",
    )
    plt.plot(
        M_values,
        [r["mean_jaccard"] for r in summary_rows],
        marker="s",
        label="Mean Jaccard similarity",
    )
    plt.xscale("log")
    plt.ylim(0.0, 1.02)
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Prediction-set agreement")
    plt.title("Exact versus Monte Carlo prediction sets on HAPT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_set_agreement.png", dpi=300)
    plt.close()

    # 3. Runtime ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        M_values,
        [r["exact_mean_runtime_seconds"] for r in summary_rows],
        linestyle="--",
        label="Exact enumeration",
    )
    plt.plot(
        M_values,
        [r["mc_mean_runtime_seconds"] for r in summary_rows],
        marker="o",
        label="Monte Carlo",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Mean runtime per prediction point (seconds)")
    plt.title("Real-data computational cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_runtime.png", dpi=300)
    plt.close()

    # 4. Overall balanced-sample coverage -------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        M_values,
        [r["exact_empirical_coverage"] for r in summary_rows],
        linestyle="--",
        label="Exact",
    )
    plt.plot(
        M_values,
        [r["raw_mc_empirical_coverage"] for r in summary_rows],
        marker="o",
        label="Raw Monte Carlo",
    )
    plt.plot(
        M_values,
        [r["corrected_mc_empirical_coverage"] for r in summary_rows],
        marker="s",
        label="Hoeffding-corrected MC",
    )
    plt.axhline(1.0 - ALPHA, linestyle=":", label=f"Nominal 1-alpha={1.0 - ALPHA:.2f}")
    plt.xscale("log")
    plt.ylim(0.0, 1.02)
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Empirical coverage")
    plt.title("State-balanced real-data empirical coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_coverage.png", dpi=300)
    plt.close()

    # 5. Coverage by true state -----------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        M_values,
        [r["raw_mc_coverage_stationary"] for r in summary_rows],
        marker="o",
        label="Raw MC: stationary",
    )
    plt.plot(
        M_values,
        [r["raw_mc_coverage_moving"] for r in summary_rows],
        marker="s",
        label="Raw MC: moving",
    )
    plt.plot(
        M_values,
        [r["exact_coverage_stationary"] for r in summary_rows],
        linestyle="--",
        label="Exact: stationary",
    )
    plt.plot(
        M_values,
        [r["exact_coverage_moving"] for r in summary_rows],
        linestyle=":",
        label="Exact: moving",
    )
    plt.xscale("log")
    plt.ylim(0.0, 1.02)
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Empirical coverage")
    plt.title("Real-data empirical coverage by future state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_coverage_by_state.png", dpi=300)
    plt.close()

    # 6. Set size --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        M_values,
        [r["exact_mean_scaled_set_size"] for r in summary_rows],
        linestyle="--",
        label="Exact",
    )
    plt.plot(
        M_values,
        [r["raw_mc_mean_scaled_set_size"] for r in summary_rows],
        marker="o",
        label="Raw Monte Carlo",
    )
    plt.plot(
        M_values,
        [r["corrected_mc_mean_scaled_set_size"] for r in summary_rows],
        marker="s",
        label="Hoeffding-corrected MC",
    )
    plt.xscale("log")
    plt.ylim(0.0, 1.02)
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Mean scaled prediction-set size")
    plt.title("Real-data prediction-set efficiency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_set_size.png", dpi=300)
    plt.close()

    # 7. Workload --------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        M_values,
        [r["mean_exact_score_evaluations"] for r in summary_rows],
        linestyle="--",
        label="Exact permutation evaluations",
    )
    plt.plot(
        M_values,
        [r["mean_mc_score_evaluations"] for r in summary_rows],
        marker="o",
        label="Monte Carlo evaluations",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Monte Carlo sample size M")
    plt.ylabel("Mean conformity-score evaluations per prediction point")
    plt.title("Real-data exhaustive versus Monte Carlo workload")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_directory / "exp05_real_workload.png", dpi=300)
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("\n========================================")
    print("Experiment 5: Balanced real-data HAPT validation")
    print("========================================\n")

    raw_directory = find_hapt_raw_directory()
    print(f"HAPT RawData directory:\n{raw_directory}\n")

    all_labels = load_hapt_labels(raw_directory)
    sessions = discover_sessions(raw_directory)

    print(f"Matched sessions found: {len(sessions)}")
    print(
        f"Target cap: up to {TARGET_PER_STATE} stationary + "
        f"{TARGET_PER_STATE} moving; if unavailable, use the largest "
        "balanced sample supported by the screening rules"
    )
    print(f"Calibration length: {CALIBRATION_WINDOWS} one-second windows")
    print(f"Prediction horizon: T1={T1}")
    print(f"alpha={ALPHA}, delta={DELTA}")
    print(f"M grid={M_GRID}\n")

    table_directory = PROJECT_ROOT / "results" / "tables"
    figure_directory = PROJECT_ROOT / "results" / "figures"
    table_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(MASTER_SEED)

    # ------------------------------------------------------------------
    # Phase 1: select balanced real-data prediction points
    # ------------------------------------------------------------------

    selected_splits, session_rows, n_basic_rejected, n_preflight_rejected = (
        collect_balanced_splits(
            sessions=sessions,
            all_labels=all_labels,
            rng_master=rng_master,
        )
    )

    print("\nBalanced evaluation sample selected:")
    print(f"  stationary = {sum(s['future_state'] == 0 for s in selected_splits)}")
    print(f"  moving     = {sum(s['future_state'] == 1 for s in selected_splits)}")
    print(f"  total      = {len(selected_splits)}")
    print(f"  basic screening rejections   = {n_basic_rejected}")
    print(f"  preflight/workload rejections = {n_preflight_rejected}\n")

    # ------------------------------------------------------------------
    # Phase 2: exact + Monte Carlo evaluation
    # ------------------------------------------------------------------

    split_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for split_index, split in enumerate(selected_splits):
        split_part, candidate_part = evaluate_split(
            split=split,
            split_index=split_index,
            total_splits=len(selected_splits),
            rng_master=rng_master,
        )
        split_rows.extend(split_part)
        candidate_rows.extend(candidate_part)

    summary_rows = aggregate_results(split_rows=split_rows, candidate_rows=candidate_rows)

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------

    summary_csv = table_directory / "exp05_real_data_summary.csv"
    split_csv = table_directory / "exp05_real_data_splits.csv"
    candidate_csv = table_directory / "exp05_real_data_candidates.csv"
    session_csv = table_directory / "exp05_real_data_sessions.csv"

    write_csv(summary_csv, summary_rows)
    write_csv(split_csv, split_rows)
    write_csv(candidate_csv, candidate_rows)
    write_csv(session_csv, session_rows)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    make_figures(summary_rows=summary_rows, figure_directory=figure_directory)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print("\n========================================")
    print("Experiment 5 complete")
    print("========================================\n")

    for row in summary_rows:
        print(
            f"M={row['M']:4d} | "
            f"MAE={row['q_mae']:.4f} | "
            f"RMSE={row['q_rmse']:.4f} | "
            f"set match={row['exact_set_match_rate']:.3f} | "
            f"Jaccard={row['mean_jaccard']:.3f} | "
            f"exact cov={row['exact_empirical_coverage']:.3f} | "
            f"raw cov={row['raw_mc_empirical_coverage']:.3f} | "
            f"corr cov={row['corrected_mc_empirical_coverage']:.3f} | "
            f"raw stationary={row['raw_mc_coverage_stationary']:.3f} | "
            f"raw moving={row['raw_mc_coverage_moving']:.3f} | "
            f"speed ratio={row['runtime_ratio_exact_over_mc']:.2f}x"
        )

    print("\nTables:")
    print(summary_csv)
    print(split_csv)
    print(candidate_csv)
    print(session_csv)

    print("\nFigures saved under:")
    print(figure_directory)

    print(
        "\nInterpretation note: coverage is descriptive for the deliberately "
        "state-balanced evaluation sample. Do not describe it as verification "
        "of the theoretical HMM coverage theorem on HAPT."
    )


if __name__ == "__main__":
    main()