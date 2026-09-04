"""
exp_03_runtime_scaling.py

Experiment 3: Computational scaling of exact versus Monte Carlo
block-permutation conformal prediction.

Goal
----
Experiments 1 and 2 studied statistical approximation:

    1. q_hat_M -> q
    2. C_hat_M -> C_exact

Experiment 3 studies the computational motivation for introducing
Monte Carlo sampling.

For one candidate future hidden-state sequence, the exact method
evaluates every member of the reduced permutation space

        |Pi| = d! / (d - T1 - 1)!

where d is the number of usable (i, j)-blocks.

The Monte Carlo method instead evaluates only M sampled arrangements.

Ignoring common preprocessing and constants, their dominant costs are

    Exact:
        O(|Pi| * T1 * |X|^2)

    Monte Carlo:
        O(M * T1 * |X|^2)

so the approximate score-evaluation workload ratio is

        |Pi| / M.

Design
------
We hold M fixed and increase the prediction horizon T1.

To isolate the permutation bottleneck, we evaluate ONE candidate at
each horizon rather than constructing the entire prediction set.
The candidate is the true future hidden-state sequence for that
horizon.

The calibration sequence is held fixed. A single long HMM realization
is generated, and increasing horizons use progressively longer
prefixes of the same future trajectory.

Exact enumeration is automatically skipped when |Pi| exceeds a
safety threshold. This prevents an accidental multi-hour exhaustive
enumeration while still recording how large the exact workload would
have been.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.simulation import (
    simulate_binary_experiment,
    estimate_hmm_parameters,
    UndefinedHMMEstimateError,
)

from src.conformal import (
    ordered_arrangement_count,
)

from src.exact_method import (
    exact_q_for_candidate,
    find_valid_block_family,
)

from src.monte_carlo_method import (
    monte_carlo_q_for_candidate,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Same basic HMM regime used in Experiments 1 and 2.

P = 0.7
B = 0.75

T = 100

N_STATES = 2
N_OBSERVATIONS = 2

DATA_SEED = 20260904
MC_MASTER_SEED = 314159


# Prediction horizons to investigate.

T1_GRID = [
    1,
    2,
    3,
    4,
    5,
    6,
]


# Fixed Monte Carlo computational budget.

M_FIXED = 1000


# Repeat MC timing several times because wall-clock measurements have
# some noise.

N_MC_TIMING_REPEATS = 3


# Safety limit for exact enumeration.
#
# If |Pi| is larger than this value, we DO NOT execute the exact
# enumeration. We still record |Pi| and the theoretical workload
# ratio |Pi| / M.
#
# Increase this only if you intentionally want longer exact runs.

MAX_EXACT_PERMUTATIONS = 200_000


# ---------------------------------------------------------------------
# Candidate preflight inspection
# ---------------------------------------------------------------------


def inspect_candidate(
    calibration_states,
    calibration_observations,
    future_observations,
    candidate,
) -> dict:
    """
    Inspect the exact permutation problem BEFORE enumeration.

    This reproduces the preprocessing common to the exact and
    Monte Carlo procedures and determines

        - whether the empirical HMM estimate exists,
        - whether a valid block family exists,
        - number of usable blocks d,
        - size of exact permutation space |Pi|.

    No permutation scores are evaluated here.
    """

    X_cal = np.asarray(
        calibration_states,
        dtype=np.int64,
    )

    Y_cal = np.asarray(
        calibration_observations,
        dtype=np.int64,
    )

    Y_future = np.asarray(
        future_observations,
        dtype=np.int64,
    )

    x = np.asarray(
        candidate,
        dtype=np.int64,
    )

    horizon = len(x)

    X_augmented = np.concatenate(
        [
            X_cal,
            x,
        ]
    )

    Y_augmented = np.concatenate(
        [
            Y_cal,
            Y_future,
        ]
    )

    # -------------------------------------------------------------
    # Check whether P_hat and B_hat are defined.
    # -------------------------------------------------------------

    try:
        estimate_hmm_parameters(
            states=X_augmented,
            observations=Y_augmented,
            n_states=N_STATES,
            n_observations=N_OBSERVATIONS,
        )

    except UndefinedHMMEstimateError:

        return {
            "valid": False,
            "fallback_reason": "undefined_hmm_estimate",
            "n_blocks": 0,
            "permutation_count": 0,
            "block_pair": None,
            "ij_index": None,
        }

    # -------------------------------------------------------------
    # Find the same block family used by exact and MC methods.
    # -------------------------------------------------------------

    (
        pair,
        blocks,
        ij_index,
    ) = find_valid_block_family(
        states=X_augmented,
        observations=Y_augmented,
        horizon=horizon,
    )

    if pair is None:

        return {
            "valid": False,
            "fallback_reason": "insufficient_blocks",
            "n_blocks": 0,
            "permutation_count": 0,
            "block_pair": None,
            "ij_index": None,
        }

    permutation_count = ordered_arrangement_count(
        n_blocks=len(blocks),
        n_selected=horizon + 1,
    )

    return {
        "valid": True,
        "fallback_reason": None,
        "n_blocks": len(blocks),
        "permutation_count": permutation_count,
        "block_pair": pair,
        "ij_index": ij_index,
    }


# ---------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------


def time_exact_candidate(
    X_cal,
    Y_cal,
    Y_future,
    candidate,
):
    """
    Time one complete exact q-value computation.
    """

    start = perf_counter()

    result = exact_q_for_candidate(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        candidate=candidate,
        n_states=N_STATES,
        n_observations=N_OBSERVATIONS,
    )

    elapsed = (
        perf_counter()
        - start
    )

    if result.q_value is None:
        raise RuntimeError(
            "Exact method unexpectedly used fallback during timing: "
            f"{result.fallback_reason}"
        )

    return (
        float(elapsed),
        float(result.q_value),
    )


def time_mc_candidate(
    X_cal,
    Y_cal,
    Y_future,
    candidate,
    M: int,
    seed: int,
):
    """
    Time one complete Monte Carlo q-value computation.
    """

    rng = np.random.default_rng(
        seed
    )

    start = perf_counter()

    result = monte_carlo_q_for_candidate(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        candidate=candidate,
        n_states=N_STATES,
        n_observations=N_OBSERVATIONS,
        M=M,
        rng=rng,
    )

    elapsed = (
        perf_counter()
        - start
    )

    if result.q_hat is None:
        raise RuntimeError(
            "Monte Carlo method unexpectedly used fallback during "
            f"timing: {result.fallback_reason}"
        )

    return (
        float(elapsed),
        float(result.q_hat),
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:

    max_horizon = max(
        T1_GRID
    )

    # -------------------------------------------------------------
    # Generate ONE long trajectory.
    #
    # Every horizon uses the same calibration data and a prefix of
    # this same future trajectory.
    # -------------------------------------------------------------

    (
        X_cal,
        Y_cal,
        X_future_full,
        Y_future_full,
    ) = simulate_binary_experiment(
        p=P,
        b=B,
        T=T,
        T1=max_horizon,
        seed=DATA_SEED,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Experiment 3: Runtime scaling"
    )
    print(
        "========================================"
    )

    print(
        f"HMM parameters: p={P}, b={B}"
    )

    print(
        f"Calibration length T={T}"
    )

    print(
        f"Horizons={T1_GRID}"
    )

    print(
        f"Fixed Monte Carlo budget M={M_FIXED}"
    )

    print(
        f"MC timing repeats={N_MC_TIMING_REPEATS}"
    )

    print(
        f"Exact enumeration safety cap="
        f"{MAX_EXACT_PERMUTATIONS:,}"
    )

    print()

    print(
        "Full future hidden trajectory:"
    )

    print(
        tuple(
            int(v)
            for v in X_future_full
        )
    )

    print(
        "Full future observation trajectory:"
    )

    print(
        tuple(
            int(v)
            for v in Y_future_full
        )
    )

    print()

    # -------------------------------------------------------------
    # Output directories
    # -------------------------------------------------------------

    table_directory = (
        PROJECT_ROOT
        / "results"
        / "tables"
    )

    figure_directory = (
        PROJECT_ROOT
        / "results"
        / "figures"
    )

    table_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng_master = np.random.default_rng(
        MC_MASTER_SEED
    )

    rows = []

    # =============================================================
    # Loop over prediction horizons
    # =============================================================

    for T1 in T1_GRID:

        Y_future = np.asarray(
            Y_future_full[:T1],
            dtype=np.int64,
        )

        candidate = np.asarray(
            X_future_full[:T1],
            dtype=np.int64,
        )

        candidate_tuple = tuple(
            int(v)
            for v in candidate
        )

        print(
            "----------------------------------------"
        )

        print(
            f"T1={T1}"
        )

        print(
            f"candidate={candidate_tuple}"
        )

        # ---------------------------------------------------------
        # Preflight:
        # determine d and |Pi| without enumerating Pi.
        # ---------------------------------------------------------

        inspection = inspect_candidate(
            calibration_states=X_cal,
            calibration_observations=Y_cal,
            future_observations=Y_future,
            candidate=candidate,
        )

        if not inspection["valid"]:

            print(
                "    skipped because preprocessing used fallback:"
            )

            print(
                f"    {inspection['fallback_reason']}"
            )

            rows.append(
                {
                    "T1": T1,
                    "candidate": str(
                        candidate_tuple
                    ),
                    "n_blocks": 0,
                    "permutation_count": 0,
                    "M": M_FIXED,
                    "workload_ratio_exact_over_mc": (
                        float("nan")
                    ),
                    "mc_runtime_mean_seconds": (
                        float("nan")
                    ),
                    "mc_runtime_sd_seconds": (
                        float("nan")
                    ),
                    "mean_q_hat": (
                        float("nan")
                    ),
                    "exact_runtime_seconds": (
                        float("nan")
                    ),
                    "exact_q": (
                        float("nan")
                    ),
                    "absolute_q_error": (
                        float("nan")
                    ),
                    "measured_speedup": (
                        float("nan")
                    ),
                    "exact_executed": False,
                    "fallback_reason": (
                        inspection[
                            "fallback_reason"
                        ]
                    ),
                }
            )

            continue

        n_blocks = int(
            inspection[
                "n_blocks"
            ]
        )

        permutation_count = int(
            inspection[
                "permutation_count"
            ]
        )

        workload_ratio = (
            permutation_count
            / M_FIXED
        )

        print(
            f"    blocks d={n_blocks}"
        )

        print(
            f"    |Pi|={permutation_count:,}"
        )

        print(
            f"    |Pi| / M={workload_ratio:,.3f}"
        )

        if permutation_count < M_FIXED:

            print(
                "    NOTE: |Pi| < M, so exact enumeration "
                "requires fewer score evaluations than the "
                "fixed-M Monte Carlo method."
            )

        else:

            print(
                "    MC evaluates fewer permutation scores "
                "than exact enumeration."
            )

        # ---------------------------------------------------------
        # Monte Carlo timing.
        # ---------------------------------------------------------

        mc_times = []
        mc_q_values = []

        for _ in range(
            N_MC_TIMING_REPEATS
        ):

            mc_seed = int(
                rng_master.integers(
                    0,
                    np.iinfo(np.uint32).max,
                )
            )

            elapsed, q_hat = time_mc_candidate(
                X_cal=X_cal,
                Y_cal=Y_cal,
                Y_future=Y_future,
                candidate=candidate,
                M=M_FIXED,
                seed=mc_seed,
            )

            mc_times.append(
                elapsed
            )

            mc_q_values.append(
                q_hat
            )

        mc_times_array = np.asarray(
            mc_times,
            dtype=float,
        )

        mc_runtime_mean = float(
            np.mean(
                mc_times_array
            )
        )

        if len(mc_times_array) > 1:

            mc_runtime_sd = float(
                np.std(
                    mc_times_array,
                    ddof=1,
                )
            )

        else:

            mc_runtime_sd = 0.0

        mean_q_hat = float(
            np.mean(
                mc_q_values
            )
        )

        print(
            f"    MC runtime="
            f"{mc_runtime_mean:.6f}s "
            f"(SD={mc_runtime_sd:.6f}s)"
        )

        print(
            f"    mean q_hat="
            f"{mean_q_hat:.6f}"
        )

        # ---------------------------------------------------------
        # Exact timing only when safe.
        # ---------------------------------------------------------

        if (
            permutation_count
            <= MAX_EXACT_PERMUTATIONS
        ):

            print(
                "    running exact enumeration...",
                flush=True,
            )

            (
                exact_runtime,
                exact_q,
            ) = time_exact_candidate(
                X_cal=X_cal,
                Y_cal=Y_cal,
                Y_future=Y_future,
                candidate=candidate,
            )

            measured_speedup = (
                exact_runtime
                / mc_runtime_mean
            )

            absolute_q_error = abs(
                mean_q_hat
                - exact_q
            )

            exact_executed = True

            print(
                f"    exact runtime="
                f"{exact_runtime:.6f}s"
            )

            print(
                f"    exact q="
                f"{exact_q:.6f}"
            )

            print(
                f"    |mean q_hat - q|="
                f"{absolute_q_error:.6f}"
            )

            print(
                f"    measured exact/MC runtime ratio="
                f"{measured_speedup:.3f}x"
            )

        else:

            exact_runtime = float(
                "nan"
            )

            exact_q = float(
                "nan"
            )

            absolute_q_error = float(
                "nan"
            )

            measured_speedup = float(
                "nan"
            )

            exact_executed = False

            print(
                "    exact enumeration SKIPPED"
            )

            print(
                f"    because |Pi|="
                f"{permutation_count:,} exceeds "
                f"the safety cap "
                f"{MAX_EXACT_PERMUTATIONS:,}."
            )

        rows.append(
            {
                "T1": T1,
                "candidate": str(
                    candidate_tuple
                ),
                "n_blocks": n_blocks,
                "permutation_count": (
                    permutation_count
                ),
                "M": M_FIXED,
                "workload_ratio_exact_over_mc": (
                    workload_ratio
                ),
                "mc_runtime_mean_seconds": (
                    mc_runtime_mean
                ),
                "mc_runtime_sd_seconds": (
                    mc_runtime_sd
                ),
                "mean_q_hat": mean_q_hat,
                "exact_runtime_seconds": (
                    exact_runtime
                ),
                "exact_q": exact_q,
                "absolute_q_error": (
                    absolute_q_error
                ),
                "measured_speedup": (
                    measured_speedup
                ),
                "exact_executed": (
                    exact_executed
                ),
                "fallback_reason": None,
            }
        )

        print()

    # =============================================================
    # Save CSV
    # =============================================================

    output_csv = (
        table_directory
        / "exp03_runtime_scaling.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # =============================================================
    # Keep only valid preprocessing rows for plotting.
    # =============================================================

    valid_rows = [
        row
        for row in rows
        if row["fallback_reason"]
        is None
    ]

    if not valid_rows:
        raise RuntimeError(
            "No valid horizons were available for plotting."
        )

    horizons = np.asarray(
        [
            row["T1"]
            for row in valid_rows
        ],
        dtype=float,
    )

    permutation_counts = np.asarray(
        [
            row[
                "permutation_count"
            ]
            for row in valid_rows
        ],
        dtype=float,
    )

    mc_runtimes = np.asarray(
        [
            row[
                "mc_runtime_mean_seconds"
            ]
            for row in valid_rows
        ],
        dtype=float,
    )

    # =============================================================
    # Figure 1:
    # Actual measured runtime versus T1
    # =============================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        horizons,
        mc_runtimes,
        marker="o",
        label=(
            f"Monte Carlo, M={M_FIXED}"
        ),
    )

    exact_rows = [
        row
        for row in valid_rows
        if row[
            "exact_executed"
        ]
    ]

    if exact_rows:

        exact_horizons = np.asarray(
            [
                row["T1"]
                for row in exact_rows
            ],
            dtype=float,
        )

        exact_runtimes = np.asarray(
            [
                row[
                    "exact_runtime_seconds"
                ]
                for row in exact_rows
            ],
            dtype=float,
        )

        plt.plot(
            exact_horizons,
            exact_runtimes,
            marker="s",
            label="Exact enumeration",
        )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Prediction horizon T1"
    )

    plt.ylabel(
        "Runtime per candidate (seconds)"
    )

    plt.title(
        "Exact versus Monte Carlo runtime scaling"
    )

    plt.legend()

    plt.tight_layout()

    runtime_figure = (
        figure_directory
        / "exp03_runtime_vs_horizon.png"
    )

    plt.savefig(
        runtime_figure,
        dpi=300,
    )

    plt.close()

    # =============================================================
    # Figure 2:
    # Number of permutation-score evaluations.
    #
    # Exact uses |Pi|.
    # MC always uses M.
    # =============================================================

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        horizons,
        permutation_counts,
        marker="o",
        label="Exact: |Pi| score evaluations",
    )

    plt.plot(
        horizons,
        np.full_like(
            horizons,
            M_FIXED,
        ),
        linestyle="--",
        label=f"Monte Carlo: M={M_FIXED}",
    )

    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Prediction horizon T1"
    )

    plt.ylabel(
        "Number of permutation-score evaluations"
    )

    plt.title(
        "Combinatorial exact workload versus fixed Monte Carlo budget"
    )

    plt.legend()

    plt.tight_layout()

    workload_figure = (
        figure_directory
        / "exp03_workload_vs_horizon.png"
    )

    plt.savefig(
        workload_figure,
        dpi=300,
    )

    plt.close()

    # =============================================================
    # Final summary
    # =============================================================

    print()
    print(
        "========================================"
    )

    print(
        "Experiment 3 complete"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Summary:"
    )

    for row in valid_rows:

        exact_text = (
            f"{row['exact_runtime_seconds']:.4f}s"
            if row["exact_executed"]
            else "SKIPPED"
        )

        print(
            f"T1={row['T1']} | "
            f"d={row['n_blocks']} | "
            f"|Pi|={row['permutation_count']:,} | "
            f"|Pi|/M="
            f"{row['workload_ratio_exact_over_mc']:,.2f} | "
            f"MC={row['mc_runtime_mean_seconds']:.4f}s | "
            f"Exact={exact_text}"
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        output_csv
    )

    print(
        runtime_figure
    )

    print(
        workload_figure
    )


if __name__ == "__main__":
    main()