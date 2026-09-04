"""
exp_01_mc_convergence.py

Experiment 1: Monte Carlo convergence.

Goal
----
Verify empirically that

    q_hat_M(x)
        =
    (1 / M) sum_{m=1}^M 1{S(Pi_m) >= S(I)}

converges to the exact permutation quantity

    q(x)
        =
    (1 / |Pi|) sum_{pi in Pi} 1{S(pi) >= S(I)}

as M increases.

The HMM dataset is held FIXED throughout this experiment.
Only the Monte Carlo permutation sampling is repeated.

This cleanly isolates Monte Carlo approximation error from
randomness in the underlying HMM sample path.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.simulation import simulate_binary_experiment

from src.exact_method import exact_q_for_candidate

from src.monte_carlo_method import monte_carlo_q_for_candidate


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

P = 0.7
B = 0.75

T = 100
T1 = 1

N_STATES = 2
N_OBSERVATIONS = 2

DATA_SEED = 20260904
MC_MASTER_SEED = 84721


# Monte Carlo sample-size grid
M_GRID = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


# Number of independent Monte Carlo repetitions for every M
N_MC_REPETITIONS = 200


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------


def summarize_errors(
    q_exact: float,
    q_hats: np.ndarray,
) -> dict:
    """
    Compute Monte Carlo error summaries.
    """

    errors = q_hats - q_exact

    absolute_errors = np.abs(
        errors
    )

    squared_errors = (
        errors ** 2
    )

    bias = float(
        np.mean(errors)
    )

    mae = float(
        np.mean(absolute_errors)
    )

    rmse = float(
        np.sqrt(
            np.mean(squared_errors)
        )
    )

    empirical_sd = float(
        np.std(
            q_hats,
            ddof=1,
        )
    )

    return {
        "mean_q_hat": float(
            np.mean(q_hats)
        ),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "empirical_sd": empirical_sd,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    # -------------------------------------------------------------
    # Generate ONE fixed HMM realization.
    # -------------------------------------------------------------

    (
        X_cal,
        Y_cal,
        X_future,
        Y_future,
    ) = simulate_binary_experiment(
        p=P,
        b=B,
        T=T,
        T1=T1,
        seed=DATA_SEED,
    )

    print()
    print("========================================")
    print("Experiment 1: Monte Carlo convergence")
    print("========================================")

    print(
        f"HMM parameters: p={P}, b={B}"
    )

    print(
        f"T={T}, T1={T1}"
    )

    print(
        f"Fixed data seed={DATA_SEED}"
    )

    print(
        f"MC repetitions per M={N_MC_REPETITIONS}"
    )

    print()

    print(
        "True future hidden sequence:",
        tuple(
            int(v)
            for v in X_future
        ),
    )

    print(
        "Future observations:",
        tuple(
            int(v)
            for v in Y_future
        ),
    )

    print()

    # -------------------------------------------------------------
    # Output locations
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

    # -------------------------------------------------------------
    # We analyze every possible candidate x.
    # For binary T1=1 these are simply (0,) and (1,).
    # -------------------------------------------------------------

    candidates = [
        (state,)
        for state in range(N_STATES)
    ]

    summary_rows = []

    rng_master = np.random.default_rng(
        MC_MASTER_SEED
    )

    # Store data for plotting.
    plot_results = {}

    for candidate in candidates:

        candidate_array = np.asarray(
            candidate,
            dtype=np.int64,
        )

        # ---------------------------------------------------------
        # Exact benchmark
        # ---------------------------------------------------------

        exact_result = exact_q_for_candidate(
            calibration_states=X_cal,
            calibration_observations=Y_cal,
            future_observations=Y_future,
            candidate=candidate_array,
            n_states=N_STATES,
            n_observations=N_OBSERVATIONS,
        )

        if exact_result.q_value is None:

            print(
                f"Skipping candidate {candidate}: "
                f"exact method used fallback "
                f"({exact_result.fallback_reason})."
            )

            continue

        q_exact = float(
            exact_result.q_value
        )

        permutation_count = (
            exact_result.n_permutations
        )

        print(
            f"Candidate {candidate}"
        )

        print(
            f"    exact q = {q_exact:.8f}"
        )

        print(
            f"    |Pi| = {permutation_count}"
        )

        print(
            f"    blocks = {exact_result.n_blocks}"
        )

        print()

        candidate_plot_rows = []

        # ---------------------------------------------------------
        # Vary M
        # ---------------------------------------------------------

        for M in M_GRID:

            q_hats = np.empty(
                N_MC_REPETITIONS,
                dtype=float,
            )

            for repetition in range(
                N_MC_REPETITIONS
            ):

                mc_seed = int(
                    rng_master.integers(
                        0,
                        np.iinfo(np.uint32).max,
                    )
                )

                rng = np.random.default_rng(
                    mc_seed
                )

                mc_result = monte_carlo_q_for_candidate(
                    calibration_states=X_cal,
                    calibration_observations=Y_cal,
                    future_observations=Y_future,
                    candidate=candidate_array,
                    n_states=N_STATES,
                    n_observations=N_OBSERVATIONS,
                    M=M,
                    rng=rng,
                )

                if mc_result.q_hat is None:
                    raise RuntimeError(
                        "Monte Carlo method unexpectedly used "
                        "fallback when exact method was valid."
                    )

                q_hats[repetition] = (
                    mc_result.q_hat
                )

            summaries = summarize_errors(
                q_exact=q_exact,
                q_hats=q_hats,
            )

            # -----------------------------------------------------
            # Theoretical Bernoulli Monte Carlo SD
            #
            # Var(q_hat_M) = q(1-q)/M
            # -----------------------------------------------------

            theoretical_sd = float(
                np.sqrt(
                    q_exact
                    * (1.0 - q_exact)
                    / M
                )
            )

            row = {
                "candidate": str(candidate),
                "M": M,
                "q_exact": q_exact,
                "permutation_count": permutation_count,
                "mc_repetitions": N_MC_REPETITIONS,
                "mean_q_hat": summaries[
                    "mean_q_hat"
                ],
                "bias": summaries[
                    "bias"
                ],
                "mae": summaries[
                    "mae"
                ],
                "rmse": summaries[
                    "rmse"
                ],
                "empirical_sd": summaries[
                    "empirical_sd"
                ],
                "theoretical_sd": theoretical_sd,
            }

            summary_rows.append(
                row
            )

            candidate_plot_rows.append(
                row
            )

            print(
                f"    M={M:4d} | "
                f"mean={summaries['mean_q_hat']:.6f} | "
                f"bias={summaries['bias']:+.6f} | "
                f"MAE={summaries['mae']:.6f} | "
                f"RMSE={summaries['rmse']:.6f} | "
                f"SD={summaries['empirical_sd']:.6f} | "
                f"theory SD={theoretical_sd:.6f}"
            )

        print()

        plot_results[
            candidate
        ] = candidate_plot_rows

    # -------------------------------------------------------------
    # Save summary CSV
    # -------------------------------------------------------------

    output_csv = (
        table_directory
        / "exp01_mc_convergence.csv"
    )

    if not summary_rows:
        raise RuntimeError(
            "No valid candidate results were generated."
        )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # -------------------------------------------------------------
    # Plot 1:
    # MAE versus M
    # -------------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    for candidate, rows in plot_results.items():

        M_values = np.array(
            [
                row["M"]
                for row in rows
            ],
            dtype=float,
        )

        mae_values = np.array(
            [
                row["mae"]
                for row in rows
            ],
            dtype=float,
        )

        plt.plot(
            M_values,
            mae_values,
            marker="o",
            label=f"x={candidate}",
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        "Monte Carlo sample size M"
    )

    plt.ylabel(
        "Mean absolute error"
    )

    plt.title(
        "Monte Carlo convergence to exact q"
    )

    plt.legend()

    plt.tight_layout()

    mae_figure = (
        figure_directory
        / "exp01_mae_vs_M.png"
    )

    plt.savefig(
        mae_figure,
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------
    # Plot 2:
    # empirical SD versus theoretical Bernoulli SD
    # -------------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    for candidate, rows in plot_results.items():

        M_values = np.array(
            [
                row["M"]
                for row in rows
            ],
            dtype=float,
        )

        empirical_values = np.array(
            [
                row["empirical_sd"]
                for row in rows
            ],
            dtype=float,
        )

        theoretical_values = np.array(
            [
                row["theoretical_sd"]
                for row in rows
            ],
            dtype=float,
        )

        plt.plot(
            M_values,
            empirical_values,
            marker="o",
            label=f"Empirical SD, x={candidate}",
        )

        plt.plot(
            M_values,
            theoretical_values,
            linestyle="--",
            label=f"Theoretical SD, x={candidate}",
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        "Monte Carlo sample size M"
    )

    plt.ylabel(
        "Standard deviation"
    )

    plt.title(
        "Empirical versus theoretical Monte Carlo variability"
    )

    plt.legend()

    plt.tight_layout()

    sd_figure = (
        figure_directory
        / "exp01_sd_vs_M.png"
    )

    plt.savefig(
        sd_figure,
        dpi=300,
    )

    plt.close()

    # -------------------------------------------------------------
    # Estimate log-log RMSE slope
    #
    # If RMSE ~ C M^{-1/2}, the slope should be near -0.5.
    # -------------------------------------------------------------

    print(
        "----------------------------------------"
    )

    print(
        "Estimated log-log RMSE slopes"
    )

    print(
        "Expected Monte Carlo slope: approximately -0.5"
    )

    for candidate, rows in plot_results.items():

        M_values = np.array(
            [
                row["M"]
                for row in rows
            ],
            dtype=float,
        )

        rmse_values = np.array(
            [
                row["rmse"]
                for row in rows
            ],
            dtype=float,
        )

        valid = (
            rmse_values > 0
        )

        slope, intercept = np.polyfit(
            np.log(M_values[valid]),
            np.log(rmse_values[valid]),
            deg=1,
        )

        print(
            f"candidate {candidate}: "
            f"slope = {slope:.4f}"
        )

    print(
        "\nResults saved to:"
    )

    print(
        output_csv
    )

    print(
        mae_figure
    )

    print(
        sd_figure
    )


if __name__ == "__main__":
    main()