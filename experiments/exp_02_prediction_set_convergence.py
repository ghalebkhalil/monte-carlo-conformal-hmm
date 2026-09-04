"""
exp_02_prediction_set_convergence.py

Experiment 2: Prediction-set convergence.

Goal
----
Experiment 1 established that the Monte Carlo estimator

    q_hat_M(x)

converges to the exact permutation quantity

    q(x)

at the usual Monte Carlo rate.

Experiment 2 studies the more consequential question:

    Does the Monte Carlo PREDICTION SET converge to the exact
    conformal prediction set as M increases?

The exact decision for a candidate x is

    include x  <=>  q(x) > alpha.

The Monte Carlo decision is

    include x  <=>  q_hat_M(x) > alpha.

Therefore, candidate-level decision stability depends strongly on

    Delta(x) = |q(x) - alpha|.

Candidates far from alpha should require relatively few Monte Carlo
samples. Candidates close to alpha may require substantially larger M.

The HMM realization is held FIXED throughout the experiment so that
only Monte Carlo permutation randomness is being studied.
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
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.simulation import simulate_binary_experiment

from src.exact_method import exact_prediction_set

from src.monte_carlo_method import monte_carlo_prediction_set


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Same fixed HMM configuration used in Experiment 1.

P = 0.7
B = 0.75

T = 100
T1 = 1

ALPHA = 0.20

N_STATES = 2
N_OBSERVATIONS = 2

DATA_SEED = 20260904
MC_MASTER_SEED = 190847


# We extend M substantially beyond Experiment 1 because the hard
# candidate in this fixed dataset lies very close to alpha.

M_GRID = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
]


# Number of independent Monte Carlo prediction-set constructions
# performed for every M.

N_MC_REPETITIONS = 200


# ---------------------------------------------------------------------
# Set metrics
# ---------------------------------------------------------------------


def jaccard_similarity(
    set_a: set,
    set_b: set,
) -> float:
    """
    Compute Jaccard similarity

        |A intersection B|
        ------------------
           |A union B|

    If both sets are empty, they are identical and the similarity is
    defined as 1.
    """

    union = set_a | set_b

    if len(union) == 0:
        return 1.0

    intersection = (
        set_a & set_b
    )

    return float(
        len(intersection)
        / len(union)
    )


def symmetric_difference_size(
    set_a: set,
    set_b: set,
) -> int:
    """
    Number of candidates appearing in exactly one of the two sets.
    """

    return len(
        set_a.symmetric_difference(
            set_b
        )
    )


# ---------------------------------------------------------------------
# Decision utility
# ---------------------------------------------------------------------


def exact_candidate_decision(
    q_value: float,
    alpha: float,
) -> bool:
    """
    Exact conformal inclusion rule.

        include x iff q(x) > alpha
    """

    return bool(
        q_value > alpha
    )


# ---------------------------------------------------------------------
# Hoeffding decision-error bound
# ---------------------------------------------------------------------


def hoeffding_decision_bound(
    q_value: float,
    alpha: float,
    M: int,
) -> float:
    """
    Hoeffding upper bound on candidate decision disagreement.

    Let

        Delta = |q - alpha|.

    If q != alpha, disagreement between

        1{q_hat_M > alpha}

    and

        1{q > alpha}

    requires a Monte Carlo deviation of at least Delta in the
    relevant direction.

    For Bernoulli Monte Carlo samples, the one-sided Hoeffding bound
    gives

        P(decision disagreement)
            <=
        exp(-2 M Delta^2).

    The bound is clipped to [0, 1].

    When q == alpha, Delta = 0 and the bound is simply 1.
    """

    margin = abs(
        q_value - alpha
    )

    if margin == 0.0:
        return 1.0

    bound = np.exp(
        -2.0
        * M
        * margin**2
    )

    return float(
        min(
            1.0,
            bound,
        )
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:

    # -------------------------------------------------------------
    # Generate the SAME fixed HMM realization as Experiment 1.
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
    print(
        "============================================"
    )
    print(
        "Experiment 2: Prediction-set convergence"
    )
    print(
        "============================================"
    )

    print(
        f"HMM parameters: p={P}, b={B}"
    )

    print(
        f"T={T}, T1={T1}"
    )

    print(
        f"alpha={ALPHA}"
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
    # Exact benchmark
    # -------------------------------------------------------------

    exact_result = exact_prediction_set(
        calibration_states=X_cal,
        calibration_observations=Y_cal,
        future_observations=Y_future,
        alpha=ALPHA,
        n_states=N_STATES,
        n_observations=N_OBSERVATIONS,
        seed=DATA_SEED,
    )

    # -------------------------------------------------------------
    # For this experiment we do NOT want fallback randomness.
    #
    # We want to isolate Monte Carlo approximation error only.
    # -------------------------------------------------------------

    for candidate_result in exact_result.candidate_results:

        if candidate_result.q_value is None:
            raise RuntimeError(
                "Experiment 2 requires a fixed dataset on which "
                "the exact q-value exists for every candidate. "
                f"Candidate {candidate_result.candidate} used "
                f"fallback because "
                f"{candidate_result.fallback_reason}."
            )

    exact_set = set(
        exact_result.prediction_set
    )

    print(
        "Exact prediction set:"
    )

    print(
        exact_result.prediction_set
    )

    print()

    # -------------------------------------------------------------
    # Store exact candidate information.
    # -------------------------------------------------------------

    exact_candidate_info = {}

    print(
        "Exact candidate diagnostics:"
    )

    for result in exact_result.candidate_results:

        q_exact = float(
            result.q_value
        )

        margin = abs(
            q_exact - ALPHA
        )

        exact_decision = (
            q_exact > ALPHA
        )

        exact_candidate_info[
            result.candidate
        ] = {
            "q_exact": q_exact,
            "margin": margin,
            "exact_decision": exact_decision,
            "permutation_count": (
                result.n_permutations
            ),
        }

        print(
            f"    candidate={result.candidate}, "
            f"q_exact={q_exact:.8f}, "
            f"margin=|q-alpha|={margin:.8f}, "
            f"included={exact_decision}, "
            f"|Pi|={result.n_permutations}"
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

    # -------------------------------------------------------------
    # RNG used only to generate independent seeds for MC repetitions.
    # -------------------------------------------------------------

    rng_master = np.random.default_rng(
        MC_MASTER_SEED
    )

    summary_rows = []
    candidate_rows = []

    # =============================================================
    # Vary M
    # =============================================================

    for M in M_GRID:

        exact_set_match_count = 0

        jaccard_values = np.empty(
            N_MC_REPETITIONS,
            dtype=float,
        )

        symmetric_difference_values = np.empty(
            N_MC_REPETITIONS,
            dtype=float,
        )

        # Candidate-specific disagreement counts.

        disagreement_counts = {
            candidate: 0
            for candidate
            in exact_candidate_info
        }

        # Also store q_hat values so that we can see the average
        # Monte Carlo estimate for each candidate.

        q_hat_values = {
            candidate: []
            for candidate
            in exact_candidate_info
        }

        # ---------------------------------------------------------
        # Independent MC repetitions for this M
        # ---------------------------------------------------------

        for repetition in range(
            N_MC_REPETITIONS
        ):

            mc_seed = int(
                rng_master.integers(
                    0,
                    np.iinfo(np.uint32).max,
                )
            )

            mc_result = monte_carlo_prediction_set(
                calibration_states=X_cal,
                calibration_observations=Y_cal,
                future_observations=Y_future,
                alpha=ALPHA,
                n_states=N_STATES,
                n_observations=N_OBSERVATIONS,
                M=M,
                seed=mc_seed,
            )

            # -----------------------------------------------------
            # Exact method was valid for every candidate.
            #
            # Since MC uses identical pre-permutation logic, it
            # should also be valid for every candidate.
            # -----------------------------------------------------

            for candidate_result in mc_result.candidate_results:

                if candidate_result.q_hat is None:
                    raise RuntimeError(
                        "Monte Carlo method unexpectedly used "
                        "fallback for candidate "
                        f"{candidate_result.candidate}: "
                        f"{candidate_result.fallback_reason}"
                    )

            mc_set = set(
                mc_result.prediction_set
            )

            # -----------------------------------------------------
            # Whole prediction-set metrics
            # -----------------------------------------------------

            if mc_set == exact_set:
                exact_set_match_count += 1

            jaccard_values[
                repetition
            ] = jaccard_similarity(
                exact_set,
                mc_set,
            )

            symmetric_difference_values[
                repetition
            ] = symmetric_difference_size(
                exact_set,
                mc_set,
            )

            # -----------------------------------------------------
            # Candidate-level decision disagreement
            # -----------------------------------------------------

            for candidate_result in mc_result.candidate_results:

                candidate = (
                    candidate_result.candidate
                )

                q_hat = float(
                    candidate_result.q_hat
                )

                q_hat_values[
                    candidate
                ].append(
                    q_hat
                )

                mc_decision = bool(
                    candidate_result.included
                )

                exact_decision = bool(
                    exact_candidate_info[
                        candidate
                    ][
                        "exact_decision"
                    ]
                )

                if (
                    mc_decision
                    != exact_decision
                ):
                    disagreement_counts[
                        candidate
                    ] += 1

        # ---------------------------------------------------------
        # Whole-set summaries for this M
        # ---------------------------------------------------------

        exact_set_match_rate = (
            exact_set_match_count
            / N_MC_REPETITIONS
        )

        mean_jaccard = float(
            np.mean(
                jaccard_values
            )
        )

        mean_symmetric_difference = float(
            np.mean(
                symmetric_difference_values
            )
        )

        probability_any_set_error = (
            1.0
            - exact_set_match_rate
        )

        summary_row = {
            "M": M,
            "mc_repetitions": N_MC_REPETITIONS,
            "exact_set_size": len(
                exact_set
            ),
            "exact_set_match_rate": (
                exact_set_match_rate
            ),
            "probability_any_set_error": (
                probability_any_set_error
            ),
            "mean_jaccard_similarity": (
                mean_jaccard
            ),
            "mean_symmetric_difference_size": (
                mean_symmetric_difference
            ),
        }

        summary_rows.append(
            summary_row
        )

        print(
            f"M={M:5d} | "
            f"exact-set match={exact_set_match_rate:.4f} | "
            f"Jaccard={mean_jaccard:.4f} | "
            f"mean sym-diff={mean_symmetric_difference:.4f}"
        )

        # ---------------------------------------------------------
        # Candidate-specific summaries
        # ---------------------------------------------------------

        for candidate, info in exact_candidate_info.items():

            disagreement_rate = (
                disagreement_counts[
                    candidate
                ]
                / N_MC_REPETITIONS
            )

            q_values = np.asarray(
                q_hat_values[
                    candidate
                ],
                dtype=float,
            )

            hoeffding_bound = (
                hoeffding_decision_bound(
                    q_value=info[
                        "q_exact"
                    ],
                    alpha=ALPHA,
                    M=M,
                )
            )

            candidate_row = {
                "candidate": str(
                    candidate
                ),
                "M": M,
                "q_exact": info[
                    "q_exact"
                ],
                "alpha": ALPHA,
                "margin": info[
                    "margin"
                ],
                "exact_included": info[
                    "exact_decision"
                ],
                "permutation_count": info[
                    "permutation_count"
                ],
                "mc_repetitions": (
                    N_MC_REPETITIONS
                ),
                "mean_q_hat": float(
                    np.mean(q_values)
                ),
                "decision_disagreement_rate": (
                    disagreement_rate
                ),
                "hoeffding_decision_bound": (
                    hoeffding_bound
                ),
            }

            candidate_rows.append(
                candidate_row
            )

            print(
                f"          candidate={candidate} | "
                f"margin={info['margin']:.6f} | "
                f"disagreement={disagreement_rate:.4f} | "
                f"Hoeffding bound={hoeffding_bound:.4f}"
            )

        print()

    # =============================================================
    # Save tables
    # =============================================================

    summary_csv = (
        table_directory
        / "exp02_prediction_set_convergence.csv"
    )

    candidate_csv = (
        table_directory
        / "exp02_candidate_disagreement.csv"
    )

    with summary_csv.open(
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

    with candidate_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                candidate_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            candidate_rows
        )

    # =============================================================
    # Figure 1:
    # Whole prediction-set convergence
    # =============================================================

    M_values = np.asarray(
        [
            row["M"]
            for row in summary_rows
        ],
        dtype=float,
    )

    match_rates = np.asarray(
        [
            row[
                "exact_set_match_rate"
            ]
            for row in summary_rows
        ],
        dtype=float,
    )

    jaccard_values_plot = np.asarray(
        [
            row[
                "mean_jaccard_similarity"
            ]
            for row in summary_rows
        ],
        dtype=float,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        M_values,
        match_rates,
        marker="o",
        label="Exact-set match rate",
    )

    plt.plot(
        M_values,
        jaccard_values_plot,
        marker="s",
        label="Mean Jaccard similarity",
    )

    plt.xscale(
        "log"
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.xlabel(
        "Monte Carlo sample size M"
    )

    plt.ylabel(
        "Prediction-set agreement"
    )

    plt.title(
        "Monte Carlo prediction-set convergence"
    )

    plt.legend()

    plt.tight_layout()

    agreement_figure = (
        figure_directory
        / "exp02_set_agreement_vs_M.png"
    )

    plt.savefig(
        agreement_figure,
        dpi=300,
    )

    plt.close()

    # =============================================================
    # Figure 2:
    # Candidate decision disagreement versus M
    # =============================================================

    plt.figure(
        figsize=(7, 5)
    )

    for candidate, info in exact_candidate_info.items():

        rows = [
            row
            for row in candidate_rows
            if row["candidate"]
            == str(candidate)
        ]

        candidate_M = np.asarray(
            [
                row["M"]
                for row in rows
            ],
            dtype=float,
        )

        disagreement = np.asarray(
            [
                row[
                    "decision_disagreement_rate"
                ]
                for row in rows
            ],
            dtype=float,
        )

        bounds = np.asarray(
            [
                row[
                    "hoeffding_decision_bound"
                ]
                for row in rows
            ],
            dtype=float,
        )

        plt.plot(
            candidate_M,
            disagreement,
            marker="o",
            label=(
                f"Empirical disagreement, "
                f"x={candidate}, "
                f"Delta={info['margin']:.4f}"
            ),
        )

        plt.plot(
            candidate_M,
            bounds,
            linestyle="--",
            label=(
                f"Hoeffding bound, "
                f"x={candidate}"
            ),
        )

    plt.xscale(
        "log"
    )

    plt.ylim(
        -0.02,
        1.02,
    )

    plt.xlabel(
        "Monte Carlo sample size M"
    )

    plt.ylabel(
        "Decision disagreement probability"
    )

    plt.title(
        "Candidate decision stability versus distance from threshold"
    )

    plt.legend()

    plt.tight_layout()

    disagreement_figure = (
        figure_directory
        / "exp02_candidate_disagreement_vs_M.png"
    )

    plt.savefig(
        disagreement_figure,
        dpi=300,
    )

    plt.close()

    # =============================================================
    # Final summary
    # =============================================================

    print(
        "--------------------------------------------"
    )

    print(
        "Experiment 2 complete"
    )

    print(
        "--------------------------------------------"
    )

    print()

    print(
        "Exact candidate margins:"
    )

    for candidate, info in exact_candidate_info.items():

        print(
            f"    {candidate}: "
            f"q={info['q_exact']:.8f}, "
            f"|q-alpha|={info['margin']:.8f}"
        )

    print()

    print(
        "Interpretation:"
    )

    print(
        "Candidates farther from alpha should exhibit "
        "much faster decision stabilization."
    )

    print(
        "Candidates very close to alpha may require "
        "substantially larger M even when q_hat itself "
        "is already an accurate estimator of q."
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        summary_csv
    )

    print(
        candidate_csv
    )

    print(
        agreement_figure
    )

    print(
        disagreement_figure
    )


if __name__ == "__main__":
    main()