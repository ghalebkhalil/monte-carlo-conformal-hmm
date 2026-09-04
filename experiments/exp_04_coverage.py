"""
exp_04_coverage.py

Experiment 4: Coverage of exact, raw Monte Carlo, and
Hoeffding-corrected Monte Carlo conformal prediction.

Goals
-----
Experiments 1--3 established:

    Exp 1:
        q_hat_M converges to exact q.

    Exp 2:
        Monte Carlo prediction sets converge to the exact set.

    Exp 3:
        Monte Carlo avoids the combinatorial exact-enumeration cost.

Experiment 4 asks the central inferential question:

        What happens to empirical coverage?

We compare:

    1. Exact conformal prediction

           C_exact = {x : q(x) > alpha}

    2. Raw Monte Carlo approximation

           C_raw,M = {x : q_hat_M(x) > alpha}

    3. Hoeffding-corrected Monte Carlo approximation

           C_corr,M
               =
           {x : q_hat_M(x) > alpha - epsilon_M}

       where

           epsilon_M
               =
           sqrt(log(2 / delta) / (2M)).

The correction protects against Monte Carlo underestimation of q(x).

Important design choice
-----------------------
For a given dataset, M, and candidate, the SAME q_hat_M is used for
both the raw and corrected prediction sets.

Thus differences between raw and corrected coverage arise ONLY from
the threshold correction and not from different Monte Carlo samples.

We also record prediction-set size because a conservative coverage
correction can increase coverage by enlarging the prediction set.
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

from src.exact_method import (
    candidate_state_sequences,
    exact_prediction_set,
    sequence_is_covered,
    scaled_prediction_set_size,
)

from src.monte_carlo_method import (
    monte_carlo_q_for_candidate,
)


# ---------------------------------------------------------------------
# Global experiment configuration
# ---------------------------------------------------------------------

ALPHA = 0.20

# Concentration failure probability used in the finite-M correction.
DELTA = 0.05

TARGET_COVERAGE = (
    1.0 - ALPHA
)

CORRECTED_COVERAGE_BENCHMARK = (
    1.0
    - ALPHA
    - DELTA
)


# ---------------------------------------------------------------------
# HMM configurations
# ---------------------------------------------------------------------
#
# We use three representative regimes.
#
# All use T=100 and T1=1 so that:
#
#   - exact enumeration remains practical,
#   - coverage can be estimated over many independent datasets,
#   - fallback behavior should remain uncommon.
#
# The three regimes provide qualitatively different dependence and
# observation-information structures.
# ---------------------------------------------------------------------

CONFIGURATIONS = [
    {
        "name": "iid_uninformative",
        "p": 0.5,
        "b": 0.5,
    },
    {
        "name": "moderate",
        "p": 0.7,
        "b": 0.75,
    },
    {
        "name": "strong_markov",
        "p": 0.9,
        "b": 0.75,
    },
]


T = 100
T1 = 1

N_STATES = 2
N_OBSERVATIONS = 2


# ---------------------------------------------------------------------
# Monte Carlo sample-size grid
# ---------------------------------------------------------------------

M_GRID = [
    25,
    50,
    100,
    250,
    500,
    1000,
]


# ---------------------------------------------------------------------
# Number of independent HMM datasets
# ---------------------------------------------------------------------
#
# 300 is a substantial first run.
#
# For final publication-quality numbers, we can later increase this
# to 500 once we have checked that the experiment behaves correctly.
# ---------------------------------------------------------------------

N_REPETITIONS = 300


MASTER_SEED = 20260904


# ---------------------------------------------------------------------
# Hoeffding correction
# ---------------------------------------------------------------------


def hoeffding_epsilon(
    M: int,
    delta: float,
) -> float:
    """
    Compute

        epsilon_M
            =
        sqrt(log(2 / delta) / (2M)).

    This follows from

        P(
            |q_hat_M - q| > epsilon
        )
        <=
        2 exp(-2 M epsilon^2).

    Solving

        2 exp(-2 M epsilon^2) = delta

    gives the radius above.
    """

    if M <= 0:
        raise ValueError(
            "M must be positive."
        )

    if not 0.0 < delta < 1.0:
        raise ValueError(
            "delta must lie strictly between 0 and 1."
        )

    return float(
        np.sqrt(
            np.log(
                2.0 / delta
            )
            / (2.0 * M)
        )
    )


def corrected_threshold(
    alpha: float,
    M: int,
    delta: float,
) -> float:
    """
    Compute the conservative Monte Carlo threshold

        alpha_M
            =
        alpha - epsilon_M.

    We intentionally do NOT truncate the threshold at zero.

    If alpha - epsilon_M < 0, then every valid q_hat >= 0 satisfies

        q_hat > alpha - epsilon_M,

    so the corrected prediction set becomes the full candidate space.

    That conservativeness is part of the finite-M behavior we want
    Experiment 4 to reveal.
    """

    epsilon = hoeffding_epsilon(
        M=M,
        delta=delta,
    )

    return float(
        alpha - epsilon
    )


# ---------------------------------------------------------------------
# Coverage standard error
# ---------------------------------------------------------------------


def binomial_standard_error(
    proportion: float,
    n: int,
) -> float:
    """
    Standard error of an empirical Bernoulli proportion.
    """

    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    return float(
        np.sqrt(
            proportion
            * (1.0 - proportion)
            / n
        )
    )


# ---------------------------------------------------------------------
# Construct raw and corrected MC sets using SAME MC q-values
# ---------------------------------------------------------------------


def monte_carlo_raw_and_corrected_sets(
    calibration_states,
    calibration_observations,
    future_observations,
    alpha: float,
    M: int,
    delta: float,
    n_states: int,
    n_observations: int,
    seed: int,
):
    """
    Construct both the raw and corrected MC prediction sets.

    Crucially, q_hat_M(x) is evaluated only ONCE per candidate.

    The same estimate is then tested against:

        raw:
            q_hat > alpha

        corrected:
            q_hat > alpha - epsilon_M.

    This creates a paired comparison between the two decision rules.

    If a candidate requires the practical fallback, one shared
    fallback random draw is used for BOTH prediction sets. Thus
    fallback randomness cannot itself create a raw/corrected
    difference.
    """

    Y_future = np.asarray(
        future_observations,
        dtype=np.int64,
    )

    horizon = len(
        Y_future
    )

    threshold_corrected = corrected_threshold(
        alpha=alpha,
        M=M,
        delta=delta,
    )

    rng = np.random.default_rng(
        seed
    )

    raw_set = []
    corrected_set = []

    fallback_count = 0
    fallback_reasons = {
        "undefined_hmm_estimate": 0,
        "insufficient_blocks": 0,
    }

    candidate_results = []

    for candidate in candidate_state_sequences(
        n_states=n_states,
        horizon=horizon,
    ):

        candidate_array = np.asarray(
            candidate,
            dtype=np.int64,
        )

        result = monte_carlo_q_for_candidate(
            calibration_states=calibration_states,
            calibration_observations=calibration_observations,
            future_observations=Y_future,
            candidate=candidate_array,
            n_states=n_states,
            n_observations=n_observations,
            M=M,
            rng=rng,
        )

        candidate_tuple = (
            result.candidate
        )

        # ---------------------------------------------------------
        # Standard valid MC case
        # ---------------------------------------------------------

        if result.q_hat is not None:

            q_hat = float(
                result.q_hat
            )

            raw_included = (
                q_hat > alpha
            )

            corrected_included = (
                q_hat
                > threshold_corrected
            )

        # ---------------------------------------------------------
        # Practical fallback
        #
        # Use ONE shared randomized decision so fallback randomness
        # does not create artificial differences between raw and
        # corrected sets.
        # ---------------------------------------------------------

        else:

            fallback_count += 1

            reason = (
                result.fallback_reason
            )

            if reason in fallback_reasons:
                fallback_reasons[
                    reason
                ] += 1

            fallback_included = bool(
                rng.random()
                <= (1.0 - alpha)
            )

            raw_included = (
                fallback_included
            )

            corrected_included = (
                fallback_included
            )

        if raw_included:
            raw_set.append(
                candidate_tuple
            )

        if corrected_included:
            corrected_set.append(
                candidate_tuple
            )

        candidate_results.append(
            result
        )

    return {
        "raw_set": tuple(
            raw_set
        ),
        "corrected_set": tuple(
            corrected_set
        ),
        "corrected_threshold": (
            threshold_corrected
        ),
        "epsilon": hoeffding_epsilon(
            M=M,
            delta=delta,
        ),
        "fallback_count": (
            fallback_count
        ),
        "fallback_reasons": (
            fallback_reasons
        ),
        "candidate_results": tuple(
            candidate_results
        ),
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:

    print()
    print(
        "========================================"
    )
    print(
        "Experiment 4: Coverage"
    )
    print(
        "========================================"
    )

    print(
        f"alpha={ALPHA}"
    )

    print(
        f"Target exact coverage="
        f"{TARGET_COVERAGE:.3f}"
    )

    print(
        f"delta={DELTA}"
    )

    print(
        "Finite-M corrected benchmark "
        f"1-alpha-delta="
        f"{CORRECTED_COVERAGE_BENCHMARK:.3f}"
    )

    print(
        f"T={T}, T1={T1}"
    )

    print(
        f"Repetitions per configuration="
        f"{N_REPETITIONS}"
    )

    print(
        f"M grid={M_GRID}"
    )

    print()

    # -------------------------------------------------------------
    # Show correction magnitude before running simulations.
    # -------------------------------------------------------------

    print(
        "Monte Carlo correction thresholds:"
    )

    for M in M_GRID:

        epsilon = hoeffding_epsilon(
            M=M,
            delta=DELTA,
        )

        threshold = corrected_threshold(
            alpha=ALPHA,
            M=M,
            delta=DELTA,
        )

        print(
            f"    M={M:4d} | "
            f"epsilon={epsilon:.6f} | "
            f"corrected threshold="
            f"{threshold:.6f}"
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

    # Master RNG generates independent data and MC seeds.

    rng_master = np.random.default_rng(
        MASTER_SEED
    )

    all_summary_rows = []

    # =============================================================
    # Configuration loop
    # =============================================================

    for config_index, config in enumerate(
        CONFIGURATIONS,
        start=1,
    ):

        name = config[
            "name"
        ]

        p = float(
            config["p"]
        )

        b = float(
            config["b"]
        )

        print(
            "========================================"
        )

        print(
            f"Configuration "
            f"{config_index}/"
            f"{len(CONFIGURATIONS)}:"
        )

        print(
            f"{name}"
        )

        print(
            f"p={p}, b={b}"
        )

        print(
            "========================================"
        )

        # ---------------------------------------------------------
        # Exact-method accumulators
        # ---------------------------------------------------------

        exact_covered_count = 0

        exact_scaled_size_sum = 0.0

        exact_fallback_candidates = 0

        exact_fallback_undefined = 0

        exact_fallback_blocks = 0

        exact_true_candidate_fallback = 0

        # ---------------------------------------------------------
        # M-specific accumulators
        # ---------------------------------------------------------

        stats = {}

        for M in M_GRID:

            stats[M] = {
                "raw_covered": 0,
                "corrected_covered": 0,
                "raw_scaled_size_sum": 0.0,
                "corrected_scaled_size_sum": 0.0,
                "fallback_candidates": 0,
                "fallback_undefined": 0,
                "fallback_blocks": 0,
                "true_candidate_fallback": 0,
            }

        # =========================================================
        # Independent HMM datasets
        # =========================================================

        for repetition in range(
            N_REPETITIONS
        ):

            # -----------------------------------------------------
            # Independent data seed
            # -----------------------------------------------------

            data_seed = int(
                rng_master.integers(
                    0,
                    np.iinfo(
                        np.uint32
                    ).max,
                )
            )

            (
                X_cal,
                Y_cal,
                X_future,
                Y_future,
            ) = simulate_binary_experiment(
                p=p,
                b=b,
                T=T,
                T1=T1,
                seed=data_seed,
            )

            true_sequence = tuple(
                int(v)
                for v in X_future
            )

            # =====================================================
            # Exact benchmark
            # =====================================================

            exact_seed = int(
                rng_master.integers(
                    0,
                    np.iinfo(
                        np.uint32
                    ).max,
                )
            )

            exact_result = exact_prediction_set(
                calibration_states=X_cal,
                calibration_observations=Y_cal,
                future_observations=Y_future,
                alpha=ALPHA,
                n_states=N_STATES,
                n_observations=N_OBSERVATIONS,
                seed=exact_seed,
            )

            exact_covered = (
                sequence_is_covered(
                    true_sequence=X_future,
                    prediction_set=(
                        exact_result.prediction_set
                    ),
                )
            )

            exact_covered_count += int(
                exact_covered
            )

            exact_scaled_size_sum += (
                scaled_prediction_set_size(
                    prediction_set=(
                        exact_result.prediction_set
                    ),
                    n_states=N_STATES,
                    horizon=T1,
                )
            )

            # -----------------------------------------------------
            # Exact fallback diagnostics
            # -----------------------------------------------------

            for candidate_result in (
                exact_result.candidate_results
            ):

                if candidate_result.used_fallback:

                    exact_fallback_candidates += 1

                    if (
                        candidate_result.fallback_reason
                        == "undefined_hmm_estimate"
                    ):
                        exact_fallback_undefined += 1

                    elif (
                        candidate_result.fallback_reason
                        == "insufficient_blocks"
                    ):
                        exact_fallback_blocks += 1

                    if (
                        candidate_result.candidate
                        == true_sequence
                    ):
                        exact_true_candidate_fallback += 1

            # =====================================================
            # Monte Carlo methods
            # =====================================================

            for M in M_GRID:

                mc_seed = int(
                    rng_master.integers(
                        0,
                        np.iinfo(
                            np.uint32
                        ).max,
                    )
                )

                mc_result = (
                    monte_carlo_raw_and_corrected_sets(
                        calibration_states=X_cal,
                        calibration_observations=Y_cal,
                        future_observations=Y_future,
                        alpha=ALPHA,
                        M=M,
                        delta=DELTA,
                        n_states=N_STATES,
                        n_observations=N_OBSERVATIONS,
                        seed=mc_seed,
                    )
                )

                raw_set = (
                    mc_result[
                        "raw_set"
                    ]
                )

                corrected_set = (
                    mc_result[
                        "corrected_set"
                    ]
                )

                # -------------------------------------------------
                # Coverage
                # -------------------------------------------------

                raw_covered = (
                    true_sequence
                    in raw_set
                )

                corrected_covered = (
                    true_sequence
                    in corrected_set
                )

                stats[M][
                    "raw_covered"
                ] += int(
                    raw_covered
                )

                stats[M][
                    "corrected_covered"
                ] += int(
                    corrected_covered
                )

                # -------------------------------------------------
                # Prediction-set size
                # -------------------------------------------------

                total_candidates = (
                    N_STATES ** T1
                )

                stats[M][
                    "raw_scaled_size_sum"
                ] += (
                    len(raw_set)
                    / total_candidates
                )

                stats[M][
                    "corrected_scaled_size_sum"
                ] += (
                    len(corrected_set)
                    / total_candidates
                )

                # -------------------------------------------------
                # Fallback diagnostics
                # -------------------------------------------------

                stats[M][
                    "fallback_candidates"
                ] += mc_result[
                    "fallback_count"
                ]

                stats[M][
                    "fallback_undefined"
                ] += mc_result[
                    "fallback_reasons"
                ][
                    "undefined_hmm_estimate"
                ]

                stats[M][
                    "fallback_blocks"
                ] += mc_result[
                    "fallback_reasons"
                ][
                    "insufficient_blocks"
                ]

                for candidate_result in (
                    mc_result[
                        "candidate_results"
                    ]
                ):

                    if (
                        candidate_result.used_fallback
                        and candidate_result.candidate
                        == true_sequence
                    ):

                        stats[M][
                            "true_candidate_fallback"
                        ] += 1

            # -----------------------------------------------------
            # Progress
            # -----------------------------------------------------

            if (
                (repetition + 1) % 50
                == 0
            ):

                print(
                    f"    completed "
                    f"{repetition + 1}/"
                    f"{N_REPETITIONS} datasets",
                    flush=True,
                )

        # =========================================================
        # Exact summary
        # =========================================================

        exact_coverage = (
            exact_covered_count
            / N_REPETITIONS
        )

        exact_coverage_se = (
            binomial_standard_error(
                proportion=exact_coverage,
                n=N_REPETITIONS,
            )
        )

        exact_mean_size = (
            exact_scaled_size_sum
            / N_REPETITIONS
        )

        # There are |X|^T1 candidate evaluations per dataset.

        total_candidate_evaluations = (
            N_REPETITIONS
            * (
                N_STATES ** T1
            )
        )

        exact_fallback_rate = (
            exact_fallback_candidates
            / total_candidate_evaluations
        )

        exact_true_fallback_rate = (
            exact_true_candidate_fallback
            / N_REPETITIONS
        )

        print()

        print(
            f"Exact coverage="
            f"{exact_coverage:.4f} "
            f"(SE={exact_coverage_se:.4f})"
        )

        print(
            f"Exact mean scaled set size="
            f"{exact_mean_size:.4f}"
        )

        print(
            f"Exact candidate fallback rate="
            f"{exact_fallback_rate:.4f}"
        )

        print(
            f"Exact TRUE-candidate fallback rate="
            f"{exact_true_fallback_rate:.4f}"
        )

        print()

        # =========================================================
        # M-specific summaries
        # =========================================================

        config_rows = []

        for M in M_GRID:

            raw_coverage = (
                stats[M][
                    "raw_covered"
                ]
                / N_REPETITIONS
            )

            corrected_coverage = (
                stats[M][
                    "corrected_covered"
                ]
                / N_REPETITIONS
            )

            raw_se = (
                binomial_standard_error(
                    proportion=raw_coverage,
                    n=N_REPETITIONS,
                )
            )

            corrected_se = (
                binomial_standard_error(
                    proportion=corrected_coverage,
                    n=N_REPETITIONS,
                )
            )

            raw_mean_size = (
                stats[M][
                    "raw_scaled_size_sum"
                ]
                / N_REPETITIONS
            )

            corrected_mean_size = (
                stats[M][
                    "corrected_scaled_size_sum"
                ]
                / N_REPETITIONS
            )

            fallback_rate = (
                stats[M][
                    "fallback_candidates"
                ]
                / total_candidate_evaluations
            )

            true_fallback_rate = (
                stats[M][
                    "true_candidate_fallback"
                ]
                / N_REPETITIONS
            )

            epsilon = hoeffding_epsilon(
                M=M,
                delta=DELTA,
            )

            threshold = corrected_threshold(
                alpha=ALPHA,
                M=M,
                delta=DELTA,
            )

            row = {
                "configuration": name,
                "p": p,
                "b": b,
                "T": T,
                "T1": T1,
                "alpha": ALPHA,
                "delta": DELTA,
                "M": M,
                "epsilon": epsilon,
                "corrected_threshold": (
                    threshold
                ),
                "repetitions": (
                    N_REPETITIONS
                ),
                "target_coverage": (
                    TARGET_COVERAGE
                ),
                "corrected_coverage_benchmark": (
                    CORRECTED_COVERAGE_BENCHMARK
                ),
                "exact_coverage": (
                    exact_coverage
                ),
                "exact_coverage_se": (
                    exact_coverage_se
                ),
                "raw_mc_coverage": (
                    raw_coverage
                ),
                "raw_mc_coverage_se": (
                    raw_se
                ),
                "corrected_mc_coverage": (
                    corrected_coverage
                ),
                "corrected_mc_coverage_se": (
                    corrected_se
                ),
                "raw_minus_exact_coverage": (
                    raw_coverage
                    - exact_coverage
                ),
                "corrected_minus_exact_coverage": (
                    corrected_coverage
                    - exact_coverage
                ),
                "exact_mean_scaled_set_size": (
                    exact_mean_size
                ),
                "raw_mean_scaled_set_size": (
                    raw_mean_size
                ),
                "corrected_mean_scaled_set_size": (
                    corrected_mean_size
                ),
                "exact_candidate_fallback_rate": (
                    exact_fallback_rate
                ),
                "exact_true_candidate_fallback_rate": (
                    exact_true_fallback_rate
                ),
                "mc_candidate_fallback_rate": (
                    fallback_rate
                ),
                "mc_true_candidate_fallback_rate": (
                    true_fallback_rate
                ),
                "exact_undefined_hmm_count": (
                    exact_fallback_undefined
                ),
                "exact_insufficient_blocks_count": (
                    exact_fallback_blocks
                ),
                "mc_undefined_hmm_count": (
                    stats[M][
                        "fallback_undefined"
                    ]
                ),
                "mc_insufficient_blocks_count": (
                    stats[M][
                        "fallback_blocks"
                    ]
                ),
            }

            config_rows.append(
                row
            )

            all_summary_rows.append(
                row
            )

            print(
                f"M={M:4d} | "
                f"raw coverage="
                f"{raw_coverage:.4f} | "
                f"corrected coverage="
                f"{corrected_coverage:.4f} | "
                f"raw size="
                f"{raw_mean_size:.4f} | "
                f"corrected size="
                f"{corrected_mean_size:.4f}"
            )

        print()

        # =========================================================
        # Coverage figure for this configuration
        # =========================================================

        M_values = np.asarray(
            [
                row["M"]
                for row in config_rows
            ],
            dtype=float,
        )

        raw_coverages = np.asarray(
            [
                row[
                    "raw_mc_coverage"
                ]
                for row in config_rows
            ],
            dtype=float,
        )

        raw_ses = np.asarray(
            [
                row[
                    "raw_mc_coverage_se"
                ]
                for row in config_rows
            ],
            dtype=float,
        )

        corrected_coverages = np.asarray(
            [
                row[
                    "corrected_mc_coverage"
                ]
                for row in config_rows
            ],
            dtype=float,
        )

        corrected_ses = np.asarray(
            [
                row[
                    "corrected_mc_coverage_se"
                ]
                for row in config_rows
            ],
            dtype=float,
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.errorbar(
            M_values,
            raw_coverages,
            yerr=raw_ses,
            marker="o",
            capsize=3,
            label="Raw Monte Carlo",
        )

        plt.errorbar(
            M_values,
            corrected_coverages,
            yerr=corrected_ses,
            marker="s",
            capsize=3,
            label="Hoeffding-corrected Monte Carlo",
        )

        plt.axhline(
            exact_coverage,
            linestyle="--",
            label=(
                f"Exact empirical coverage "
                f"({exact_coverage:.3f})"
            ),
        )

        plt.axhline(
            TARGET_COVERAGE,
            linestyle=":",
            label=(
                f"Target 1-alpha="
                f"{TARGET_COVERAGE:.2f}"
            ),
        )

        plt.axhline(
            CORRECTED_COVERAGE_BENCHMARK,
            linestyle="-.",
            label=(
                f"1-alpha-delta="
                f"{CORRECTED_COVERAGE_BENCHMARK:.2f}"
            ),
        )

        plt.xscale(
            "log"
        )

        plt.ylim(
            0.55,
            1.02,
        )

        plt.xlabel(
            "Monte Carlo sample size M"
        )

        plt.ylabel(
            "Empirical coverage"
        )

        plt.title(
            f"Coverage versus Monte Carlo budget\n"
            f"{name}: p={p}, b={b}"
        )

        plt.legend()

        plt.tight_layout()

        coverage_figure = (
            figure_directory
            / (
                f"exp04_coverage_"
                f"{name}.png"
            )
        )

        plt.savefig(
            coverage_figure,
            dpi=300,
        )

        plt.close()

        # =========================================================
        # Prediction-set size figure
        # =========================================================

        raw_sizes = np.asarray(
            [
                row[
                    "raw_mean_scaled_set_size"
                ]
                for row in config_rows
            ],
            dtype=float,
        )

        corrected_sizes = np.asarray(
            [
                row[
                    "corrected_mean_scaled_set_size"
                ]
                for row in config_rows
            ],
            dtype=float,
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            M_values,
            raw_sizes,
            marker="o",
            label="Raw Monte Carlo",
        )

        plt.plot(
            M_values,
            corrected_sizes,
            marker="s",
            label="Hoeffding-corrected Monte Carlo",
        )

        plt.axhline(
            exact_mean_size,
            linestyle="--",
            label=(
                f"Exact mean size "
                f"({exact_mean_size:.3f})"
            ),
        )

        plt.xscale(
            "log"
        )

        plt.ylim(
            0.0,
            1.02,
        )

        plt.xlabel(
            "Monte Carlo sample size M"
        )

        plt.ylabel(
            "Mean scaled prediction-set size"
        )

        plt.title(
            f"Prediction-set size versus Monte Carlo budget\n"
            f"{name}: p={p}, b={b}"
        )

        plt.legend()

        plt.tight_layout()

        size_figure = (
            figure_directory
            / (
                f"exp04_set_size_"
                f"{name}.png"
            )
        )

        plt.savefig(
            size_figure,
            dpi=300,
        )

        plt.close()

        print(
            f"Coverage figure saved to:"
        )

        print(
            coverage_figure
        )

        print(
            f"Set-size figure saved to:"
        )

        print(
            size_figure
        )

        print()

    # =============================================================
    # Save combined CSV
    # =============================================================

    output_csv = (
        table_directory
        / "exp04_coverage.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_summary_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_summary_rows
        )

    # =============================================================
    # Final summary
    # =============================================================

    print()
    print(
        "========================================"
    )

    print(
        "Experiment 4 complete"
    )

    print(
        "========================================"
    )

    print()

    print(
        f"Target coverage = "
        f"{TARGET_COVERAGE:.3f}"
    )

    print(
        f"Corrected benchmark "
        f"1-alpha-delta = "
        f"{CORRECTED_COVERAGE_BENCHMARK:.3f}"
    )

    print()

    for config in CONFIGURATIONS:

        name = config[
            "name"
        ]

        rows = [
            row
            for row in all_summary_rows
            if row[
                "configuration"
            ]
            == name
        ]

        print(
            name
        )

        print(
            f"    exact coverage="
            f"{rows[0]['exact_coverage']:.4f}"
        )

        for row in rows:

            print(
                f"    M={row['M']:4d}: "
                f"raw="
                f"{row['raw_mc_coverage']:.4f}, "
                f"corrected="
                f"{row['corrected_mc_coverage']:.4f}, "
                f"raw size="
                f"{row['raw_mean_scaled_set_size']:.4f}, "
                f"corrected size="
                f"{row['corrected_mean_scaled_set_size']:.4f}"
            )

        print()

    print(
        "Combined results saved to:"
    )

    print(
        output_csv
    )


if __name__ == "__main__":
    main()