"""
exp_00_reproduce_original.py

Baseline validation experiment.

Goal
----
Before evaluating our Monte Carlo approximation, verify that our clean
implementation of the exact Nettasinghe et al. (2023) procedure exhibits
the empirical behavior reported in the original paper.

We begin with T1 = 1 because exhaustive enumeration is inexpensive
enough to run repeatedly.

For each configuration (p, b, T), we simulate independent binary HMM
paths, construct the exact conformal prediction set, and estimate:

    1. empirical coverage,
    2. average scaled prediction-set size,
    3. frequency with which the implementation uses the fallback.

The original paper uses alpha = 0.2, so the desired coverage is 0.8.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


# ---------------------------------------------------------------------
# Allow imports from project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.simulation import simulate_binary_experiment

from src.exact_method import (
    exact_prediction_set,
    scaled_prediction_set_size,
    sequence_is_covered,
)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

ALPHA = 0.20

P_GRID = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
]

B_GRID = [
    0.5,
    0.75,
    0.9,
]

T_GRID = [
    50,
    100,
    200,
]

# Start with one-step prediction.
T1 = 1

# The paper uses 500.
#
# We intentionally start smaller as a validation run.
N_REPETITIONS = 100

MASTER_SEED = 20260904

N_STATES = 2
N_OBSERVATIONS = 2


# ---------------------------------------------------------------------
# One parameter configuration
# ---------------------------------------------------------------------


def run_configuration(
    p: float,
    b: float,
    T: int,
    T1: int,
    n_repetitions: int,
    rng: np.random.Generator,
) -> dict:
    """
    Run repeated independent simulations for one HMM configuration.
    """

    covered_count = 0

    scaled_sizes = []

    fallback_candidate_count = 0
    total_candidate_count = 0

    permutation_counts = []

    start = perf_counter()

    for repetition in range(n_repetitions):

        # Independent seed for this simulated path.
        data_seed = int(
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
            )
        )

        # Separate seed for the rare randomized fallback.
        fallback_seed = int(
            rng.integers(
                0,
                np.iinfo(np.uint32).max,
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

        result = exact_prediction_set(
            calibration_states=X_cal,
            calibration_observations=Y_cal,
            future_observations=Y_future,
            alpha=ALPHA,
            n_states=N_STATES,
            n_observations=N_OBSERVATIONS,
            seed=fallback_seed,
        )

        covered = sequence_is_covered(
            true_sequence=X_future,
            prediction_set=result.prediction_set,
        )

        covered_count += int(covered)

        scaled_sizes.append(
            scaled_prediction_set_size(
                prediction_set=result.prediction_set,
                n_states=N_STATES,
                horizon=T1,
            )
        )

        for candidate_result in result.candidate_results:

            total_candidate_count += 1

            fallback_candidate_count += int(
                candidate_result.used_fallback
            )

            if not candidate_result.used_fallback:
                permutation_counts.append(
                    candidate_result.n_permutations
                )

    elapsed = perf_counter() - start

    empirical_coverage = (
        covered_count
        / n_repetitions
    )

    mean_scaled_size = float(
        np.mean(scaled_sizes)
    )

    fallback_rate = (
        fallback_candidate_count
        / total_candidate_count
    )

    if permutation_counts:

        mean_permutations = float(
            np.mean(permutation_counts)
        )

        max_permutations = int(
            np.max(permutation_counts)
        )

    else:

        mean_permutations = float("nan")
        max_permutations = 0

    return {
        "p": p,
        "b": b,
        "T": T,
        "T1": T1,
        "alpha": ALPHA,
        "target_coverage": 1.0 - ALPHA,
        "repetitions": n_repetitions,
        "empirical_coverage": empirical_coverage,
        "mean_scaled_set_size": mean_scaled_size,
        "fallback_rate": fallback_rate,
        "mean_permutation_count": mean_permutations,
        "max_permutation_count": max_permutations,
        "runtime_seconds": elapsed,
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:

    rng = np.random.default_rng(
        MASTER_SEED
    )

    results = []

    total_configurations = (
        len(P_GRID)
        * len(B_GRID)
        * len(T_GRID)
    )

    configuration_number = 0

    print(
        "\nExact-method baseline validation"
    )

    print(
        f"Target coverage = {1.0 - ALPHA:.2f}"
    )

    print(
        f"T1 = {T1}"
    )

    print(
        f"Repetitions per configuration = "
        f"{N_REPETITIONS}"
    )

    print(
        f"Total configurations = "
        f"{total_configurations}"
    )

    print()

    for T in T_GRID:

        for b in B_GRID:

            for p in P_GRID:

                configuration_number += 1

                print(
                    f"[{configuration_number}/"
                    f"{total_configurations}] "
                    f"T={T}, p={p:.1f}, b={b:.2f}",
                    flush=True,
                )

                result = run_configuration(
                    p=p,
                    b=b,
                    T=T,
                    T1=T1,
                    n_repetitions=N_REPETITIONS,
                    rng=rng,
                )

                results.append(
                    result
                )

                print(
                    "    "
                    f"coverage="
                    f"{result['empirical_coverage']:.3f}, "
                    f"scaled_size="
                    f"{result['mean_scaled_set_size']:.3f}, "
                    f"fallback="
                    f"{result['fallback_rate']:.3f}, "
                    f"time="
                    f"{result['runtime_seconds']:.2f}s"
                )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    output_directory = (
        PROJECT_ROOT
        / "results"
        / "tables"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / "exp00_original_baseline.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    coverages = np.array(
        [
            row["empirical_coverage"]
            for row in results
        ]
    )

    scaled_sizes = np.array(
        [
            row["mean_scaled_set_size"]
            for row in results
        ]
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "Baseline validation complete"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Target coverage: "
        f"{1.0 - ALPHA:.3f}"
    )

    print(
        f"Mean empirical coverage: "
        f"{coverages.mean():.3f}"
    )

    print(
        f"Minimum empirical coverage: "
        f"{coverages.min():.3f}"
    )

    print(
        f"Maximum empirical coverage: "
        f"{coverages.max():.3f}"
    )

    print(
        f"Mean scaled prediction-set size: "
        f"{scaled_sizes.mean():.3f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()