#!/usr/bin/env python
"""
Example script demonstrating the NBM verification system.

This script shows how to use the verification system components
to compute metrics from forecast and observation data.
"""

import numpy as np
from pathlib import Path
from datetime import datetime

# Import verification system components
from nbm_verification.accumulation.spatial_accumulator import (
    GridpointAccumulator,
    RegionalAccumulator,
)
from nbm_verification.metrics.calculator import MetricsCalculator
from nbm_verification.metrics.continuous import compute_all_continuous_metrics
from nbm_verification.metrics.categorical import (
    compute_contingency_table,
    compute_categorical_metrics,
)
from nbm_verification.utils.logging import setup_logging


def demonstrate_metrics_computation():
    """Demonstrate basic metrics computation."""
    print("=" * 80)
    print("NBM Verification System - Metrics Computation Demonstration")
    print("=" * 80)
    print()

    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting metrics computation demonstration")

    # Create sample forecast and observation data
    print("1. Creating sample forecast and observation data...")
    np.random.seed(42)

    # Simulate 100 forecast-observation pairs for temperature (in Kelvin)
    true_temps = np.random.normal(280.0, 10.0, 100)  # Mean 280K, std 10K
    forecast_temps = true_temps + np.random.normal(0.0, 2.0, 100)  # Add forecast error

    print(f"   - Generated {len(true_temps)} forecast-observation pairs")
    print(f"   - Mean observed temperature: {np.mean(true_temps):.2f} K")
    print(f"   - Mean forecast temperature: {np.mean(forecast_temps):.2f} K")
    print()

    # Compute continuous metrics
    print("2. Computing continuous metrics (MAE, Bias, RMSE)...")
    continuous_metrics = compute_all_continuous_metrics(forecast_temps, true_temps)

    print(f"   - MAE:        {continuous_metrics['mae']:.3f} K")
    print(f"   - Bias:       {continuous_metrics['bias']:.3f} K")
    print(f"   - RMSE:       {continuous_metrics['rmse']:.3f} K")
    print(f"   - Bias Ratio: {continuous_metrics['bias_ratio']:.3f}")
    print()

    # Compute categorical metrics
    print("3. Computing categorical metrics for freezing threshold (273.15 K)...")
    threshold = 273.15
    contingency_table = compute_contingency_table(forecast_temps, true_temps, threshold)
    categorical_metrics = compute_categorical_metrics(contingency_table)

    print(f"   - Hits:              {contingency_table['hits']}")
    print(f"   - Misses:            {contingency_table['misses']}")
    print(f"   - False Alarms:      {contingency_table['false_alarms']}")
    print(f"   - Correct Negatives: {contingency_table['correct_negatives']}")
    print(f"   - Hit Rate (HR):     {categorical_metrics['hit_rate']:.3f}")
    print(f"   - False Alarm Ratio: {categorical_metrics['false_alarm_ratio']:.3f}")
    print(f"   - CSI:               {categorical_metrics['critical_success_index']:.3f}")
    print()


def demonstrate_accumulator_pattern():
    """Demonstrate the accumulator pattern for chunked processing."""
    print("=" * 80)
    print("NBM Verification System - Accumulator Pattern Demonstration")
    print("=" * 80)
    print()

    print("4. Demonstrating accumulator pattern for chunked processing...")
    print()

    # Create a gridpoint accumulator
    accumulator = GridpointAccumulator(
        grid_i=100,
        grid_j=200,
        lat=41.5,
        lon=-88.0,
        thresholds=[273.15, 283.15],  # Freezing and 10°C
    )

    print("   - Created GridpointAccumulator for point (100, 200) at (41.5°N, 88.0°W)")
    print()

    # Simulate processing 3 chunks of data
    print("5. Processing 3 chunks of data...")
    np.random.seed(42)

    for chunk_id in range(3):
        # Generate random data for this chunk
        chunk_forecasts = np.random.normal(280.0, 5.0, 20)
        chunk_obs = chunk_forecasts + np.random.normal(0.0, 2.0, 20)

        # Update accumulator
        accumulator.update(chunk_forecasts, chunk_obs)

        print(f"   - Chunk {chunk_id + 1}: Added 20 samples, total now {accumulator.n_samples}")

    print()

    # Compute final metrics from accumulator
    print("6. Computing final metrics from accumulated statistics...")
    metrics = accumulator.compute_metrics()

    print(f"   - Total samples:  {metrics['n_samples']}")
    print(f"   - MAE:           {metrics['mae']:.3f} K")
    print(f"   - Bias:          {metrics['bias']:.3f} K")
    print(f"   - RMSE:          {metrics['rmse']:.3f} K")
    print()

    # Show categorical metrics for each threshold
    print("7. Categorical metrics by threshold:")
    for threshold, cat_metrics in metrics['categorical'].items():
        print(f"   - Threshold {threshold} K:")
        print(f"     * Hit Rate: {cat_metrics['hit_rate']:.3f}")
        print(f"     * FAR:      {cat_metrics['false_alarm_ratio']:.3f}")
        print(f"     * CSI:      {cat_metrics['critical_success_index']:.3f}")
    print()


def demonstrate_regional_accumulator():
    """Demonstrate regional accumulation across gridpoints."""
    print("=" * 80)
    print("NBM Verification System - Regional Accumulator Demonstration")
    print("=" * 80)
    print()

    print("8. Creating regional accumulator for a County Warning Area...")
    print()

    # Create a regional accumulator
    regional_acc = RegionalAccumulator(
        region_id="LOT",
        region_type="CWA",
        region_name="Chicago/Romeoville",
        thresholds=[273.15],
    )

    print(f"   - Region: {regional_acc.region_name} ({regional_acc.region_id})")
    print()

    # Simulate accumulating data from 10 gridpoints in the region
    print("9. Accumulating data from 10 gridpoints in the region...")
    np.random.seed(42)

    for gridpoint_id in range(10):
        # Each gridpoint has 20 forecast-obs pairs
        fcst = np.random.normal(280.0, 5.0, 20)
        obs = fcst + np.random.normal(0.0, 2.0, 20)

        regional_acc.update(fcst, obs)

    print(f"   - Total samples accumulated: {regional_acc.n_samples}")
    print(f"   - (From 10 gridpoints × 20 samples each)")
    print()

    # Compute regional metrics
    print("10. Computing regional metrics...")
    regional_metrics = regional_acc.compute_metrics()

    print(f"    - Region: {regional_metrics['region_name']}")
    print(f"    - MAE:   {regional_metrics['mae']:.3f} K")
    print(f"    - Bias:  {regional_metrics['bias']:.3f} K")
    print(f"    - RMSE:  {regional_metrics['rmse']:.3f} K")
    print()


def main():
    """Run all demonstrations."""
    print()
    print("#" * 80)
    print("# NBM VERIFICATION SYSTEM DEMONSTRATION")
    print("#" * 80)
    print()

    try:
        # Run demonstrations
        demonstrate_metrics_computation()
        print()

        demonstrate_accumulator_pattern()
        print()

        demonstrate_regional_accumulator()
        print()

        print("=" * 80)
        print("All demonstrations completed successfully!")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
