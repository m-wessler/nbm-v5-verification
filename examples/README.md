# NBM Verification System - Examples

This directory contains example scripts and tutorials for using the NBM verification system.

## Available Examples

### basic_usage_demo.py

Demonstrates the core functionality of the verification system:

1. **Metrics Computation**: Shows how to compute continuous (MAE, Bias, RMSE) and categorical (HR, FAR, CSI) metrics from forecast-observation pairs.

2. **Accumulator Pattern**: Demonstrates how to use accumulators for memory-efficient chunked processing.

3. **Regional Accumulation**: Shows how to aggregate statistics across multiple gridpoints within a region.

**Run the demo:**

```bash
python examples/basic_usage_demo.py
```

**Expected output:**
- Computation of verification metrics on sample data
- Demonstration of chunked processing workflow
- Regional aggregation example

## Usage Patterns

### Basic Metrics Computation

```python
from nbm_verification.metrics.continuous import compute_all_continuous_metrics
from nbm_verification.metrics.categorical import (
    compute_contingency_table,
    compute_categorical_metrics,
)

# Compute continuous metrics
metrics = compute_all_continuous_metrics(forecasts, observations)
print(f"MAE: {metrics['mae']}")
print(f"Bias: {metrics['bias']}")
print(f"RMSE: {metrics['rmse']}")

# Compute categorical metrics
threshold = 273.15  # Freezing point
ct = compute_contingency_table(forecasts, observations, threshold)
cat_metrics = compute_categorical_metrics(ct)
print(f"Hit Rate: {cat_metrics['hit_rate']}")
print(f"CSI: {cat_metrics['critical_success_index']}")
```

### Using Accumulators for Chunked Processing

```python
from nbm_verification.accumulation.spatial_accumulator import GridpointAccumulator

# Create accumulator for a gridpoint
acc = GridpointAccumulator(
    grid_i=100, grid_j=200, 
    lat=41.5, lon=-88.0,
    thresholds=[273.15, 283.15]
)

# Process data in chunks
for chunk_forecasts, chunk_obs in data_chunks:
    acc.update(chunk_forecasts, chunk_obs)

# Compute final metrics after all chunks processed
metrics = acc.compute_metrics()
```

### Regional Aggregation

```python
from nbm_verification.accumulation.spatial_accumulator import RegionalAccumulator

# Create accumulator for a region
region_acc = RegionalAccumulator(
    region_id="LOT",
    region_type="CWA",
    region_name="Chicago/Romeoville",
    thresholds=[273.15]
)

# Accumulate data from all gridpoints in the region
for gridpoint_forecasts, gridpoint_obs in region_data:
    region_acc.update(gridpoint_forecasts, gridpoint_obs)

# Compute regional metrics
regional_metrics = region_acc.compute_metrics()
```

## Running a Complete Verification

For a complete verification run using configuration files:

```python
from pathlib import Path
from nbm_verification.workflow.orchestrator import VerificationOrchestrator

# Initialize orchestrator with configuration
orchestrator = VerificationOrchestrator(
    config_path=Path("config/verification_config.yaml")
)

# Run complete verification
orchestrator.run()
```

See `config/verification_config.yaml` for a complete example configuration.

## Next Steps

1. Review the configuration file: `config/verification_config.yaml`
2. Prepare your data sources (NBM, URMA, Synoptic API credentials)
3. Set up regional shapefiles in `config/shapefiles/`
4. Customize configuration for your verification needs
5. Run verification and analyze results

## Additional Resources

- Main README: `../README.md`
- API Documentation: (to be added)
- Full verification workflow: `src/nbm_verification/workflow/orchestrator.py`
