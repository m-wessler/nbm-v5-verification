# Configuration Directory

This directory contains configuration files and shapefiles for the NBM verification system.

## Files

### verification_config.yaml

Example verification configuration file demonstrating all available options:

- **Temporal configuration**: Date ranges, initialization hours, forecast hours
- **Spatial configuration**: Domain bounds, chunking strategy, regions, stations
- **Variables**: List of variables to verify with thresholds and metrics
- **Data sources**: Paths to NBM, URMA, and Synoptic data
- **Output**: Output formats and paths
- **Processing**: Parallel processing and memory management
- **QC**: Quality control thresholds
- **Logging**: Logging configuration

### Shapefiles

The `shapefiles/` subdirectories should contain regional boundary shapefiles:

- **cwa/**: County Warning Area shapefiles
- **rfc/**: River Forecast Center shapefiles
- **zones/**: NWS Public Forecast Zone shapefiles

Each shapefile directory should include the standard shapefile components (.shp, .shx, .dbf, .prj).

## Usage

Copy and customize the example configuration for your verification run:

```bash
cp config/verification_config.yaml config/my_verification.yaml
# Edit my_verification.yaml with your settings
```

Run verification with your configuration:

```python
from pathlib import Path
from nbm_verification.workflow.orchestrator import VerificationOrchestrator

orchestrator = VerificationOrchestrator(
    config_path=Path("config/my_verification.yaml")
)
orchestrator.run()
```

## Configuration Options

### Key Sections

1. **verification_run.name**: Unique identifier for the run
2. **temporal**: Date range and time groupings
3. **spatial**: Grid domain and regional configurations
4. **variables**: Variables to verify with their thresholds
5. **data_sources**: Paths to forecast and observation data
6. **output**: Where to save results
7. **processing**: Performance tuning options
8. **qc**: Quality control thresholds
9. **logging**: Logging verbosity and output

See the example configuration file for detailed documentation of each option.
