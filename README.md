# NBM 5.0 Verification

A comprehensive verification framework for evaluating the performance of the National Blend of Models (NBM) version 5.0 against NBM 4.3 and observed weather data.

## Project Overview

This repository contains Python scripts and Jupyter notebooks designed to:

1. **Verify NBM 5.0 performance** against ground truth observations
2. **Benchmark against NBM 4.3** to identify improvements and new issues
3. **Generate comprehensive reports** for field-level trust and buy-in
4. **Replicate and extend** existing NBM 4.x verification tools

## Key Objectives

- Confirm if NBM 5.0 has improved upon known issues in NBM 4.3
- Identify any new systematic issues in NBM 5.0
- Support CONUS-wide verification (OCONUS as reach goal)
- Provide interactive analysis tools and reproducible workflows

## Repository Structure

```
nbm-v5-verification/
├── src/nbm_verification/     # Core Python package
│   ├── io/                   # Data loading and database operations
│   ├── preprocessing/        # Data transformation and pairing
│   ├── metrics/              # Verification metric calculations
│   ├── visualization/        # Plotting and visualization
│   ├── export/               # Data export for Colab
│   ├── config/               # Configuration management
│   └── utils/                # Utility functions
│
├── scripts/                  # Executable scripts
│   ├── data_download/        # Download NBM 4.3, URMA, Synoptic data
│   ├── processing/           # Preprocessing pipelines
│   ├── verification/         # Metric computation workflows
│   ├── export/               # Export for Colab visualization
│   └── reporting/            # Report generation
│
├── notebooks/                # Jupyter notebooks
│   ├── local/                # Local server analysis (primary)
│   └── colab/                # Google Colab visualization (optional)
│
├── tests/                    # Unit tests (pytest)
├── docs/                     # Documentation
└── config/                   # Configuration files
```

## Data Sources

### Forecast Data
- **NBM 5.0**: Local archive (server-based access only)
- **NBM 4.3**: AWS S3 (GRIB2 files)

### Ground Truth Data
- **URMA (gridded)**: AWS S3 (GRIB2 files)
- **Synoptic (point observations)**: API access

## Verification Metrics

### Deterministic Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Error (Bias)

### Categorical Metrics
- Hit Rate (HR)
- False Alarm Ratio (FAR)
- Critical Success Index (CSI)

### Probabilistic Metrics
- Reliability Diagrams
- ROC Curves / AUC
- Brier Score (BS) and Brier Skill Score (BSS)
- Continuous Ranked Probability Score (CRPS) and CRPSS

### Temporal Coverage
All metrics support analysis at 6, 12, 24, 48, and 72-hour periods.

## Installation

### Option 1: Pip Installation

```bash
# Clone the repository
git clone https://github.com/m-wessler/nbm-v5-verification.git
cd nbm-v5-verification

# Install the package
pip install -e .
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate nbm-verification

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Configure Paths

Copy example configuration files and edit with your settings:

```bash
cp config/config.example.yaml config/config.yaml
cp config/paths.example.yaml config/paths.yaml
cp config/aws_credentials.example config/aws_credentials.yaml
```

### 2. Download Data (NBM 4.3 and URMA)

```bash
python scripts/data_download/fetch_nbm43.py --date 2024-01-01
python scripts/data_download/fetch_urma.py --date 2024-01-01
```

### 3. Process Data

```bash
# Process NBM 5.0 from local archive
python scripts/processing/preprocess_nbm50.py --date 2024-01-01

# Process NBM 4.3
python scripts/processing/preprocess_nbm43.py --date 2024-01-01

# Create forecast-observation pairs
python scripts/processing/create_pairs.py --date 2024-01-01
```

### 4. Run Verification

```bash
# Run full verification suite
python scripts/verification/run_full_suite.py --date 2024-01-01
```

### 5. Analyze Results

Open and run notebooks in `notebooks/local/` for interactive analysis.

## Workflow

### Local Server Workflow (Primary)

1. **Data Acquisition**: Download NBM 4.3, URMA, and Synoptic data
2. **Data Processing**: Preprocess NBM 5.0/4.3 and create forecast-obs pairs
3. **Verification**: Compute all metrics
4. **Analysis**: Use local notebooks for detailed analysis
5. **Export** (optional): Export metrics database for Colab visualization

### Google Colab Workflow (Optional)

6. **Upload**: Upload compressed metrics database to Google Drive
7. **Visualize**: Use Colab notebooks for interactive visualization
8. **Report**: Generate reports in Colab environment

## Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Server Setup](docs/server_setup.md)
- [Data Sources](docs/data_sources.md)
- [Metrics Guide](docs/metrics_guide.md)
- [Colab Visualization](docs/colab_visualization.md)
- [Technical Guide](docs/technical_guide.md)

## Testing

Run the test suite using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nbm_verification

# Run specific test module
pytest tests/test_metrics/test_deterministic.py
```

## Technology Stack

- **Data I/O**: xarray, cfgrib, pandas, boto3, s3fs, requests
- **Processing**: geopandas, MetPy
- **Metrics**: metpy.calc, scikit-learn, scoringrules
- **Visualization**: matplotlib, Cartopy, seaborn, metpy.plots
- **Testing**: pytest

## Project Timeline

- **Initial Findings Report**: TBD
- **Final Verification Report**: TBD

## Contributing

This is an internal project. For questions or issues, please contact the project team.

## License

TBD

## Authors

- Mike Wessler (@m-wessler)

## Acknowledgments

- National Weather Service
- Meteorological Development Laboratory (MDL)
- NBM Development Team

---

**Last Updated**: 2025-11-13