"""
JFWPRB Fire Weather Probability Statistics Calculator

This script processes all JFWPRB GRIB2 files across all dates and initialization times,
calculating MAX, MIN, MEDIAN, and AVERAGE statistics for each variable.

Output: CSV file with statistics for each variable from each file for each timestamp.

Structure: N:/data/nbm_para/{date}/{init_hour}/jfwprb_qmd_f{forecast_hour}.grib2
"""

import os
import sys
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
from pathlib import Path
import pygrib
from datetime import datetime
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('jfwprb_stats_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data directory - adjust as needed
DATA_DIR = Path(r'N:/data/nbm_para/')

# Variable names (16 fire weather probability variables)
VARIABLE_NAMES = [
    "H35S10", "H30S15", "H25S10", "H25S15", "H25S20", 
    "H20S15", "H20S20", "H20S30", "H15S15", "H15S20", 
    "H10S30", "H35G25", "H25G30", "H25G55", "H15G25", "H15G35"
]

# File prefix to search for
FILE_PREFIX = 'jfwprb'

# Output directory
OUTPUT_DIR = Path(r'c:/Users/michael.wessler/Code/nbm-v5-verification/output')

# ==============================================================================
# STATISTICS FUNCTIONS
# ==============================================================================

def calculate_variable_statistics(data: np.ndarray) -> dict:
    """
    Calculate MAX, MIN, MEDIAN, and AVERAGE statistics for a 2D array.
    
    Args:
        data: 2D numpy array of values
        
    Returns:
        Dictionary with max, min, median, mean statistics
    """
    # Handle masked arrays and NaN values
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)
    
    # Flatten and remove NaN values for statistics
    flat_data = data.flatten()
    valid_data = flat_data[~np.isnan(flat_data)]
    
    if len(valid_data) == 0:
        return {
            'max': np.nan,
            'min': np.nan,
            'median': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'nonzero_count': 0,
            'total_cells': len(flat_data)
        }
    
    return {
        'max': float(np.max(valid_data)),
        'min': float(np.min(valid_data)),
        'median': float(np.median(valid_data)),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'nonzero_count': int(np.sum(valid_data > 0)),
        'total_cells': int(len(valid_data))
    }


def process_single_grib_file(file_path: Path, variable_names: list) -> list:
    """
    Process a single GRIB file and extract statistics for each variable.
    
    Args:
        file_path: Path to the GRIB file
        variable_names: List of variable names corresponding to GRIB messages
        
    Returns:
        List of dictionaries containing statistics for each variable
    """
    results = []
    
    try:
        # Extract metadata from filename
        # Format: jfwprb_qmd_f{forecast_hour}.grib2
        fname = file_path.stem
        try:
            forecast_hour = int(fname.split('_f')[-1])
        except (ValueError, IndexError):
            forecast_hour = 0
            logger.warning(f"Could not extract forecast hour from {fname}")
        
        # Extract date and init hour from path
        # Structure: .../date/init_hour/filename
        init_hour = file_path.parent.name
        date_str = file_path.parent.parent.name
        
        # Open GRIB file
        grbs = pygrib.open(str(file_path))
        
        # Process each message (variable)
        for msg_idx, grb in enumerate(grbs):
            if msg_idx >= len(variable_names):
                break
            
            var_name = variable_names[msg_idx]
            
            try:
                # Get data values
                data = grb.values
                
                # Calculate statistics
                stats = calculate_variable_statistics(data)
                
                # Build result record
                result = {
                    'date': date_str,
                    'init_hour': init_hour,
                    'forecast_hour': forecast_hour,
                    'variable': var_name,
                    'file_name': file_path.name,
                    'max': stats['max'],
                    'min': stats['min'],
                    'median': stats['median'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'nonzero_count': stats['nonzero_count'],
                    'total_cells': stats['total_cells']
                }
                
                # Add valid time calculation (if data_date available)
                try:
                    data_date = grb['dataDate']
                    data_time = grb['dataTime']
                    result['data_date'] = data_date
                    result['data_time'] = data_time
                except:
                    pass
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing variable {var_name} in {file_path}: {e}")
                continue
        
        grbs.close()
        
    except Exception as e:
        logger.error(f"Error opening GRIB file {file_path}: {e}")
    
    return results


@dask.delayed
def process_grib_file_delayed(file_path: Path, variable_names: list) -> list:
    """
    Dask-delayed wrapper for processing a single GRIB file.
    """
    return process_single_grib_file(file_path, variable_names)


# ==============================================================================
# DISCOVERY FUNCTIONS
# ==============================================================================

def discover_date_folders(data_dir: Path) -> list:
    """
    Discover all date folders in the data directory.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        List of date folder paths, sorted
    """
    date_folders = []
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return date_folders
    
    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if folder name looks like a date (8 digits: YYYYMMDD)
            if item.name.isdigit() and len(item.name) == 8:
                date_folders.append(item)
    
    return sorted(date_folders)


def discover_init_hour_folders(date_folder: Path) -> list:
    """
    Discover all initialization hour folders within a date folder.
    
    Args:
        date_folder: Path to date folder
        
    Returns:
        List of init hour folder paths, sorted
    """
    init_folders = []
    
    for item in date_folder.iterdir():
        if item.is_dir():
            # Check if folder name is a valid init hour (00, 06, 12, 18, etc.)
            if item.name.isdigit() and len(item.name) == 2:
                init_folders.append(item)
    
    return sorted(init_folders)


def discover_grib_files(init_folder: Path, file_prefix: str = FILE_PREFIX) -> list:
    """
    Discover all GRIB files within an init hour folder.
    
    Args:
        init_folder: Path to init hour folder
        file_prefix: File prefix to filter (default: 'jfwprb')
        
    Returns:
        List of GRIB file paths, sorted by forecast hour
    """
    grib_files = []
    
    for item in init_folder.iterdir():
        if item.is_file() and item.name.startswith(file_prefix):
            if item.suffix in ['.grib2', '.grb2', '.grib', '.grb']:
                grib_files.append(item)
    
    # Sort by forecast hour
    def extract_forecast_hour(f):
        try:
            return int(f.stem.split('_f')[-1])
        except:
            return 0
    
    return sorted(grib_files, key=extract_forecast_hour)


def discover_all_files(data_dir: Path, 
                       date_filter: str = None, 
                       init_filter: str = None) -> dict:
    """
    Discover all GRIB files across all dates and init times.
    
    Args:
        data_dir: Base data directory
        date_filter: Optional specific date to process (YYYYMMDD)
        init_filter: Optional specific init hour to process (00, 06, 12, 18)
        
    Returns:
        Dictionary with structure: {date: {init_hour: [file_paths]}}
    """
    all_files = {}
    total_files = 0
    total_size = 0
    
    date_folders = discover_date_folders(data_dir)
    
    # Apply date filter if specified
    if date_filter:
        date_folders = [d for d in date_folders if d.name == date_filter]
    
    logger.info(f"Found {len(date_folders)} date folders to process")
    
    for date_folder in date_folders:
        date_str = date_folder.name
        all_files[date_str] = {}
        
        init_folders = discover_init_hour_folders(date_folder)
        
        # Apply init filter if specified
        if init_filter:
            init_folders = [i for i in init_folders if i.name == init_filter]
        
        for init_folder in init_folders:
            init_hour = init_folder.name
            grib_files = discover_grib_files(init_folder)
            
            if grib_files:
                all_files[date_str][init_hour] = grib_files
                total_files += len(grib_files)
                total_size += sum(f.stat().st_size for f in grib_files)
    
    logger.info(f"Total files discovered: {total_files}")
    logger.info(f"Total data size: {total_size / 1024**3:.2f} GB")
    
    return all_files


# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================

def process_all_files_sequential(all_files: dict, 
                                  variable_names: list,
                                  progress_callback=None) -> pd.DataFrame:
    """
    Process all GRIB files sequentially (lower memory usage).
    
    Args:
        all_files: Dictionary of files from discover_all_files()
        variable_names: List of variable names
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame with all statistics
    """
    all_results = []
    total_files = sum(
        len(files) 
        for date_files in all_files.values() 
        for files in date_files.values()
    )
    processed = 0
    
    for date_str, init_hours in all_files.items():
        for init_hour, files in init_hours.items():
            logger.info(f"Processing {date_str}/{init_hour}: {len(files)} files")
            
            for file_path in files:
                results = process_single_grib_file(file_path, variable_names)
                all_results.extend(results)
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_files)
                
                if processed % 10 == 0:
                    logger.info(f"Progress: {processed}/{total_files} files ({100*processed/total_files:.1f}%)")
    
    return pd.DataFrame(all_results)


def process_all_files_parallel(all_files: dict, 
                                variable_names: list,
                                n_workers: int = 4,
                                batch_size: int = 20) -> pd.DataFrame:
    """
    Process all GRIB files in parallel using Dask.
    
    Args:
        all_files: Dictionary of files from discover_all_files()
        variable_names: List of variable names
        n_workers: Number of Dask workers
        batch_size: Number of files to process per batch
        
    Returns:
        DataFrame with all statistics
    """
    # Flatten file list
    flat_files = []
    for date_str, init_hours in all_files.items():
        for init_hour, files in init_hours.items():
            flat_files.extend(files)
    
    total_files = len(flat_files)
    logger.info(f"Processing {total_files} files in parallel with {n_workers} workers")
    
    # Set up Dask cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=2,
        memory_limit='2GB'
    )
    client = Client(cluster)
    logger.info(f"Dask Dashboard: {client.dashboard_link}")
    
    all_results = []
    
    try:
        # Process in batches to manage memory
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = flat_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                       f"files {batch_start+1}-{batch_end} of {total_files}")
            
            # Create delayed tasks
            tasks = [
                process_grib_file_delayed(f, variable_names) 
                for f in batch_files
            ]
            
            # Execute batch
            batch_results = dask.compute(*tasks)
            
            # Flatten results
            for file_results in batch_results:
                all_results.extend(file_results)
            
            logger.info(f"Batch complete. Total records: {len(all_results)}")
    
    finally:
        client.close()
        cluster.close()
    
    return pd.DataFrame(all_results)


# ==============================================================================
# OUTPUT FUNCTIONS
# ==============================================================================

def save_results(df: pd.DataFrame, output_dir: Path, output_name: str = None):
    """
    Save results to CSV and summary files.
    
    Args:
        df: Results DataFrame
        output_dir: Output directory path
        output_name: Optional custom output name
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_name is None:
        output_name = f'jfwprb_statistics_{timestamp}'
    
    # Save full results
    csv_path = output_dir / f'{output_name}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved full results to: {csv_path}")
    
    # Create summary by variable
    summary = df.groupby('variable').agg({
        'max': ['max', 'mean'],
        'min': ['min', 'mean'],
        'median': 'mean',
        'mean': 'mean',
        'nonzero_count': 'sum'
    }).round(6)
    
    summary_path = output_dir / f'{output_name}_summary.csv'
    summary.to_csv(summary_path)
    logger.info(f"Saved summary to: {summary_path}")
    
    # Create summary by date/init
    date_init_summary = df.groupby(['date', 'init_hour', 'variable']).agg({
        'max': 'max',
        'min': 'min',
        'median': 'mean',
        'mean': 'mean'
    }).round(6)
    
    date_summary_path = output_dir / f'{output_name}_by_date_init.csv'
    date_init_summary.to_csv(date_summary_path)
    logger.info(f"Saved date/init summary to: {date_summary_path}")
    
    return csv_path, summary_path, date_summary_path


def print_summary(df: pd.DataFrame):
    """
    Print a summary of the results to console.
    """
    print("\n" + "=" * 60)
    print("JFWPRB STATISTICS SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal records: {len(df):,}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Unique init hours: {df['init_hour'].nunique()}")
    print(f"Unique forecast hours: {df['forecast_hour'].nunique()}")
    print(f"Variables: {df['variable'].nunique()}")
    
    print(f"\n{'Variable':<10} {'Max':<12} {'Min':<12} {'Median':<12} {'Mean':<12}")
    print("-" * 60)
    
    for var in VARIABLE_NAMES:
        var_data = df[df['variable'] == var]
        if not var_data.empty:
            print(f"{var:<10} "
                  f"{var_data['max'].max():<12.6f} "
                  f"{var_data['min'].min():<12.6f} "
                  f"{var_data['median'].mean():<12.6f} "
                  f"{var_data['mean'].mean():<12.6f}")
    
    print("=" * 60)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Main entry point for the statistics calculator.
    """
    parser = argparse.ArgumentParser(
        description='Calculate statistics for JFWPRB GRIB2 files'
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=str(DATA_DIR),
        help='Base data directory path'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=str(OUTPUT_DIR),
        help='Output directory for results'
    )
    parser.add_argument(
        '--date', 
        type=str, 
        default=None,
        help='Specific date to process (YYYYMMDD)'
    )
    parser.add_argument(
        '--init-hour', 
        type=str, 
        default=None,
        help='Specific init hour to process (00, 06, 12, 18)'
    )
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Use parallel processing with Dask'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=4,
        help='Number of Dask workers for parallel processing'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=20,
        help='Batch size for parallel processing'
    )
    parser.add_argument(
        '--output-name', 
        type=str, 
        default=None,
        help='Custom output file name (without extension)'
    )
    
    args = parser.parse_args()
    
    # Update paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("JFWPRB STATISTICS CALCULATOR")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Date filter: {args.date or 'All'}")
    logger.info(f"Init hour filter: {args.init_hour or 'All'}")
    logger.info(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
    
    # Discover files
    logger.info("\nDiscovering files...")
    all_files = discover_all_files(
        data_dir, 
        date_filter=args.date, 
        init_filter=args.init_hour
    )
    
    if not any(all_files.values()):
        logger.error("No files found to process!")
        return 1
    
    # Process files
    logger.info("\nProcessing files...")
    start_time = datetime.now()
    
    if args.parallel:
        df = process_all_files_parallel(
            all_files, 
            VARIABLE_NAMES,
            n_workers=args.workers,
            batch_size=args.batch_size
        )
    else:
        df = process_all_files_sequential(all_files, VARIABLE_NAMES)
    
    elapsed = datetime.now() - start_time
    logger.info(f"\nProcessing complete in {elapsed}")
    
    # Save results
    logger.info("\nSaving results...")
    save_results(df, output_dir, args.output_name)
    
    # Print summary
    print_summary(df)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
