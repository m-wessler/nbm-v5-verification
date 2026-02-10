import os
import sys
from contextlib import contextmanager
from collections import defaultdict
import pygrib
from tqdm import tqdm
import datetime

# Suppress ecCodes warnings
os.environ['ECCODES_LOG'] = 'NONE'

@contextmanager
def suppress_output():
    """Suppress both stdout and stderr to silence warnings from external libraries."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def parse_filename(filename):
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None, None  # Handle unexpected filename formats

    variable_name = parts[0]
    variable_set = parts[1]
    forecast_hour = parts[2].replace("f", "").replace(".grib2", "")  # Remove the 'f' prefix and '.grib2'

    # Ensure forecast_hour is numeric
    if not forecast_hour.isdigit():
        return None, None, None

    return variable_name, variable_set, forecast_hour

def list_files_recursively(directory, start_date=None, end_date=None):
    results = []
    for root, dirs, files in os.walk(directory):
        path_parts = root.split(os.sep)
        if len(path_parts) < 3:
            continue  # Skip directories that don't have enough levels

        init_date = path_parts[-2]  # Subdir level 1
        init_time = path_parts[-1]  # Subdir level 2

        # Filter by start_date and end_date if provided
        if start_date and init_date < start_date:
            continue
        if end_date and init_date > end_date:
            continue

        for file in files:
            variable_name, variable_set, forecast_hour = parse_filename(file)
            if variable_name and variable_set and forecast_hour:
                file_path = os.path.join(root, file)
                results.append((init_date, init_time, forecast_hour, variable_set, variable_name, file_path))

    # Sort results by init_date, init_time, and forecast_hour
    results.sort(key=lambda x: (x[0], x[1], int(x[2])))

    return results

def extract_grib_metadata(file_path):
    metadata = []

    def safe_get(msg, key, default="N/A", is_numeric=False):
        try:
            value = msg[key]
            if is_numeric and isinstance(value, (int, float)):
                if value == 255.00:  # Replace error value with 'N/A'
                    return default
                return f"{value:.2f}"
            elif not is_numeric:
                return str(value)
        except KeyError:
            return default
        except Exception:
            return default
        return default

    try:
        with suppress_output():
            with pygrib.open(file_path) as grb:
                for msg in grb:
                    entry = {
                        "message": str(msg),
                        "parameterName": safe_get(msg, 'parameterName'),
                        "parameterUnits": safe_get(msg, 'parameterUnits'),
                        "lengthOfTimeRange": safe_get(msg, 'lengthOfTimeRange', is_numeric=True),
                        "min": safe_get(msg, 'minimum', is_numeric=True),
                        "max": safe_get(msg, 'maximum', is_numeric=True),
                        "avg": safe_get(msg, 'average', is_numeric=True),
                        "probabilityTypeName": safe_get(msg, 'probabilityTypeName'),
                        "lowerLimit": safe_get(msg, 'lowerLimit', is_numeric=True),
                        "upperLimit": safe_get(msg, 'upperLimit', is_numeric=True),
                        "percentileValue": safe_get(msg, 'percentileValue', is_numeric=True)
                    }
                    metadata.append(entry)
    except Exception as e:
        metadata.append({
            "message": f"Error reading GRIB file {file_path}",
            "parameterName": "N/A",
            "parameterUnits": "N/A",
            "lengthOfTimeRange": "N/A",
            "min": "N/A",
            "max": "N/A",
            "avg": "N/A",
            "probabilityTypeName": "N/A",
            "lowerLimit": "N/A",
            "upperLimit": "N/A",
            "percentileValue": "N/A"
        })

    return metadata

def save_results_to_file(results, output_file):
    with open(output_file, "w") as f:
        # Write header row
        f.write("init_date,init_time,forecast_hour,variable_set,variable_name,grib_message,parameterName,parameterUnits,lengthOfTimeRange,min,max,avg,probabilityTypeName,lowerLimit,upperLimit,percentileValue\n")
        for result in tqdm(results, desc="Processing files"):
            init_date, init_time, forecast_hour, variable_set, variable_name, file_path = result
            grib_metadata = extract_grib_metadata(file_path)
            for meta in grib_metadata:
                f.write(",".join([
                    init_date,
                    init_time,
                    forecast_hour,
                    variable_set,
                    variable_name,
                    meta["message"],
                    meta["parameterName"],
                    meta["parameterUnits"],
                    meta["lengthOfTimeRange"],
                    meta["min"],
                    meta["max"],
                    meta["avg"],
                    meta["probabilityTypeName"],
                    meta["lowerLimit"],
                    meta["upperLimit"],
                    meta["percentileValue"]
                ]) + "\n")

if __name__ == "__main__":
    # Prompt user for start and end dates interactively
    directory = r"N:\data\nbm_para"
    print(f"Cataloging grib files in directory: {directory}")

    start_date = input("Please enter a start date (YYYYMMDD) or press Enter to skip: ").strip()
    end_date = input("Please enter an end date (YYYYMMDD) or press Enter to skip: ").strip()

    # Validate dates or set to None if skipped
    start_date = start_date if start_date else None
    end_date = end_date if end_date else None

    # Example usage
    script_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = list_files_recursively(
        directory=directory,
        start_date=start_date,
        end_date=end_date
    )

    output_file = f"output/catalog_results_{script_start_time}.csv"
    save_results_to_file(results, output_file)

    print(f"Cataloging complete! Results saved to {output_file}")