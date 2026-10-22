import pandas as pd
from pathlib import Path
import os
import sys
import json
import re

BASE_DIR = Path('results')

# Sorts all dirs in BASE_DIR and gets the most recent
def get_most_recent_experiment_folder() -> Path:
    folders = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    folders.sort()
    return folders[-1]


# Gets the folder path given by argv or the most recent one
def get_results_folder(argv: list[str]) -> Path:
    try:
        if (len(argv) == 2):
            results_folder = Path(argv[1])
        else:
            results_folder = get_most_recent_experiment_folder()
    
    except (Exception) as e:
        print(f"Results folder not found: {e}")
        sys.exit(1)
    return results_folder


# Gets roofline json file
def get_roofline_json():
    rl_json_path = BASE_DIR / 'roofline.json'

    try:
        with open(rl_json_path) as f:
            roofline_json = json.load(f)
    
    except (Exception) as e:
        print(f"Error opening Roofline JSON: {e}")
        sys.exit(1)

    return roofline_json


# Returns all needed info for summary.py
def setup_summary(argv: list[str]) -> Path:
    # CSV File
    results_csv = get_results_folder(argv) / "Results.csv"
    return results_csv


# Parses an NCU result filename ("<timestamp>_<kernel>_<N>_<precision>.csv"/".ncu-rep")
def parse_NCU_file_name(filename) -> tuple[str, int, str]:
    stem = Path(filename).stem  # "20261009_184307_FFT_Stockham_4194304_double.csv" -> "...4194304_double"
    m = re.match(r'^\d{8}_\d{6}_(.+)_(\d+)_(float|double)$', stem)
    if not m:
        raise ValueError(f"Filename does not match expected NCU report pattern: {filename}")
    kname = m.group(1)
    Nval = int(m.group(2))
    precision = m.group(3)
    return kname, Nval, precision


#
def get_profile_metrics(results_folder: Path) -> dict:
    results = {}
    csv_files = [f for f in results_folder.iterdir() if f.name.endswith(".csv")]
    csv_files.sort()
    csv_files.pop() # removes "Results.csv"

    unit_to_bytes = {'byte': 1, 'Kbyte': 1e3, 'Mbyte': 1e6, 'Gbyte': 1e9}

    for f in csv_files:
        kname, Nval, precision = parse_NCU_file_name(f)

        # NCU auto-scales the unit per metric (byte/Kbyte/Mbyte/Gbyte) based on
        # magnitude, so it can differ between files and even between the read
        # and write columns of the same file. Read the units row (row 1) to
        # find the actual scale instead of assuming one fixed unit.
        units_row = pd.read_csv(f, nrows=2).iloc[0]
        read_unit = units_row['dram__bytes_read.sum']
        write_unit = units_row['dram__bytes_write.sum']
        if read_unit not in unit_to_bytes or write_unit not in unit_to_bytes:
            raise ValueError(f"Unexpected byte unit in {f}: read={read_unit!r}, write={write_unit!r}")

        df = pd.read_csv(f, skiprows=[1], thousands=',')
        bytes_read  = df['dram__bytes_read.sum'].sum()  * unit_to_bytes[read_unit]
        bytes_write = df['dram__bytes_write.sum'].sum() * unit_to_bytes[write_unit]
        if kname not in results: results[kname] = {}
        if precision not in results[kname]: results[kname][precision] = {}
        results[kname][precision][Nval] = (bytes_read, bytes_write)

    return results


# Returns all needed info for plots.py
def setup_plots(argv: list[str]):

    # Results folder
    results_folder = get_results_folder(argv)

    # CSV File
    results_df = pd.read_csv(results_folder / "Results.csv", skipinitialspace=True)

    # Roofline
    roofline_json = get_roofline_json()

    # Profile statistics (just what we want, CSV too big)
    results_profile = get_profile_metrics(results_folder)

    # Charts output folder
    output_folder = Path("images") / results_folder.name
    os.makedirs(output_folder, exist_ok=True)

    return output_folder, results_df, roofline_json, results_profile