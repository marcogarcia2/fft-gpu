from pathlib import Path
import os
import sys
import json

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
def setup_summary(argv: list[str]) -> None:
    
    # CSV File
    results_csv = get_results_folder(argv) / "Results.csv"
    # ...

    return results_csv


# Returns all needed info for plots.py
def setup_plots(argv: list[str]):

    # Results folder
    results_folder = get_results_folder(argv)

    # CSV File
    results_csv = results_folder / "Results.csv"

    # Roofline
    roofline_json = get_roofline_json()

    # Profile (not yet implemented)
    results_profile = None

    # Charts output folder
    output_folder = Path("images") / results_folder.name
    os.makedirs(output_folder, exist_ok=True)

    return output_folder, results_csv, roofline_json, results_profile