import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
import sys
import os

RESULTS_FOLDER = Path('results')
ROOFLINE_JSON = RESULTS_FOLDER / 'roofline.json'
RESULTS_CSV = None

def get_most_recent_results() -> Path:
    csv_files = [f for f in os.listdir(RESULTS_FOLDER) if f.endswith(".csv")]
    csv_files.sort()
    return RESULTS_FOLDER / csv_files[-1]


def main():

    if len(sys.argv) == 2:
        RESULTS_CSV = Path(sys.argv[1])
    else:
        RESULTS_CSV = get_most_recent_results()

    roof = None
    with open(ROOFLINE_JSON) as f:
        roof = json.load(f)

    df = pd.read_csv(RESULTS_CSV)

    for i in range(df.shape[0]):
        kernel_time_in_minutes = df['time_mean_ms'][i] * df['ACC'][i] * (df['RPT'][i]+1) / 1000.0 / 60.0
        transfer_time_in_minutes = (df['time_h2d_ms'][i] + df['time_d2h_ms'][i]) * (df['RPT'][i]+1) / 1000.0 / 60.0
        total_time_in_minutes = kernel_time_in_minutes + transfer_time_in_minutes
        print("---------------------------------------------------------")
        print(f"{df['kernel_version'][i]} ({df['precision'][i]})\t\t{df['N'][i]} elements")
        print("---------------------------------------------------------")
        print(f"Execution Time:\t\t {total_time_in_minutes:.2f} min")
        print(f"Num. ACC:\t\t {df['ACC'][i]}")
        print(f"Mean Kernel Time:\t {df['time_mean_ms'][i]} ms")
        print(f"Mean H2D Time:\t\t {df['time_h2d_ms'][i]} ms")
        print(f"Mean D2H Time:\t\t {df['time_d2h_ms'][i]} ms")
        print(f"Energy Mean:\t\t {df['energy_mean_mj'][i]} mJ")
        print(f"GPU Temperature:\t {df['temp_begin'][i]}°C - {df['temp_end'][i]}°C")
        print(f"GFLOPS:\t\t\t {df['gflops'][i]}")
        # print(f"Energy-Delay Product:\t {df['edp'][i]} Js")
        print()


if __name__ == '__main__':
    main()

