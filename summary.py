import pandas as pd
import sys

from arcane import setup_summary


def main():

    # Sets up all paths
    results_csv = setup_summary(sys.argv)
    
    # CSV 
    df = pd.read_csv(results_csv, skipinitialspace=True)

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
        print(f"Setup time:\t\t {df['time_setup_ms'][i]} ms")
        print(f"SM Clock:\t\t {df['sm_mhz_begin'][i]} MHz - {df['sm_mhz_end'][i]} MHz")
        print(f"Kernel Time Stddev:\t {df['time_std'][i]} ms")
        print(f"Mean H2D Time:\t\t {df['time_h2d_ms'][i]} ms")
        print(f"Mean D2H Time:\t\t {df['time_d2h_ms'][i]} ms")
        print(f"Energy Mean:\t\t {df['energy_mean_mj'][i]} mJ")
        print(f"Energy Stddev:\t\t {df['energy_std'][i]} mJ")
        print(f"GPU Temperature:\t {df['temp_begin'][i]}°C - {df['temp_end'][i]}°C")
        print(f"GFLOPS:\t\t\t {df['gflops'][i]}")
        print(f"Energy-Delay Product:\t {df['edp'][i]} Js")
        print()


if __name__ == '__main__':
    main()

