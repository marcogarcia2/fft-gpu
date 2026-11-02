import pandas as pd
import sys

from arcane import setup_summary


def main():

    # Sets up all paths
    results_csv = setup_summary(sys.argv)
    
    # CSV 
    df = pd.read_csv(results_csv, skipinitialspace=True)

    clock_list = []
    temperature_list = []

    for i in range(df.shape[0]):
        if df['kernel_version'][i] == 'cuFFT':
            kernel_time_in_minutes = df['time_mean_ms'][i] * df['ACC'][i] * (df['RPT'][i]+1) / 1000.0 / 60.0
            kernel_acc_time = df['time_mean_ms'][i] * df['ACC'][i] / 1000.0
            transfer_time_in_minutes = (df['time_h2d_ms'][i] + df['time_d2h_ms'][i]) * (df['RPT'][i]+1) / 1000.0 / 60.0
            total_time_in_minutes = kernel_time_in_minutes + transfer_time_in_minutes
            print("---------------------------------------------------------")
            print(f"{df['kernel_version'][i]} ({df['precision'][i]})\t\t{df['N'][i]} elements")
            print("---------------------------------------------------------")
            print(f"Execution Time:\t\t {total_time_in_minutes:.2f} min")
            print(f"Kernel Window:\t\t {df['window_ms'][i]} ms")
            # print(f"Mean Kernel Time:\t {df['time_mean_ms'][i]} ms")
            # print(f"Setup time:\t\t {df['time_setup_ms'][i]} ms")
            print(f"SM Clock:\t\t {df['sm_mhz_begin'][i]} MHz - {df['sm_mhz_end'][i]} MHz")
            print(f"GPU Temperature:\t {df['temp_begin'][i]}°C - {df['temp_end'][i]}°C")
            # print(f"GFLOPS:\t\t\t {df['gflops'][i]}")
            # print(f"Kernel Mean Energy:\t {df['energy_mean_mj'][i]} mJ")
            # print(f"Energy Median:\t\t {df['energy_median_mj'][i]}")
            # print(f"Energy Std:\t\t {df['energy_std'][i]}")
            # print(f"Energy CI95:\t\t {df['energy_ci95'][i]}")
            print()

            clock_list.append(df['sm_mhz_begin'][i])
            clock_list.append(df['sm_mhz_end'][i])
            temperature_list.append(df['temp_begin'][i])
            temperature_list.append(df['temp_end'][i])

    print("----------------------------------------")
    print(f"Clock Range:\t\t {min(clock_list)} - {max(clock_list)} MHz")
    print(f"Temperature Range:\t {min(temperature_list)} - {max(temperature_list)} °C")
    print("----------------------------------------")

if __name__ == '__main__':
    main()

