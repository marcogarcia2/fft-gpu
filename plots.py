import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import math

from arcane import setup_plots

PALETTE = { # OKABE_ITO PALETTE
    'cuFFT':                   '#009E73',
    'FFT_Naive':               '#0072B2',
    'FFT_Tabled':        '#F0E442',
    'FFT_Stockham':     '#D55E00',
    'FFT_Shared':              '#E69F00',
    'FFT_Shared_Pure':           '#CC79A7',
    'FFT_Stockham_Pure':           '#56B4E9',
}
DEFAULT_COLOR = '#000000' # used for any kernel not listed above

PRECISION_STYLE = {
    'float':  dict(marker='o', linestyle='-'),
    'double': dict(marker='s', linestyle='--'),
}

def apply_academic_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
    })

def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, which='major', alpha=0.25, linewidth=0.6)

def pow2_ticks(ax, ns):
    ax.set_xticks(ns)
    ax.set_xticklabels([rf'$2^{{{int(np.log2(n))}}}$' for n in ns])
    ax.minorticks_off()


# Roofline Model: the most important one
def plot_0(out_folder: Path, df: pd.DataFrame, roofline_json: dict, results_profile: dict, FP32: bool=False, FP64: bool=False):

    if (not FP32 and not FP64): return
    if (FP32 and FP64): return

    # Assembling the results with data
    results = {}
    for i in range(df.shape[0]):
        kernel_name = df['kernel_version'][i]
        precision   = df['precision'][i]
        N_val       = df['N'][i]
        gflops      = df['gflops'][i]

        if (precision == 'float'  and FP32 == False): continue
        if (precision == 'double' and FP64 == False): continue

        # Filters N
        if (N_val != 1 << 24): continue

        read_write_bytes = results_profile[kernel_name][precision][N_val]
        sum_bytes = read_write_bytes[0] + read_write_bytes[1]
        arithmetic_intensity = 5 * N_val * math.log(N_val, 2) / sum_bytes

        if kernel_name not in results: results[kernel_name] = {}
        if precision not in results[kernel_name]: results[kernel_name][precision] = {}
        results[kernel_name][precision][N_val] = (arithmetic_intensity, gflops)

    fig, ax = plt.subplots(figsize=(6, 3.5)) # 

    # Plot
    # For each Kernel
    for kernel, _ in results.items():
        ## For each precision
        for pr, data in _.items():
            n = []
            ar_in = []
            gfl = []
            for N_val, t in data.items():
                n.append(N_val)
                ar_in.append(t[0])
                gfl.append(t[1])

            prec_name = f" ({pr})" if FP32 and FP64 is True else ""
            label = kernel + prec_name
            marker = 'o' if FP32 else 's'
            ax.plot(ar_in, gfl, color=PALETTE.get(kernel, DEFAULT_COLOR),
                        marker=marker,
                        linestyle='none',
                        label=label)

    # Plot Rooflines
    peak_bw = roofline_json['peak_bandwidth_gbps']
    peak_gflops = roofline_json['peak_gflops_fp32'] if FP32 else roofline_json['peak_gflops_fp64']
    x = np.logspace(-1, 4, 200)                       # 0.1 a 100 FLOP/byte
    roof = np.minimum(peak_bw * x, peak_gflops)
    ax.plot(x, roof, color='black', linewidth=1.5)

    ridge = peak_gflops / peak_bw                     # ponto de inflexão
    ax.axvline(ridge, color='gray', linestyle=':', linewidth=0.8)

    # Style 
    style_axes(ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Desempenho (GFLOP/s)')
    ax.set_xlabel('Intensidade aritmética (FLOP/byte)')
    ax.legend(frameon=False, loc='upper left')

    # Saving
    prec_name = ""
    if (FP32): prec_name = "_FP32" 
    else     : prec_name = "_FP64" 
    file_name = "plot_00" + prec_name + ".pdf"
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()

# Mean-time (ms) X vector size
def plot_1(out_folder: Path, df: pd.DataFrame, FP32: bool=True, FP64: bool=True):

    if (not FP32 and not FP64): return # both can't be false at the same time

    # Assembling the results dictionary with data
    results = {}
    for i in range(df.shape[0]):
        kernel_name  = df['kernel_version'][i]
        precision    = df['precision'][i]
        N_val        = df['N'][i]
        time_mean_ms = df['time_mean_ms'][i]
        time_ci95    = df['time_ci95'][i]

        # Filtering
        if (precision == 'float'  and FP32 is False): continue
        if (precision == 'double' and FP64 is False): continue

        # Appending results
        if kernel_name not in results: results[kernel_name] = {}
        if precision not in results[kernel_name]: results[kernel_name][precision] = {}
        results[kernel_name][precision][N_val] = (time_mean_ms, time_ci95)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.5)) # final size in inches

    # For each Kernel
    for kernel, _ in results.items():
        ## For each precision
        for pr, data in _.items():
            n = []
            t_mean = []
            t_ci95 = []
            ### For each value of N there are t_mean +- t_ci95
            for N_val, t in data.items():
                n.append(N_val)
                t_mean.append(t[0])
                t_ci95.append(t[1])

            prec_name = f" ({pr})" if FP32 and FP64 is True else ""
            label = kernel + prec_name
            ax.errorbar(n, t_mean, yerr=t_ci95, capsize=3,
                        color=PALETTE.get(kernel, DEFAULT_COLOR),
                        **PRECISION_STYLE[pr],
                        label=label)

    # Style 
    style_axes(ax)
    ax.set_xscale('log', base=2)
    pow2_ticks(ax, sorted(df['N'].unique()))
    ax.set_yscale('log')
    ax.set_ylabel('Tempo médio por execução (ms)')
    ax.set_xlabel('Tamanho do vetor N')
    ax.legend(frameon=False, loc='upper left')

    # Saving
    prec_name = ""
    if (FP32 and not FP64): prec_name = "_FP32"
    if (FP64 and not FP32): prec_name = "_FP64"
    file_name = "plot_01" + prec_name + ".pdf"
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()

# GFLOPS X vector size
def plot_1_gflops(out_folder: Path, df: pd.DataFrame, FP32: bool=True, FP64: bool=True):

    if (not FP32 and not FP64): return # both can't be false at the same time

    # Assembling the results dictionary with data
    results = {}
    for i in range(df.shape[0]):
        kernel_name  = df['kernel_version'][i]
        precision    = df['precision'][i]
        N_val        = df['N'][i]
        gflops       = df['gflops'][i]

        # Filtering
        if (precision == 'float'  and FP32 is False): continue
        if (precision == 'double' and FP64 is False): continue

        # Appending results
        if kernel_name not in results: results[kernel_name] = {}
        if precision not in results[kernel_name]: results[kernel_name][precision] = {}
        results[kernel_name][precision][N_val] = (gflops)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.5)) # final size in inches

    # For each Kernel
    for kernel, _ in results.items():
        ## For each precision
        for pr, data in _.items():
            n = []
            gflops = []
            ### For each value of N there are t_mean +- t_ci95
            for N_val, gfl in data.items():
                n.append(N_val)
                gflops.append(gfl)

            prec_name = f" ({pr})" if FP32 and FP64 is True else ""
            label = kernel + prec_name
            ax.plot(n, gflops,
                        color=PALETTE.get(kernel, DEFAULT_COLOR),
                        **PRECISION_STYLE[pr],
                        label=label)

    # Style 
    style_axes(ax)
    ax.set_xscale('log', base=2)
    pow2_ticks(ax, sorted(df['N'].unique()))
    ax.set_yscale('log')
    ax.set_ylabel('GFLOPS por execução')
    ax.set_xlabel('Tamanho do vetor N')
    ax.legend(frameon=False, loc='upper left')

    # Saving
    prec_name = ""
    if (FP32 and not FP64): prec_name = "_FP32"
    if (FP64 and not FP32): prec_name = "_FP64"
    file_name = "plot_01_GFLOPS" + prec_name + ".pdf"
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()

# 
def plot_2():
    pass

# Performance per watt X N
def plot_3(out_folder: Path, df: pd.DataFrame, FP32: bool=False, FP64: bool=False):
    
    if (not FP32 and not FP64): return
    if (FP32 and FP64): return

    # Assembling the results dictionary with data
    results = {}
    for i in range(df.shape[0]):
        kernel_name   = df['kernel_version'][i]
        precision     = df['precision'][i]
        N_val         = df['N'][i]
        perf_per_watt = df['perf_per_watt'][i]

        # Filtering
        if (precision == 'float'  and FP32 is False): continue
        if (precision == 'double' and FP64 is False): continue

        # Appending results
        if kernel_name not in results: results[kernel_name] = {}
        if precision not in results[kernel_name]: results[kernel_name][precision] = {}
        results[kernel_name][precision][N_val] = (perf_per_watt)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.5)) # final size in inches
    
    # For each Kernel
    for kernel, _ in results.items():
        ## For each precision
        for pr, data in _.items():
            n = []
            ppw = []
            ### For each value of N there are t_mean +- t_ci95
            for N_val, p in data.items():
                n.append(N_val)
                ppw.append(p)

        prec_name = f" ({pr})" if FP32 and FP64 is True else ""
        label = kernel + prec_name
        ax.plot(n, ppw,
                    color=PALETTE.get(kernel, DEFAULT_COLOR),
                    **PRECISION_STYLE[pr],
                    label=label)
    
    # Style 
    style_axes(ax)
    ax.set_xscale('log', base=2)
    pow2_ticks(ax, sorted(df['N'].unique()))
    ax.set_yscale('log')
    ax.set_ylabel('GFLOPS por Watt')
    ax.set_xlabel('Tamanho do vetor N')
    ax.legend(frameon=False, loc='upper left')

    # Saving
    prec_name = ""
    if (FP32): prec_name = "_FP32"
    if (FP64): prec_name = "_FP64"
    file_name = "plot_03" + prec_name + ".pdf"
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()


def plot_4():
    pass


def main():
    apply_academic_style()

    output_folder, results_df, roofline_json, results_profile = setup_plots(sys.argv)

    # Plot 0: Roofline Model
    plot_0(output_folder, results_df, roofline_json, results_profile, FP32=True)
    plot_0(output_folder, results_df, roofline_json, results_profile, FP64=True)

    # Plot 1: Kernel Performance (mean time per N)
    # plot_1(output_folder, results_df)
    plot_1(output_folder, results_df, FP32=True, FP64=False)
    plot_1(output_folder, results_df, FP32=False, FP64=True)

    # Plot 1.1: Kernel Performance (GFLOPS per N)
    # plot_1_gflops(output_folder, results_df)
    plot_1_gflops(output_folder, results_df, FP32=True, FP64=False)
    plot_1_gflops(output_folder, results_df, FP32=False, FP64=True)

    # Plot 2: Accumulated Speedup


    # Plot 3:
    plot_3(output_folder, results_df, FP32=True)
    plot_3(output_folder, results_df, FP64=True)


if __name__ == '__main__':
    main()
