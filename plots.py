import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

from arcane import setup_plots

PALETTE = { # OKABE_ITO
    'FFT_Naive':             '#0072B2',   # TODO: final color
    'FFT_Naive_OutOfPlace':  '#E69F00',   # TODO: final color
    'cuFFT':                 '#009E73',   # TODO: placeholder
    'FFT_Radix4':            '#D55E00',   # TODO: placeholder
    'FFT_Other':             '#CC79A7',   # TODO: placeholder
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


def plot_0():
    pass    

# Mean-time per vector size
def plot_1(df: pd.DataFrame, out_folder: Path, FP32: bool=True, FP64: bool=True) -> None:
    
    if (not FP32 and not FP64): return # both can't be false at the same time

    fig, ax = plt.subplots(figsize=(6, 3.5)) # final size in inches

    # Assembling the results dictionary with data
    results = {}
    for i in range(df.shape[0]):
        kernel_name = df['kernel_version'][i]
        precision = df['precision'][i]
        N_val = df['N'][i]
        time_mean_ms = df['time_mean_ms'][i]
        time_ci95 = df['time_ci95'][i]

        if (precision == 'float'  and FP32 is False): continue
        if (precision == 'double' and FP64 is False): continue

        if kernel_name not in results: results[kernel_name] = {}
        if precision not in results[kernel_name]: results[kernel_name][precision] = {}
        results[kernel_name][precision][N_val] = (time_mean_ms, time_ci95)

    # Plot

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

def plot_2():
    pass
def plot_3():
    pass
def plot_4():
    pass


def main():
    apply_academic_style()

    output_folder, results_csv, roofline_json, results_profile = setup_plots(sys.argv)
    
    df = pd.read_csv(results_csv)
    
    plot_1(df, output_folder)
    plot_1(df, output_folder, FP32=True, FP64=False)
    plot_1(df, output_folder, FP32=False, FP64=True)


if __name__ == '__main__':
    main()
