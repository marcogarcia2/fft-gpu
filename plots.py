import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import NullFormatter, FuncFormatter
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path
import sys
import math

from arcane import setup_plots, get_results_folder, parse_NCU_file_name

COLOR = {  # OKABE_ITO PALETTE 
    'green':        '#009E73', #
    'blue':         '#0072B2', #
    'yellow':       '#F0E442', #
    'pink':         '#CC79A7', #
    'orange':       '#E69F00', #
    'red':          '#D55E00', #
    'lightblue':    '#56B4E9', #
    'black':        '#000000'
}


# Name shown in the figures: 'FFT_Radix4_Tabled' -> 'Radix4'. Charts that mix both versions
# pass mark_tabled=True to tell them apart: 'Radix4 (tab.)'
def display_name(kernel: str, mark_tabled: bool=False) -> str:
    name = kernel.removeprefix('FFT_')
    if name.endswith('_Tabled'):
        name = name.removesuffix('_Tabled') + (' (tab.)' if mark_tabled else '')
    return name

# Number with a decimal comma (Brazilian notation): 1.33 -> '1,33'
def comma(value: float, digits: int=2) -> str:
    return f'{value:.{digits}f}'.replace('.', ',')

# One marker shape per kernel (a Tabled version keeps its base kernel's shape), drawn hollow
# where curves or points overlap, so the ones underneath stay visible
KERNEL_MARKER = {'cuFFT': 'o', 'Naive': '^', 'Stockham': 's', 'Shared': 'D', 'BCF': 'v', 'Radix4': 'P'}

def kernel_marker(kernel: str) -> dict:
    return dict(marker=KERNEL_MARKER.get(display_name(kernel), 'o'), markerfacecolor='none')

def precision_style(pr: str) -> dict:
    return dict(marker='o', linestyle='-') if pr == 'float' else dict(marker='s', linestyle='--')

# Latin Modern (same font as the LaTeX document), taken from the TeX Live installation
LATIN_MODERN_DIR = Path('/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/lm')

# Opens each figure in an interactive window only when --show (or -show) is given
SHOW = False

def show_or_close(fig):
    if SHOW: plt.show()
    else:    plt.close(fig)      # frees the figure, since nothing will display it

def apply_academic_style():
    for font in LATIN_MODERN_DIR.glob('lmroman10-*.otf'):
        font_manager.fontManager.addfont(font)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman', 'DejaVu Serif'],  # DejaVu if Latin Modern is not found
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
def plot_0(out_folder: Path, df: pd.DataFrame, roofline_json: dict, results_profile: dict, kernels: dict, file_name: str, FP32: bool=False, FP64: bool=False):

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

        # Kernel (or precision/N) not profiled, e.g. commented out of run.sh's kernel_list: skip it
        read_write_bytes = results_profile.get(kernel_name, {}).get(precision, {}).get(N_val)
        if read_write_bytes is None:
            print(f"Warning: no NCU profile for {kernel_name} ({precision}, N={N_val}); skipping in roofline")
            continue
        sum_bytes = read_write_bytes[0] + read_write_bytes[1]
        arithmetic_intensity = 5 * N_val * math.log(N_val, 2) / sum_bytes

        if kernel_name not in results: results[kernel_name] = {}
        if precision not in results[kernel_name]: results[kernel_name][precision] = {}
        results[kernel_name][precision][N_val] = (arithmetic_intensity, gflops)

    fig, ax = plt.subplots(figsize=(6, 3.5)) # 

    # Plot
    # Several kernels land on almost the same point: hollow markers with one shape per kernel
    # keep the ones underneath visible
    # For each Kernel
    for kernel, color in kernels.items():
        if kernel not in results:
            print(f"Warning: no results for {kernel}; skipping")
            continue
        _ = results[kernel]
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
            label = display_name(kernel) + prec_name
            ax.plot(ar_in, gfl, color=color,
                        **kernel_marker(kernel),
                        markeredgewidth=1.6,
                        markersize=9,
                        linestyle='none',
                        label=label)

    # Plot Rooflines
    peak_bw = roofline_json['peak_bandwidth_gbps']
    peak_gflops = roofline_json['peak_gflops_fp32'] if FP32 else roofline_json['peak_gflops_fp64']
    ridge = peak_gflops / peak_bw                     # ponto de inflexão
    x_min = 0.1 if FP32 else 0.01
    x_max = ridge * 8                                 # a bit past the ridge: the flat roof adds nothing
    x = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    roof = np.minimum(peak_bw * x, peak_gflops)
    ax.plot(x, roof, color='black', linewidth=1.2)
    ax.axvline(ridge, color='gray', linestyle=':', linewidth=0.8)

    # Style
    style_axes(ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(peak_bw * x_min * 0.6, peak_gflops * 2.5)
    ax.set_ylabel('Desempenho (GFLOP/s)')
    ax.set_xlabel('Intensidade aritmética (FLOP/byte)')
    ax.legend(frameon=False, loc='upper left')

    # Shaded regions: memory bound left of the ridge, compute bound right of it
    ax.axvspan(x_min, ridge, color=COLOR['blue'], alpha=0.06, linewidth=0, zorder=0)
    ax.axvspan(ridge, x_max, color=COLOR['orange'], alpha=0.08, linewidth=0, zorder=0)
    ax.text(ridge / 1.15, peak_bw * x_min * 0.7, 'Limitado por memória', ha='right', va='bottom',
            fontsize=7, color='dimgray')
    ax.text(ridge * 1.15, peak_bw * x_min * 0.7, 'Limitado por computação', ha='left', va='bottom',
            fontsize=7, color='dimgray')

    # Ceiling labels on the lines. The slope of the bandwidth line on screen depends on the final
    # axes size, so the layout is fixed first and the angle taken from two points of the line
    fig.tight_layout()
    peak_name = 'FP32' if FP32 else 'FP64'
    ax.text(x_max / 1.15, peak_gflops * 1.1, f'Pico {peak_name}: {peak_gflops:,.0f} GFLOP/s'.replace(',', '.'),
            ha='right', va='bottom', fontsize=7)
    x0, x1 = x_min * 2, x_min * 6           # near the left edge, away from the kernels
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, peak_bw * x0), (x1, peak_bw * x1)])
    angle = np.degrees(np.arctan2(py1 - py0, px1 - px0))
    ax.annotate(f'Banda DRAM: {peak_bw:,.0f} GB/s', (x0, peak_bw * x0), xytext=(0, 12),
                textcoords='offset points', rotation=angle, rotation_mode='anchor',
                ha='left', va='bottom', fontsize=7)

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Mean-time (ms) X vector size
def plot_1(out_folder: Path, df: pd.DataFrame, kernels: dict, file_name: str, FP32: bool=True, FP64: bool=True):

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
    for kernel, color in kernels.items():
        if kernel not in results:
            print(f"Warning: no results for {kernel}; skipping")
            continue
        _ = results[kernel]
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
            label = display_name(kernel) + prec_name
            ax.errorbar(n, t_mean, yerr=t_ci95, capsize=3,
                        color=color,
                        **precision_style(pr),
                        label=label)

    # Style 
    style_axes(ax)
    ax.set_xscale('log', base=2)
    pow2_ticks(ax, sorted(df['N'].unique()))
    ax.set_yscale('log')
    ax.set_ylabel('Tempo (ms)')
    ax.set_xlabel(r'Tamanho da entrada ($N$)')
    ax.legend(frameon=False, loc='upper left')

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Pareto time X energy for a fixed N: one panel per precision.
# Tabled kernels use a hollow marker, so a kernel and its Tabled version can share a color.
# The dashed line is the Pareto front: kernels that no other one beats in both time and energy.
def plot_2(out_folder: Path, df: pd.DataFrame, kernels: dict, file_name: str, N: int):

    fig, axes = plt.subplots(1, 2, figsize=(6, 3)) # final size in inches

    for ax, (pr, title) in zip(axes, [('float', 'FP32'), ('double', 'FP64')]):
        data = df[(df['precision'] == pr) & (df['N'] == N)].set_index('kernel_version')

        points = []                                     # (time, energy, kernel)
        for kernel, color in kernels.items():
            if kernel not in data.index:
                print(f"Warning: no results for {kernel} ({pr}, N={N}); skipping")
                continue
            row = data.loc[kernel]
            t, e = row['time_mean_ms'], row['energy_mean_mj']
            points.append((t, e, kernel))

            hollow = kernel.endswith('_Tabled')
            ax.errorbar(t, e, xerr=row['time_ci95'], yerr=row['energy_ci95'],
                        color=color, marker='^' if hollow else 'o',
                        markerfacecolor='white' if hollow else color,
                        linestyle='none', capsize=2, label=display_name(kernel, mark_tabled=True), zorder=3)

        # Pareto front: walking by time, keep each point that lowers the energy so far
        front = []
        for t, e, kernel in sorted(points):
            if not front or e < front[-1][1]:
                front.append((t, e, kernel))
        ax.plot([p[0] for p in front], [p[1] for p in front],
                color='gray', linestyle='--', linewidth=0.8, label='Fronteira de Pareto', zorder=2)

        # Style
        style_axes(ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_minor_formatter(NullFormatter())   # minor labels overlap on short log ranges
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_xlabel('Tempo (ms)')
    axes[0].set_ylabel('Energia (mJ)')

    fig.tight_layout()

    # Legends below the panels: first one entry per color (kernel without the _Tabled suffix),
    # then one entry per marker shape plus the Pareto front
    colors = {}
    for kernel, color in kernels.items():
        colors.setdefault(display_name(kernel), color)
    color_handles = [Line2D([], [], color=c, marker='s', linestyle='none', label=k)
                     for k, c in colors.items()]
    shape_handles = [
        Line2D([], [], color='black', marker='o', linestyle='none', label='Cálculo direto'),
        Line2D([], [], color='black', marker='^', markerfacecolor='white', linestyle='none', label='Tabelado'),
        Line2D([], [], color='gray', linestyle='--', linewidth=0.8, label='Fronteira de Pareto'),
    ]
    fig.legend(handles=color_handles, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(color_handles), fontsize=8,
               handletextpad=0.3, columnspacing=1.0)
    fig.legend(handles=shape_handles, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, -0.08), ncol=len(shape_handles), fontsize=8,
               handletextpad=0.3, columnspacing=1.0)

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Performance per watt X N
def plot_3(out_folder: Path, df: pd.DataFrame, kernels: dict, file_name: str, FP32: bool=False, FP64: bool=False):
    
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
    for kernel, color in kernels.items():
        if kernel not in results:
            print(f"Warning: no results for {kernel}; skipping")
            continue
        _ = results[kernel]
        ## For each precision
        for pr, data in _.items():
            n = []
            ppw = []
            ### For each value of N there are t_mean +- t_ci95
            for N_val, p in data.items():
                n.append(N_val)
                ppw.append(p)

            prec_name = f" ({pr})" if FP32 and FP64 is True else ""
            label = display_name(kernel) + prec_name
            ax.plot(n, ppw,
                        color=color,
                        **precision_style(pr),
                        label=label)
    
    # Style 
    style_axes(ax)
    ax.set_xscale('log', base=2)
    pow2_ticks(ax, sorted(df['N'].unique()))
    ax.set_yscale('log')
    ax.set_ylabel('GFLOPS por Watt')
    ax.set_xlabel(r'Tamanho da entrada ($N$)')
    ax.legend(frameon=False, loc='upper left')

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)


# Performance per watt X N in a 2x2 grid: rows are precisions, columns direct-trig / Tabled.
# Each row shares its y axis, so a direct kernel and its Tabled version compare side by side.
# A kernel and its Tabled version share color and name, so one legend below serves all panels.
def plot_3_grid(out_folder: Path, df: pd.DataFrame, kernels_direct: dict, kernels_tabled: dict, file_name: str):

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.8), sharex=True, sharey='row') # final size in inches
    ns = sorted(df['N'].unique())

    for row, (pr, pr_title) in enumerate([('float', 'FP32'), ('double', 'FP64')]):
        ppw = (df[df['precision'] == pr]
               .pivot_table(index='N', columns='kernel_version', values='perf_per_watt'))
        for col, kernels in enumerate([kernels_direct, kernels_tabled]):
            ax = axes[row, col]
            for kernel, color in kernels.items():
                if kernel not in ppw.columns:
                    print(f"Warning: no results for {kernel} ({pr}); skipping")
                    continue
                ax.plot(ppw.index, ppw[kernel], color=color, marker='o', markersize=3.5,
                        linewidth=1.2, label=display_name(kernel))

            # Style
            style_axes(ax)
            ax.set_xscale('log', base=2)
            pow2_ticks(ax, ns)
            ax.set_yscale('log')
        axes[row, 0].set_ylabel(f'{pr_title}\nGFLOPS por Watt')

    axes[0, 0].set_title('Cálculo direto')
    axes[0, 1].set_title('Tabelados')
    for ax in axes[1]:
        ax.set_xlabel(r'Tamanho da entrada ($N$)')
    fig.tight_layout()

    # One legend for all panels, below them (only shows up in the saved file:
    # plt.show() cuts anything outside the figure)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels), fontsize=8,
               handletextpad=0.3, columnspacing=1.0)

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Ratio Tabled / direct-trig time X N: one panel per precision, one line per kernel.
# The dotted vertical line is the N where the data (N complex values), the auxiliary vector
# (N, used by every version but Naive) and the twiddle table (N/2) together outgrow the L2.
def plot_4(out_folder: Path, df: pd.DataFrame, kernels: dict, file_name: str, l2_bytes: float):

    fig, axes = plt.subplots(1, 2, figsize=(6, 3)) # final size in inches

    for ax, (pr, title, complex_bytes) in zip(axes, [('float', 'FP32', 8), ('double', 'FP64', 16)]):
        times = (df[df['precision'] == pr]
                 .pivot_table(index='N', columns='kernel_version', values='time_mean_ms'))
        ns = list(times.index)

        for kernel, color in kernels.items():
            tabled = kernel + '_Tabled'
            if kernel not in times.columns or tabled not in times.columns:
                print(f"Warning: no results for {kernel} or {tabled} ({pr}); skipping")
                continue
            ax.plot(ns, times[tabled] / times[kernel], color=color, **precision_style(pr), label=display_name(kernel))

        ax.axhline(1.0, color='black', linewidth=0.8)

        # data (N complex) + auxiliary (N) + table (N/2) = 2.5 * N * complex_bytes > L2
        n_l2 = l2_bytes / (2.5 * complex_bytes)
        ax.axvline(n_l2, color='gray', linestyle=':', linewidth=0.8, label='Dados, auxiliar e tabela excedem a L2')

        # Style
        style_axes(ax)
        ax.set_xscale('log', base=2)
        pow2_ticks(ax, ns)
        ax.set_title(title)
        ax.set_xlabel(r'Tamanho da entrada ($N$)')
    axes[0].set_ylabel('Tempo tabelado / tempo cálculo direto')
    fig.tight_layout()

    # One legend for both panels, below them (only shows up in the saved file:
    # plt.show() cuts anything outside the figure)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels), fontsize=8,
               handletextpad=0.3, columnspacing=1.0)

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Bottleneck per kernel at a fixed N, from the Nsight CSVs: one panel per precision, one group
# of bars per kernel, one bar per unit (% of its peak). Each kernel has several launches (one per
# stage), so every metric is averaged weighted by the launch duration.
def plot_5(out_folder: Path, results_folder: Path, kernels: list, file_name: str, N: int):

    metrics = {
        'DRAM':                  ('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',                 COLOR['blue']),
        'Emissão de instruções': ('sm__issue_active.avg.pct_of_peak_sustained_elapsed',                     COLOR['orange']),
        'ALU':                   ('sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed',           COLOR['yellow']),
        'Pipe FP64':             ('sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed',          COLOR['pink']),
        'Memória compartilhada': ('l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed', COLOR['green']),
    }

    # usage[precision][kernel][metric] = weighted mean (%)
    usage = {'float': {}, 'double': {}}
    for f in results_folder.glob('*.csv'):
        if f.name == 'Results.csv': continue
        kname, Nval, precision = parse_NCU_file_name(f)
        if Nval != N or kname not in kernels: continue

        ncu = pd.read_csv(f, skiprows=[1], thousands=',')     # row 1 holds the units
        w = ncu['gpu__time_duration.sum']
        usage[precision][kname] = {m: (ncu[col] * w).sum() / w.sum() for m, (col, _) in metrics.items()}

    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True) # final size in inches

    bars = len(metrics)
    width = 0.8 / bars
    x = np.arange(len(kernels))

    for ax, (pr, title) in zip(axes, [('float', 'FP32'), ('double', 'FP64')]):
        missing = [k for k in kernels if k not in usage[pr]]
        if missing: print(f"Warning: no NCU profile for {missing} ({pr}, N={N}); skipping")

        for i, (metric, (_, color)) in enumerate(metrics.items()):
            values = [usage[pr].get(k, {}).get(metric, np.nan) for k in kernels]
            pos = x + (i - (bars - 1) / 2) * width
            ax.bar(pos, values, width=width, color=color, label=metric)

        # Dotted separators where the group changes (cuFFT | direct trig | Tabled)
        groups = [0 if k == 'cuFFT' else 2 if k.endswith('_Tabled') else 1 for k in kernels]
        for i in range(1, len(kernels)):
            if groups[i] != groups[i - 1]:
                ax.axvline(i - 0.5, color='gray', linestyle=':', linewidth=0.8)

        # Style
        style_axes(ax)
        ax.grid(False, axis='x')
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel('Utilização (% do pico)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([display_name(k, mark_tabled=True) for k in kernels], rotation=30, ha='right', fontsize=7)
    fig.tight_layout()

    # One legend for both panels, below them (only shows up in the saved file:
    # plt.show() cuts anything outside the figure)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels), fontsize=8,
               handletextpad=0.3, columnspacing=1.0)

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Same data as plot_5 drawn as a heatmap: one panel per precision, metrics as rows, kernels as
# columns, and the % of peak written in each cell. The bottleneck of a kernel is the darkest cell
# of its column. White gaps split the groups (cuFFT | direct trig | Tabled), named above.
def plot_5_heatmap(out_folder: Path, results_folder: Path, kernels: list, file_name: str, N: int):

    metrics = {
        'DRAM':                  'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
        'Emissão de instruções': 'sm__issue_active.avg.pct_of_peak_sustained_elapsed',
        'ALU':                   'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed',
        'Pipe FP64':             'sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed',
        'Memória compartilhada': 'l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed',
    }

    # usage[precision][kernel][metric] = weighted mean (%)
    usage = {'float': {}, 'double': {}}
    for f in results_folder.glob('*.csv'):
        if f.name == 'Results.csv': continue
        kname, Nval, precision = parse_NCU_file_name(f)
        if Nval != N or kname not in kernels: continue

        ncu = pd.read_csv(f, skiprows=[1], thousands=',')     # row 1 holds the units
        w = ncu['gpu__time_duration.sum']
        usage[precision][kname] = {m: (ncu[col] * w).sum() / w.sum() for m, col in metrics.items()}

    groups = [0 if k == 'cuFFT' else 2 if k.endswith('_Tabled') else 1 for k in kernels]
    cmap = plt.get_cmap('Blues')
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.2), sharex=True) # final size in inches

    for ax, (pr, title) in zip(axes, [('float', 'FP32'), ('double', 'FP64')]):
        missing = [k for k in kernels if k not in usage[pr]]
        if missing: print(f"Warning: no NCU profile for {missing} ({pr}, N={N}); skipping")

        grid = np.array([[usage[pr].get(k, {}).get(m, np.nan) for k in kernels] for m in metrics])
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect='auto')

        # Value in each cell, white over the dark ones
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                if np.isnan(v): continue
                ax.text(c, r, f'{v:.0f}', ha='center', va='center', fontsize=7.5,
                        color='white' if v > 55 else 'black')

        # Thin white lines between cells, thick ones between groups
        ax.set_xticks(np.arange(len(kernels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        for i in range(1, len(kernels)):
            if groups[i] != groups[i - 1]:
                ax.axvline(i - 0.5, color='white', linewidth=6)

        # Style
        ax.tick_params(which='both', length=0)
        ax.spines[:].set_visible(False)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(list(metrics), fontsize=8)
        ax.set_title(title, loc='left', fontsize=9.5)

    axes[1].set_xticks(range(len(kernels)))
    axes[1].set_xticklabels([display_name(k) for k in kernels], rotation=30, ha='right', fontsize=8)   # group names tell Tabled apart

    # Group names above the first panel, centered over their columns
    for g, name in enumerate(['cuFFT', 'Calculados', 'Tabelados']):
        cols = [i for i, x in enumerate(groups) if x == g]
        if not cols or name == 'cuFFT': continue          # cuFFT's column already says it
        axes[0].text((cols[0] + cols[-1]) / 2, -0.75, name, ha='center', va='bottom', fontsize=8.5)

    fig.tight_layout(h_pad=2.0)                       # a bit of air between the FP32 and FP64 panels
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label('Utilização (% do pico)', fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, length=2)
    cbar.outline.set_visible(False)

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)

# Gain per optimization step at a fixed N: one panel per precision, one bar per version with its
# cumulative speedup over the first kernel (the baseline), in the given order. Above each bar goes
# the cumulative speedup and, below it, the gain of that step over the previous version.
# The reference kernel (cuFFT) is drawn as a dashed horizontal line.
def plot_6(out_folder: Path, df: pd.DataFrame, kernels: dict, file_name: str, N: int, reference: str='cuFFT'):

    names = list(kernels)
    baseline = names[0]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3)) # final size in inches

    for ax, (pr, title) in zip(axes, [('float', 'FP32'), ('double', 'FP64')]):
        data = df[(df['precision'] == pr) & (df['N'] == N)].set_index('kernel_version')
        missing = [k for k in names + [reference] if k not in data.index]
        if missing:
            print(f"Warning: no results for {missing} ({pr}, N={N}); skipping panel")
            continue

        t  = data['time_mean_ms']
        ci = data['time_ci95']
        speedup = [t[baseline] / t[k] for k in names]
        # relative errors of a ratio add in quadrature
        err = [s * math.hypot(ci[baseline] / t[baseline], ci[k] / t[k]) for s, k in zip(speedup, names)]

        x = np.arange(len(names))
        ax.bar(x, speedup, yerr=err, width=0.6, color=[kernels[k] for k in names],
               capsize=2, error_kw=dict(linewidth=0.6), zorder=3)

        ref = t[baseline] / t[reference]
        ax.axhline(ref, color='black', linestyle='--', linewidth=0.8, zorder=2)
        # label over the first bar (the baseline, always the shortest) so it never hits the others
        ax.text(-0.4, ref, f'{display_name(reference)} ({comma(ref)}×)', ha='left', va='bottom', fontsize=7)

        # Above each bar: cumulative speedup and, in parentheses, the gain of that step
        for i, s in enumerate(speedup):
            label = f'{comma(s)}×' if i == 0 else f'{comma(s)}×\n(×{comma(s / speedup[i - 1])})'
            ax.annotate(label, (i, s + err[i]), xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.5, linespacing=1.1)

        # Style
        style_axes(ax)
        ax.grid(False, axis='x')
        ax.set_ylim(0, max(ref, max(speedup)) * 1.18)   # room for the labels, even above the reference
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(k) for k in names], rotation=30, ha='right', fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel(f'Speedup acumulado sobre {display_name(baseline)}')
    fig.tight_layout()

    # Saving
    out_path = out_folder / file_name
    fig.savefig(out_path, bbox_inches='tight')
    show_or_close(fig)


def main():
    apply_academic_style()

    # Flags are taken out of argv, so arcane still sees just [script, results_folder]
    global SHOW
    flags = {'--show', '-show'}
    SHOW = any(arg in flags for arg in sys.argv)
    argv = [arg for arg in sys.argv if arg not in flags]

    output_folder, results_df, roofline_json, results_profile = setup_plots(argv)

    ### PURE VERSIONS
    kernels = {
        'cuFFT':        COLOR['green'],
        'FFT_Naive':    COLOR['red'],
        'FFT_Stockham': COLOR['lightblue'],
        'FFT_Shared':   COLOR['blue'],
        'FFT_BCF':      COLOR['orange'],
        'FFT_Radix4':   COLOR['black'],
    }

    # Plot 0: Roofline Model
    plot_0(output_folder, results_df, roofline_json, results_profile, kernels, "plot_00_FP32.pdf", FP32=True)
    plot_0(output_folder, results_df, roofline_json, results_profile, kernels, "plot_00_FP64.pdf", FP64=True)

    # Plot 1: Kernel Performance (mean time per N)
    plot_1(output_folder, results_df, kernels, "plot_01_FP32.pdf", FP32=True, FP64=False)
    plot_1(output_folder, results_df, kernels, "plot_01_FP64.pdf", FP32=False, FP64=True)


    # Plot 6: Cumulative speedup per optimization step (N = 2^24), in the order of the steps
    kernels = {
        'FFT_Naive':    COLOR['red'],
        'FFT_Stockham': COLOR['lightblue'],
        'FFT_Shared':   COLOR['blue'],
        'FFT_BCF':      COLOR['orange'],
        'FFT_Radix4':   COLOR['black'],
    }
    plot_6(output_folder, results_df, kernels, "plot_06_Ganho_Etapas_N24.pdf", N=1 << 24)


    ### TABLED VERSIONS
    kernels = {
        'cuFFT':                COLOR['green'],
        'FFT_Naive_Tabled':     COLOR['red'],
        'FFT_Stockham_Tabled':  COLOR['lightblue'],
        'FFT_Shared_Tabled':    COLOR['blue'],
        'FFT_BCF_Tabled':       COLOR['orange'],
        'FFT_Radix4_Tabled':    COLOR['black'],
    }
    
    # Plot 0: Roofline Model
    plot_0(output_folder, results_df, roofline_json, results_profile, kernels, "plot_00_FP32_TABLED.pdf", FP32=True)
    plot_0(output_folder, results_df, roofline_json, results_profile, kernels, "plot_00_FP64_TABLED.pdf", FP64=True)

    # Plot 1: Kernel Performance (mean time per N)
    plot_1(output_folder, results_df, kernels, "plot_01_FP32_TABLED.pdf", FP32=True, FP64=False)
    plot_1(output_folder, results_df, kernels, "plot_01_FP64_TABLED.pdf", FP32=False, FP64=True)

    # Plot 6: Cumulative speedup per optimization step (N = 2^24), baseline Naive_Tabled
    kernels = {
        'FFT_Naive_Tabled':    COLOR['red'],
        'FFT_Stockham_Tabled': COLOR['lightblue'],
        'FFT_Shared_Tabled':   COLOR['blue'],
        'FFT_BCF_Tabled':      COLOR['orange'],
        'FFT_Radix4_Tabled':   COLOR['black'],
    }
    plot_6(output_folder, results_df, kernels, "plot_06_Ganho_Etapas_N24_TABLED.pdf", N=1 << 24)

    ### MIXED CHARTS
    # Plot 3: Performance per watt, 2x2 grid (precision X direct/Tabled) with a single legend
    kernels_direct = {
        'cuFFT':        COLOR['green'],
        'FFT_Naive':    COLOR['red'],
        'FFT_Stockham': COLOR['lightblue'],
        'FFT_Shared':   COLOR['blue'],
        'FFT_BCF':      COLOR['orange'],
        'FFT_Radix4':   COLOR['black'],
    }
    kernels_tabled = {
        'cuFFT':               COLOR['green'],
        'FFT_Naive_Tabled':    COLOR['red'],
        'FFT_Stockham_Tabled': COLOR['lightblue'],
        'FFT_Shared_Tabled':   COLOR['blue'],
        'FFT_BCF_Tabled':      COLOR['orange'],
        'FFT_Radix4_Tabled':   COLOR['black'],
    }
    plot_3_grid(output_folder, results_df, kernels_direct, kernels_tabled, "plot_03_GFLOPS_W.pdf")

    kernels = {
        'cuFFT':               COLOR['green'],
        'FFT_Naive':           COLOR['red'],
        'FFT_Naive_Tabled':    COLOR['red'],
        'FFT_Stockham':        COLOR['lightblue'],
        'FFT_Stockham_Tabled': COLOR['lightblue'],
        'FFT_Shared':          COLOR['blue'],
        'FFT_Shared_Tabled':   COLOR['blue'],
        'FFT_BCF':             COLOR['orange'],
        'FFT_BCF_Tabled':      COLOR['orange'],
        'FFT_Radix4':          COLOR['black'],
        'FFT_Radix4_Tabled':   COLOR['black'],
    }
    
    # Plot 2: Pareto time X energy (N = 2^24)
    plot_2(output_folder, results_df, kernels, "plot_02_Pareto_N24.pdf", N=1 << 24)

    # Plot 5: Bottleneck per kernel (Nsight, N = 2^24)
    # grouped by direct-trig / Tabled (cuFFT first), so each group reads as the optimization sequence
    order = sorted(kernels, key=lambda k: 0 if k == 'cuFFT' else 2 if k.endswith('_Tabled') else 1)
    plot_5_heatmap(output_folder, get_results_folder(argv), order, "plot_05_Gargalo_N24.pdf", N=1 << 24)

    # Plot 4: Tabled / direct-trig time ratio X N (each kernel is compared with its _Tabled twin)
    kernels = {
        'FFT_Naive':    COLOR['red'],
        'FFT_Stockham': COLOR['lightblue'],
        'FFT_Shared':   COLOR['blue'],
        'FFT_BCF':      COLOR['orange'],
        'FFT_Radix4':   COLOR['black'],
    }
    plot_4(output_folder, results_df, kernels, "plot_04_Razao_Tabelado.pdf", l2_bytes=2.25 * 2**20)

if __name__ == '__main__':
    main()
