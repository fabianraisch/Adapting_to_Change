import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


def set_academic_style():
    """
    Sets font and style for academic-quality plots.
    Tries to use 'Gulliver'; if not available, falls back to 'Times New Roman'.
    """
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    if 'Gulliver' in available_fonts:
        font_family = 'Gulliver'
    elif 'Times New Roman' in available_fonts:
        font_family = 'Times New Roman'
    else:
        font_family = 'serif'

    plt.rcParams.update({
        'font.family': font_family,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
    })


def plot_aggregated_errors_multiple_csvs(file_paths,
                                         error_metric,
                                         num_update_steps,
                                         titles=None,
                                         y_limits=None,
                                         update_period=1,
                                         output_path='outputs/error_plots_combined.pdf',
                                         trend_window_size=6,
                                         trend_sample_interval=6):
    """
    Plots the average, min, and max of the specified error metric across multiple CSV files in stacked subplots.
    Adds a moving average trend line plotted every `trend_sample_interval` steps using a window of size `trend_window_size`.
    """

    set_academic_style()
    num_files = len(file_paths)
    if titles is not None and len(titles) != num_files:
        raise ValueError("Length of titles must match number of file paths.")

    fig, axs = plt.subplots(num_files, 1, figsize=(6, 2.5 * num_files), sharex=True)

    if num_files == 1:
        axs = [axs]

    for i, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path)

        required_columns = {'idx', 'update_step', error_metric}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"CSV file must contain the columns: {required_columns}")

        avg_rows = data[data['update_step'] == 'avg']
        avg_of_averages = avg_rows[error_metric].mean()
        print(f"[{os.path.basename(file_path)}] Average of the averages: {avg_of_averages}")

        data = data[data['update_step'] != 'avg']
        data['update_step'] = pd.to_numeric(data['update_step'])

        avg_values = []
        min_values = []
        max_values = []

        for step in range(num_update_steps):
            step_data = data[data['update_step'] == step][error_metric]
            avg_values.append(step_data.mean())
            min_values.append(step_data.min())
            max_values.append(step_data.max())

        # Compute moving average with adjustable window
        moving_avg = pd.Series(avg_values).rolling(window=trend_window_size, min_periods=trend_window_size).mean()

        # Determine trend sample points
        sampled_steps = list(range(trend_window_size - 1, num_update_steps, trend_sample_interval))
        sampled_moving_avg = moving_avg.iloc[sampled_steps]

        ax = axs[i]
        x_range = range(num_update_steps)

        ax.plot(x_range, avg_values, marker='o', label='Mean', color='navy')
        ax.fill_between(x_range, min_values, max_values, color='lightsteelblue', alpha=0.6, label='Min–Max')

        # Plot sparse trend line in Elsevier orange
        ax.plot(sampled_steps, sampled_moving_avg, color='#FF6F00', linestyle='-', marker='s',
                label=f'{trend_window_size}-step Trend')

        ax.set_ylabel(error_metric.upper())

        if titles:
            ax.set_title(f"{titles[i]} (Avg: {avg_of_averages:.4f})")
        else:
            ax.set_title(f"{os.path.basename(os.path.dirname(file_path))} (Avg: {avg_of_averages:.4f})")

        if y_limits:
            ax.set_ylim(y_limits)

        if i == 0:
            ax.legend(loc='upper right', frameon=False)

    axs[-1].set_xlabel("Update Step")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Combined plot saved to {output_path}")
