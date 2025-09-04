import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

def plot_aggregated_errors_from_csv(file_path, error_metric, num_update_steps, output_path='outputs/error_plot.png', update_period=1):
    """
    Plots the average, minimum, and maximum of the specified error metric across multiple targets from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        error_metric (str): The name of the error metric column to plot (e.g., 'rmse').
        num_update_steps (int): The number of update steps.
        output_path (str): Path to save the output plot.

    Returns:
        None
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Check if the required columns exist
    required_columns = {'idx', 'update_step', error_metric}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain the columns: {required_columns}")
    # Filter out rows corresponding to the 'avg' value for each target
    avg_rows = data[data['update_step'] == 'avg']
    avg_of_averages = avg_rows[error_metric].mean()

    # Remove rows with 'avg' from the data for further processing
    data = data[data['update_step'] != 'avg']

    # Convert 'update_step' to numeric for proper sorting and processing
    data['update_step'] = pd.to_numeric(data['update_step'])

    # Initialize lists to store aggregated values
    avg_values = []
    min_values = []
    max_values = []
    mean_over_years = {}
    temp = 0
    years = 1
    updates_per_years = 12/ update_period  

    # Compute aggregated values for each update step
    for step in range(num_update_steps):
        step_data = data[data['update_step'] == step][error_metric]
        avg_values.append(step_data.mean())
        min_values.append(step_data.min())
        max_values.append(step_data.max())
        if (step+2) % updates_per_years == 0 and step != 0: # first year 11, then 12
            temp += step_data.mean()
            year = f"year_{years}"
            mean_over_years[year] = (temp / 12)
            temp = 0
            years += 1
        else:
            temp += step_data.mean()
        
    # Plot the aggregated data
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_update_steps), avg_values, marker='o', label='Average', color='blue')
    plt.fill_between(range(num_update_steps), min_values, max_values, color='lightblue', alpha=0.5, label='Min-Max Range')
    plt.xlabel('Update Step')
    plt.ylabel(error_metric.capitalize())
    plt.title(f'name: {output_path}, with average error of {avg_of_averages:.4f}')
    plt.grid(True)
    plt.legend()

    # Save the plot to the outputs directory
    plt.savefig(output_path + f"/{error_metric}_plot.pdf")
    plt.close()
    print(f"Plot saved to {output_path}")

    

# Example usage:
if __name__ == "__main__":
    path = 'outputs/final_1_exp_1_clupdate_1_month'
    metric = 'mase'
    plot_aggregated_errors_from_csv(path + "/errors.csv", metric, num_update_steps=61, output_path = path)
