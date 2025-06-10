import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Read CSV file
df = pd.read_csv('visualize/new_experiment_summary.csv')

# Metrics and experiment names
metrics = ['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
experiment_names = df['Experiment_Name']

# Create output directory
output_dir = 'image'
os.makedirs(output_dir, exist_ok=True)

# Set bar chart parameters
bar_width = 0.2
x = np.arange(len(experiment_names))

# Pastel color palette
pastel_colors = ['#AEC6CF', '#FFDAB9', '#B0E0E6', '#C3B091']

plt.figure(figsize=(14, 7))

# Plot grouped bar chart for each metric
for i, metric in enumerate(metrics):
    plt.bar(x + i * bar_width, df[metric], width=bar_width, label=metric, color=pastel_colors[i])

plt.xlabel('Experiment Name')
plt.ylabel('Metric Value')
plt.title('Grouped Bar Chart of Experiment Metrics (New)')
plt.xticks(x + 1.5 * bar_width, experiment_names, rotation=30, ha='right')
plt.ylim(0.4, 1.0)  # 设置y轴起始值为0.4，上限为1.0
plt.legend()
plt.tight_layout()

# Save image
output_path = os.path.join(output_dir, 'new_experiment_summary_grouped_bar.png')
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Image saved to: {output_path}")
