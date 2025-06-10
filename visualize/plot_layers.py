import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file, set the first column as index
# Note: Please create the 'image' folder before saving the image if it does not exist

df = pd.read_csv('layers.csv', index_col=0)

# Ensure num_layers is of integer type
df['num_layers'] = df['num_layers'].astype(int)

# Sort by number of layers
df = df.sort_values('num_layers')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['num_layers'], df['HR@10'], marker='o', label='HR@10')
plt.plot(df['num_layers'], df['NDCG@10'], marker='o', label='NDCG@10')
plt.plot(df['num_layers'], df['HR@20'], marker='o', label='HR@20')
plt.plot(df['num_layers'], df['NDCG@20'], marker='o', label='NDCG@20')

plt.xlabel('Number of Layers (num_layers)')
plt.ylabel('Metric Value')
plt.title('Metrics vs. Number of Layers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('image/layers_metrics.png')
plt.show()