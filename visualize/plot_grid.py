import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 读取CSV文件
df = pd.read_csv('visualize/grid_search_summary.csv')

# 只保留前20个实验（如实验太多可适当筛选，否则图太挤）
df = df.head(20)

# 指标和实验名称
metrics = ['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
experiment_names = df.iloc[:, 0]  # 第一列为实验名

# 创建输出文件夹
output_dir = 'image'
os.makedirs(output_dir, exist_ok=True)

# 设置柱状图参数
bar_width = 0.2
x = np.arange(len(experiment_names))

# 柔和色系
pastel_colors = ['#AEC6CF', '#FFDAB9', '#B0E0E6', '#C3B091']

plt.figure(figsize=(16, 8))

# 绘制每个指标的簇形柱状图
for i, metric in enumerate(metrics):
    plt.bar(x + i * bar_width, df[metric], width=bar_width, label=metric, color=pastel_colors[i])

plt.xlabel('Experiment Name')
plt.ylabel('Metric Value')
plt.title('Grouped Bar Chart of Grid Search Results')
plt.xticks(x + 1.5 * bar_width, experiment_names, rotation=45, ha='right', fontsize=8)
plt.legend()
plt.tight_layout()

# 保存图片
output_path = os.path.join(output_dir, 'grid_search_grouped_bar.png')
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Image saved to: {output_path}")