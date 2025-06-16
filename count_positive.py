import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
def analyze_user_positive_distribution(csv_path, dataset_name="Train"):
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    # 按用户分组，统计每个用户的正样本数量
    user_positive_counts = df.groupby('user')['course'].count()
    
    # 输出统计信息
    print(f"--- {dataset_name} 集用户正样本统计 ---")
    print(f"用户总数：{len(user_positive_counts)}")
    print(f"平均每个用户正样本数：{user_positive_counts.mean():.2f}")
    print(f"最少正样本数：{user_positive_counts.min()}")
    print(f"最多正样本数：{user_positive_counts.max()}")
    
    # 绘制分布图
    plt.figure(figsize=(8, 4))
    sns.histplot(user_positive_counts, bins=30, kde=False)
    plt.title(f'{dataset_name} 用户正样本数量分布')
    plt.xlabel('正样本数量')
    plt.ylabel('用户数')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return user_positive_counts

# 文件路径（替换为你的路径）
train_path = "data/train.csv"
test_path = "data/test.csv"

# 分析训练集和测试集
train_distribution = analyze_user_positive_distribution(train_path, "Train")
test_distribution = analyze_user_positive_distribution(test_path, "Test")