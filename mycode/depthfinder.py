# 版本3 探索网络深度
import torch
import os
import sys
import pandas as pd
import json
import itertools
# 将 src 目录添加到 Python 路径中，以便可以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 从 src 模块导入函数和类
from src.data_loader import load_and_process_data_for_experiment
from src.model import LightGCN
from src.trainer import train_model, evaluate_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义数据文件和结果输出的目录
    data_dir = 'data/' 
    results_dir = '/results/depth_exploration_results/'
    os.makedirs(results_dir, exist_ok=True) # 确保结果目录存在


    # --- 1. 定义深度探索的参数 ---
    # 根据你之前的网格搜索结果，我们找到了最优的 embedding_dim 和 learning_rate
    best_embedding_dim = 32
    best_learning_rate = 0.001
    
    # 现在，我们只探索 num_layers 的变化
    depth_exploration_grid = {
        'num_layers': [1, 2, 3, 4, 5, 6,7,8,9], # 探索从1到6的所有层数
        'embedding_dim': [best_embedding_dim], # 固定为最优值
        'learning_rate': [best_learning_rate] # 固定为最优值
    }
    
    # 固定的训练参数
    num_epochs = 50
    batch_size = 2048 # 使用你之前调参时固定的batch_size

    # --- 2. 准备数据加载所需的固定配置 (只在Full KG模型上探索) ---
    full_kg_config = {
        "include_kg_course_knowledge": True, "include_kg_school_course": True,
        "include_kg_teacher_course": True, "include_kg_kp_prereq": True,
        "include_kg_knowledge_major": True
    }
    
    # --- 3. 数据只加载一次 ---
    print("--- Loading and Processing Data for Full KG Model (once) ---")
    num_nodes, edge_index, train_interactions, test_interactions, \
    user_ids_global_range, course_ids_global_range, \
    user_positive_items_train, user_positive_items_test, entity_to_id = \
        load_and_process_data_for_experiment(data_dir, **full_kg_config)
    
    print(f"Data loaded. Graph nodes: {num_nodes}, edges: {edge_index.shape[1] // 2}")
    
    # --- 4. 深度探索主循环 ---
    results_summary = {}
    
    # 生成所有超参数组合 (这里实际上只是遍历num_layers)
    param_combinations = list(itertools.product(
        depth_exploration_grid['num_layers'],
        depth_exploration_grid['embedding_dim'],
        depth_exploration_grid['learning_rate']
    ))
    
    total_experiments = len(param_combinations)
    print(f"\n--- Starting Depth Exploration for {total_experiments} Configurations ---")

    for i, (num_layers, embedding_dim, learning_rate) in enumerate(param_combinations):
        
        # 动态生成实验名称
        exp_name = f"Layers_{num_layers}" # 实验名称现在只关注层数
        
        print(f"\n--- Running Experiment {i+1}/{total_experiments}: {exp_name} ---")
        
        # 4.1 初始化模型
        model = LightGCN(num_nodes, embedding_dim, num_layers)
        print(f"Model initialized: layers={num_layers}, dim={embedding_dim}, lr={learning_rate}")

        # 4.2 训练模型
        trained_model, final_embeddings, epoch_losses = train_model(
            model, edge_index, train_interactions,
            user_ids_global_range, course_ids_global_range,
            num_epochs, batch_size, learning_rate, device
        )

        # 4.3 保存训练过程和结果
        exp_results_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_results_dir, exist_ok=True)
        
        torch.save(trained_model.state_dict(), os.path.join(exp_results_dir, 'model_weights.pt'))
        
        loss_df = pd.DataFrame({'epoch': range(1, num_epochs + 1), 'loss': epoch_losses})
        loss_df.to_csv(os.path.join(exp_results_dir, 'loss_history.csv'), index=False)
        print(f"Results for '{exp_name}' saved to {exp_results_dir}")

        # 4.4 评估模型
        hr_results, ndcg_results = evaluate_model(
            trained_model, edge_index, final_embeddings, 
            test_interactions,
            user_ids_global_range, course_ids_global_range,
            user_positive_items_train, user_positive_items_test,
            k_values=[10, 20], device=device
        )
        
        # 4.5 记录结果到汇总字典
        results_summary[exp_name] = {
            "num_layers": num_layers,
            "HR@10": hr_results[10], "NDCG@10": ndcg_results[10],
            "HR@20": hr_results[20], "NDCG@20": ndcg_results[20]
        }
    
    # 5. 打印并保存所有实验的最终结果汇总
    print("\n\n--- Depth Exploration Summary ---")
    if not results_summary:
        print("No experiments were run.")
    else:
        summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
        # summary_df = summary_df.sort_values(by="num_layers", ascending=True) # 按层数排序
        print(summary_df)
        
        summary_save_path = os.path.join(results_dir, 'depth_exploration_summary.csv')
        summary_df.to_csv(summary_save_path)
        print(f"\nDepth exploration summary saved to {summary_save_path}")