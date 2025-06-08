# 版本2 网格调参
import torch
import os
import sys
import pandas as pd
import json
import itertools # 导入itertools来轻松生成笛卡尔积
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
    results_dir = 'results/hyperparameter_tuning_results/' # 为调优结果创建一个新目录
    os.makedirs(results_dir, exist_ok=True) # 确保结果目录存在

    # --- 1. 定义超参数网格 ---
    param_grid = {
        'num_layers': [2, 3, 4],
        'embedding_dim': [32, 64, 128], # 为了节省时间，先不跑256，如果128效果好再加
        'learning_rate': [0.01, 0.005, 0.001]
    }
    
    # 固定的训练参数
    num_epochs = 50
    batch_size = 2048

    # --- 2. 准备数据加载所需的固定配置 ---
    # 我们只对 Full_KG 模型进行调优
    full_kg_config = {
        "include_kg_course_knowledge": True,
        "include_kg_school_course": True,
        "include_kg_teacher_course": True,
        "include_kg_kp_prereq": True,
        "include_kg_knowledge_major": True
    }
    
    # --- 3. 数据只加载一次，因为图结构是固定的 (Full KG) ---
    print("--- Loading and Processing Data for Full KG Model (once) ---")
    num_nodes, edge_index, train_interactions, test_interactions, \
    user_ids_global_range, course_ids_global_range, \
    user_positive_items_train, user_positive_items_test, entity_to_id = \
        load_and_process_data_for_experiment(data_dir, **full_kg_config)
    
    print(f"Data loaded. Graph nodes: {num_nodes}, edges: {edge_index.shape[1] // 2}")
    
    # --- 4. 网格搜索主循环 ---
    results_summary = {}
    
    # 生成所有超参数组合的笛卡尔积
    param_combinations = list(itertools.product(
        param_grid['num_layers'],
        param_grid['embedding_dim'],
        param_grid['learning_rate']
    ))
    
    total_experiments = len(param_combinations)
    print(f"\n--- Starting Grid Search for {total_experiments} Hyperparameter Combinations ---")

    for i, (num_layers, embedding_dim, learning_rate) in enumerate(param_combinations):
        
        # 动态生成实验名称
        exp_name = f"L{num_layers}_D{embedding_dim}_LR{learning_rate}"
        
        print(f"\n--- Running Experiment {i+1}/{total_experiments}: {exp_name} ---")
        
        # 4.1 初始化模型 (每次循环都重新初始化)
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
        
        # 保存模型权重
        model_save_path = os.path.join(exp_results_dir, 'model_weights.pt')
        torch.save(trained_model.state_dict(), model_save_path)
        
        # 保存损失历史
        loss_df = pd.DataFrame({'epoch': range(1, num_epochs + 1), 'loss': epoch_losses})
        loss_save_path = os.path.join(exp_results_dir, 'loss_history.csv')
        loss_df.to_csv(loss_save_path, index=False)
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
            "embedding_dim": embedding_dim,
            "learning_rate": learning_rate,
            "HR@10": hr_results[10], "NDCG@10": ndcg_results[10],
            "HR@20": hr_results[20], "NDCG@20": ndcg_results[20]
        }
    
    # 5. 打印并保存所有实验的最终结果汇总
    print("\n\n--- Grid Search Summary ---")
    if not results_summary:
        print("No experiments were run.")
    else:
        summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
        summary_df = summary_df.sort_values(by="NDCG@10", ascending=False) # 按NDCG@10降序排序
        print(summary_df)
        
        # 保存汇总结果到CSV文件
        summary_save_path = os.path.join(results_dir, 'grid_search_summary.csv')
        summary_df.to_csv(summary_save_path)
        print(f"\nGrid search summary saved to {summary_save_path}")