import torch
import os
import sys
import pandas as pd
import json
# 将 src 目录添加到 Python 路径中，以便可以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 从 src 模块导入函数和类
from src.data_loader import load_and_process_data_for_experiment
from src.model import LightGCN
from src.trainer import train_model, evaluate_model

if __name__ == "__main__":
    # 检查是否有可用的GPU，并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义数据文件所在的目录
    data_dir = 'data/' 
    results_dir = 'results/'
    os.makedirs(results_dir, exist_ok=True) # 确保结果目录存在
    # 2. 模型和训练超参数
    embedding_dim = 64  # 节点Embedding的维度
    num_layers = 2      # LightGCN的消息传播层数
    num_epochs = 50     # 训练轮数
    batch_size = 2048   # BPR损失计算的批处理大小
    learning_rate = 0.005 # 优化器学习率

    # --- 消融实验的配置字典 ---
    experiment_configs = {
        "Baseline_LightGCN": { # 仅用户-课程交互
            "include_kg_course_knowledge": False,
            "include_kg_school_course": False,
            "include_kg_teacher_course": False,
            "include_kg_kp_prereq": False,
            "include_kg_knowledge_major": False
        },
        "LightGCN_plus_CourseKnowledge": { # 加入课程-知识点
            "include_kg_course_knowledge": True,
            "include_kg_school_course": False,
            "include_kg_teacher_course": False,
            "include_kg_kp_prereq": False,
            "include_kg_knowledge_major": False
        },
        "LightGCN_plus_CourseKnowledge_School": { # 加入学校-课程
            "include_kg_course_knowledge": True,
            "include_kg_school_course": True,
            "include_kg_teacher_course": False,
            "include_kg_kp_prereq": False,
            "include_kg_knowledge_major": False
        },
        "LightGCN_plus_CourseKnowledge_School_Teacher": { # 加入老师-课程
            "include_kg_course_knowledge": True,
            "include_kg_school_course": True,
            "include_kg_teacher_course": True,
            "include_kg_kp_prereq": False,
            "include_kg_knowledge_major": False
        },
        "LightGCN_plus_CourseKnowledge_School_Teacher_KPMajor": { # 加入知识点-专业
            "include_kg_course_knowledge": True,
            "include_kg_school_course": True,
            "include_kg_teacher_course": True,
            "include_kg_kp_prereq": False,
            "include_kg_knowledge_major": True
        },
        "LightGCN_Full_KG": { # 最终模型 (包含所有知识图谱边)
            "include_kg_course_knowledge": True,
            "include_kg_school_course": True,
            "include_kg_teacher_course": True,
            "include_kg_kp_prereq": True,
            "include_kg_knowledge_major": True
        }
    }

    results_summary = {} # 用于存储所有实验的最终结果

    # --- 控制运行单个实验的开关 ---
    # 将此变量设置为你希望运行的实验名称（键），例如 "LightGCN_Full_KG"
    # 如果设置为 None 或 ""，则会运行 'experiment_configs' 中定义的所有实验。
    # run_specific_experiment = "LightGCN_Full_KG" # 默认先跑完整模型
    run_specific_experiment = "Baseline_LightGCN" # 也可以先跑Baseline
    # run_specific_experiment = None # 设置为 None 会运行所有实验

    if run_specific_experiment:
        print(f"Running ONLY the experiment: {run_specific_experiment}\n")
    else:
        print("Running ALL defined experiments.\n")

    for exp_name, config in experiment_configs.items():
        if run_specific_experiment and exp_name != run_specific_experiment:
            continue

        print(f"\n--- Running Experiment: {exp_name} ---")
        
        # 1. 数据加载和处理
        num_nodes, edge_index, train_interactions, test_interactions, \
        user_ids_global_range, course_ids_global_range, \
        user_positive_items_train, user_positive_items_test, entity_to_id = \
            load_and_process_data_for_experiment(data_dir, **config)

        print(f"Graph nodes: {num_nodes}, edges: {edge_index.shape[1] // 2}")
        print(f"Training interactions: {len(train_interactions)}, Test interactions: {len(test_interactions)}")

        # 3. 初始化模型
        model = LightGCN(num_nodes, embedding_dim, num_layers)
        print(f"Model initialized for {exp_name}.")

        # 4. 训练模型
        trained_model, final_embeddings, epoch_losses = train_model(
            model, edge_index, train_interactions,
            user_ids_global_range, course_ids_global_range,
            num_epochs, batch_size, learning_rate, device
        )

        # --- 保存训练过程和结果 ---
        # 创建当前实验的结果子目录
        exp_results_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_results_dir, exist_ok=True)
        
        # 4.1 保存模型权重
        model_save_path = os.path.join(exp_results_dir, 'model_weights.pt')
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"Model weights for '{exp_name}' saved to {model_save_path}")

        # 4.2 保存损失历史
        loss_df = pd.DataFrame({'epoch': range(1, num_epochs + 1), 'loss': epoch_losses})
        loss_save_path = os.path.join(exp_results_dir, 'loss_history.csv')
        loss_df.to_csv(loss_save_path, index=False)
        print(f"Loss history for '{exp_name}' saved to {loss_save_path}")
        
        # 4.3 保存最终Embedding
        embedding_save_path = os.path.join(exp_results_dir, 'final_embeddings.pt')
        torch.save(final_embeddings, embedding_save_path)
        print(f"Final embeddings for '{exp_name}' saved to {embedding_save_path}")
        
        # 4.4 保存ID映射 (非常重要，用于解释Embedding)
        id_map_save_path = os.path.join(exp_results_dir, 'entity_to_id.json')
        with open(id_map_save_path, 'w', encoding='utf-8') as f:
            json.dump(entity_to_id, f, ensure_ascii=False, indent=4)
        print(f"Entity-to-ID map for '{exp_name}' saved to {id_map_save_path}")

        # 5. 评估模型
        hr_results, ndcg_results = evaluate_model(
            trained_model, edge_index, final_embeddings, 
            test_interactions,
            user_ids_global_range, course_ids_global_range,
            user_positive_items_train, user_positive_items_test,
            k_values=[10, 20], device=device
        )
        
        results_summary[exp_name] = {
            "HR@10": hr_results[10], "NDCG@10": ndcg_results[10],
            "HR@20": hr_results[20], "NDCG@20": ndcg_results[20]
        }
    
    # 打印并保存所有实验的最终结果汇总
    print("\n\n--- All Experiments Summary ---")
    if not results_summary:
        print("No experiments were run.")
    else:
        summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
        summary_df.index.name = 'Experiment_Name'
        print(summary_df)
        
        # 保存汇总结果到CSV文件
        summary_save_path = os.path.join(results_dir, 'experiment_summary.csv')
        summary_df.to_csv(summary_save_path)
        print(f"\nExperiment summary saved to {summary_save_path}")