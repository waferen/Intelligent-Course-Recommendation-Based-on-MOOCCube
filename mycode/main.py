import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_and_process_data_for_experiment
from src.model import LightGCN
from src.trainer import train_model, evaluate_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义数据文件所在的目录
    data_dir = 'data/' 

    # 2. 模型和训练超参数
    embedding_dim = 64
    num_layers = 2
    num_epochs = 50
    batch_size = 2048
    learning_rate = 0.005
    # prereq_prob_threshold # 这个参数现在不再需要

    # --- 消融实验的主循环 ---
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
        "LightGCN_Full_KG": { # 最终模型 (所有知识图谱边都包含)
            "include_kg_course_knowledge": True,
            "include_kg_school_course": True,
            "include_kg_teacher_course": True,
            "include_kg_kp_prereq": True,
            "include_kg_knowledge_major": True
        }
    }

    results_summary = {}

    for exp_name, config in experiment_configs.items():
        print(f"\n--- Running Experiment: {exp_name} ---")
        
        # 1. 数据加载和处理 (根据当前实验配置构建图)
        num_nodes, edge_index, train_interactions, test_interactions, \
        user_ids_global_range, course_ids_global_range, \
        user_positive_items_train, user_positive_items_test, entity_to_id = \
            load_and_process_data_for_experiment(
                data_dir, # 直接传递数据目录
                **config # 将当前实验的配置参数传递给数据加载函数
            )

        print(f"Graph nodes: {num_nodes}, edges: {edge_index.shape[1] // 2} (bidirectional edges count for 2)")
        print(f"Training interactions: {len(train_interactions)}, Test interactions: {len(test_interactions)}")

        # 3. 初始化模型 (每次实验都重新初始化模型)
        model = LightGCN(num_nodes, embedding_dim, num_layers)

        # 4. 训练模型
        trained_model, final_embeddings = train_model(
            model, edge_index, train_interactions,
            user_ids_global_range, course_ids_global_range,
            num_epochs, batch_size, learning_rate, device
        )

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
    
    print("\n\n--- All Experiments Summary ---")
    for exp_name, metrics in results_summary.items():
        print(f"Experiment: {exp_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")