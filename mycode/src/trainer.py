import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm # 用于显示进度条

def train_model(model, edge_index, train_interactions,
                user_ids_global_range, course_ids_global_range,
                num_epochs, batch_size, learning_rate, device):
    """
    训练LightGCN模型。

    Args:
        model (LightGCN): LightGCN模型实例。
        edge_index (torch.LongTensor): 训练图的边索引。
        train_interactions (list): 训练集用户-课程交互列表 [(user_id, course_id), ...]。
        user_ids_global_range (tuple): 用户全局ID范围。
        course_ids_global_range (tuple): 课程全局ID范围。
        num_epochs (int): 训练轮数。
        batch_size (int): 批处理大小。
        learning_rate (float): 学习率。
        device (torch.device): 训练设备（CPU或CUDA）。
    
    Returns:
        tuple: 训练好的模型实例和最终学习到的Embedding矩阵。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device) # 将模型移动到指定设备
    edge_index = edge_index.to(device) # 将图结构也移动到指定设备

    # 获取所有课程的全局ID列表，用于负采样
    all_course_global_ids = list(range(course_ids_global_range[0], course_ids_global_range[1]))
    
    # 构建用户已交互课程的集合，以便快速判断是否已交互 (训练集)
    user_positive_items = {}
    for u_id, c_id in train_interactions:
        user_positive_items.setdefault(u_id, set()).add(c_id)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        total_loss = 0
        
        random.shuffle(train_interactions) # 每次Epoch重新打乱训练交互顺序
        
        # 遍历训练交互数据，按批次进行训练
        for i in tqdm(range(0, len(train_interactions), batch_size), desc=f"Epoch {epoch+1}"):
            batch_interactions = train_interactions[i : i + batch_size]
            
            # 采样当前批次的用户、正样本、负样本
            batch_users = []
            batch_pos_courses = []
            batch_neg_courses = []

            for u_id, pos_c_id in batch_interactions:
                batch_users.append(u_id)
                batch_pos_courses.append(pos_c_id)
                
                # 负采样：随机选择一个用户未在训练集中交互过的课程
                neg_c_id = random.choice(all_course_global_ids)
                while neg_c_id in user_positive_items.get(u_id, set()):
                    neg_c_id = random.choice(all_course_global_ids)
                batch_neg_courses.append(neg_c_id)

            # 将采样的ID列表转换为PyTorch Tensor，并移动到设备
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long).to(device)
            batch_pos_courses_tensor = torch.tensor(batch_pos_courses, dtype=torch.long).to(device)
            batch_neg_courses_tensor = torch.tensor(batch_neg_courses, dtype=torch.long).to(device)

            optimizer.zero_grad() # 清空梯度
            
            # 获取所有节点的Embedding (通过模型的前向传播)
            final_embeddings = model(edge_index)
            
            # 提取当前批次用户、正样本、负样本的Embedding
            user_embeddings = final_embeddings[batch_users_tensor]
            pos_course_embeddings = final_embeddings[batch_pos_courses_tensor]
            neg_course_embeddings = final_embeddings[batch_neg_courses_tensor]

            # 计算点积分数
            pos_scores = (user_embeddings * pos_course_embeddings).sum(dim=1)
            neg_scores = (user_embeddings * neg_course_embeddings).sum(dim=1)

            # 计算BPR损失
            loss = -F.logsigmoid(pos_scores - neg_scores).mean() # BPR loss的公式

            # 反向传播和优化器步骤
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() # 累加损失

        # 打印当前Epoch的平均损失
        print(f"Epoch {epoch+1}, Loss: {total_loss / (len(train_interactions) / batch_size):.4f}")
    
    print("Training finished.")

    return model, final_embeddings # 返回训练好的模型实例和最终Embedding

def evaluate_model(model, edge_index, final_embeddings, # final_embeddings 可以直接传入，或在内部调用 model(edge_index) 获得
                   test_interactions,
                   user_ids_global_range, course_ids_global_range,
                   user_positive_items_train, # 用户在训练集中已交互的课程
                   user_positive_items_test,  # 用户在测试集中已交互的课程
                   k_values=[10, 20], num_neg_samples=99, device='cpu'):
    """
    评估LightGCN模型在测试集上的性能。

    Args:
        model (LightGCN): 训练好的LightGCN模型实例。
        edge_index (torch.LongTensor): 图的边索引（用于获取最终Embedding）。
        final_embeddings (torch.Tensor): 训练好的所有节点的Embedding矩阵。
        test_interactions (list): 测试集中的用户-课程交互列表 [(user_id, course_id), ...]。
        user_ids_global_range (tuple): 用户全局ID范围。
        course_ids_global_range (tuple): 课程全局ID范围。
        user_positive_items_train (dict): 训练集中每个用户已交互课程的集合。
        user_positive_items_test (dict): 测试集中每个用户已交互课程的列表。
        k_values (list): 评估的K值列表，例如 [10, 20]。
        num_neg_samples (int): 评估时每个真实正样本对应的负样本数量。
        device (torch.device): 评估设备。
    
    Returns:
        tuple: 包含平均HR@K和NDCG@K分数的字典。
    """
    model.eval() # 设置模型为评估模式 (禁用 dropout 等)
    
    # 获取所有课程的全局ID列表，用于负采样
    course_start_id, course_end_id = course_ids_global_range
    all_course_global_ids = list(range(course_start_id, course_end_id))

    # 获取在测试集中有实际交互的用户列表，只评估这些用户
    test_users = list(user_positive_items_test.keys())
    
    hr_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}

    print("Starting evaluation...")
    with torch.no_grad(): # 在评估阶段不需要计算梯度，节省内存和时间
        
        for u_id in tqdm(test_users, desc="Evaluating"):
            # 1. 获取该用户在测试集中的所有真实正样本
            positive_courses_in_test = user_positive_items_test.get(u_id, [])
            if not positive_courses_in_test: 
                continue # 如果用户在测试集中没有正样本，则跳过

            # 2. 获取用户在训练集中已交互过的所有课程 (用于负样本排除)
            interacted_courses_train = user_positive_items_train.get(u_id, set())
            
            # 3. 构建候选推荐列表：包含测试集正样本 + 负样本
            candidate_courses_global_ids = []
            candidate_courses_global_ids.extend(positive_courses_in_test) # 加入测试集中的所有正样本

            # 采样负样本
            neg_candidates = []
            # 用户所有已知的交互 (训练集正样本 + 测试集正样本)
            current_user_all_interacted_courses = interacted_courses_train.union(set(positive_courses_in_test))

            while len(neg_candidates) < num_neg_samples:
                sampled_neg_id = random.choice(all_course_global_ids) # 从所有课程中随机选
                # 确保采样到的负样本：不在用户所有已交互课程中
                if sampled_neg_id not in current_user_all_interacted_courses:
                    neg_candidates.append(sampled_neg_id)
            
            candidate_courses_global_ids.extend(neg_candidates)
            random.shuffle(candidate_courses_global_ids) # 打乱顺序，避免任何偏差

            # 将候选课程IDs转换为PyTorch Tensor，并移动到指定设备
            candidate_courses_tensor = torch.tensor(candidate_courses_global_ids, dtype=torch.long).to(device)

            # 4. 预测分数
            user_emb = final_embeddings[u_id].unsqueeze(0) # (1, embedding_dim)
            candidate_embs = final_embeddings[candidate_courses_tensor] # (num_candidates, embedding_dim)
            
            scores = torch.matmul(user_emb, candidate_embs.T).squeeze(0) # (num_candidates,)

            # 5. 排序并获取Top-K推荐列表
            _, top_indices = torch.topk(scores, k=max(k_values))
            predicted_global_ids = candidate_courses_tensor[top_indices].cpu().numpy()

            # 6. 计算HR和NDCG
            for k in k_values:
                # HR@K (Hit Ratio)
                hit = False
                for pos_c_id in positive_courses_in_test:
                    if pos_c_id in predicted_global_ids[:k]: 
                        hit = True
                        break 
                hr_scores[k].append(1 if hit else 0)

                # NDCG@K (Normalized Discounted Cumulative Gain)
                dcg = 0.0 # Discounted Cumulative Gain
                
                # 计算DCG：遍历Top-K推荐列表
                for rank, pred_id in enumerate(predicted_global_ids[:k]):
                    if pred_id in positive_courses_in_test:
                        dcg += 1.0 / np.log2(rank + 2) 
                
                # 计算IDCG (Ideal DCG)：理想情况下的最大DCG
                idcg = 0.0
                num_relevant_in_k = min(len(positive_courses_in_test), k)
                for rank in range(num_relevant_in_k):
                    idcg += 1.0 / np.log2(rank + 2)

                if idcg == 0: 
                    ndcg_scores[k].append(0.0)
                else:
                    ndcg_scores[k].append(dcg / idcg)
    
    # 7. 汇总所有用户的指标
    avg_hr = {k: np.mean(hr_scores[k]) for k in k_values}
    avg_ndcg = {k: np.mean(ndcg_scores[k]) for k in k_values}

    print("\nEvaluation Results:")
    for k in k_values:
        print(f"HR@{k}: {avg_hr[k]:.4f}")
        print(f"NDCG@{k}: {avg_ndcg[k]:.4f}")

    return avg_hr, avg_ndcg