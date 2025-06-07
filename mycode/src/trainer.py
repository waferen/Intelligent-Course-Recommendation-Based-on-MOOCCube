import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

def train_model(model, edge_index, train_interactions,
                user_ids_global_range, course_ids_global_range,
                num_epochs, batch_size, learning_rate, device):
    """
    训练LightGCN模型，并返回每个Epoch的损失。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    edge_index = edge_index.to(device)

    all_course_global_ids = list(range(course_ids_global_range[0], course_ids_global_range[1]))
    
    print("Building training user-positive-items map...")
    user_positive_items = {}
    for u_id, c_id in tqdm(train_interactions, desc="  Building map"):
        user_positive_items.setdefault(u_id, set()).add(c_id)

    # 新增：用于记录每个Epoch的损失
    epoch_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        random.shuffle(train_interactions)
        
        # 遍历训练交互数据，按批次进行训练
        batch_iterator = tqdm(range(0, len(train_interactions), batch_size), desc=f"Epoch {epoch+1}")
        for i in batch_iterator:
            batch_interactions = train_interactions[i : i + batch_size]
            
            batch_users, batch_pos_courses, batch_neg_courses = [], [], []
            for u_id, pos_c_id in batch_interactions:
                batch_users.append(u_id)
                batch_pos_courses.append(pos_c_id)
                neg_c_id = random.choice(all_course_global_ids)
                while neg_c_id in user_positive_items.get(u_id, set()):
                    neg_c_id = random.choice(all_course_global_ids)
                batch_neg_courses.append(neg_c_id)

            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long).to(device)
            batch_pos_courses_tensor = torch.tensor(batch_pos_courses, dtype=torch.long).to(device)
            batch_neg_courses_tensor = torch.tensor(batch_neg_courses, dtype=torch.long).to(device)

            optimizer.zero_grad()
            
            final_embeddings = model(edge_index)
            
            user_embeddings = final_embeddings[batch_users_tensor]
            pos_course_embeddings = final_embeddings[batch_pos_courses_tensor]
            neg_course_embeddings = final_embeddings[batch_neg_courses_tensor]

            pos_scores = (user_embeddings * pos_course_embeddings).sum(dim=1)
            neg_scores = (user_embeddings * neg_course_embeddings).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 在tqdm进度条上动态显示当前批次的损失
            batch_iterator.set_postfix(loss=loss.item())
        
        avg_epoch_loss = total_loss / (len(train_interactions) / batch_size)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")
    
    print("Training finished.")

    # 返回模型、最终Embedding和损失历史
    return model, final_embeddings, epoch_losses

# evaluate_model 函数保持不变
# ... (将 evaluate_model 函数的完整代码粘贴到这里) ...
def evaluate_model(model, edge_index, final_embeddings, 
                   test_interactions,
                   user_ids_global_range, course_ids_global_range,
                   user_positive_items_train, # 用户在训练集中已交互的课程
                   user_positive_items_test,  # 用户在测试集中已交互的课程
                   k_values=[10, 20], num_neg_samples=99, device='cpu'):
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