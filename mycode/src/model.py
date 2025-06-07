import torch
import torch.nn as nn
import torch.nn.functional as F
# 从 PyTorch Geometric 导入辅助函数
from torch_geometric.utils import add_self_loops, degree

class LightGCN(nn.Module):
    """
    LightGCN模型实现。
    LightGCN通过移除GCN中的特征变换和非线性激活函数，简化图卷积过程。
    它在异构图上进行消息传播，学习所有节点的Embedding。
    """
    def __init__(self, num_nodes, embedding_dim, num_layers):
        """
        初始化LightGCN模型。

        Args:
            num_nodes (int): 图中所有节点的总数（用户+课程+知识点+学校+老师+专业）。
            embedding_dim (int): 节点Embedding的维度。
            num_layers (int): LightGCN的消息传播层数。
        """
        super(LightGCN, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 为所有节点初始化Embedding向量
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        # 使用Xavier均匀分布初始化Embedding权重，有助于训练稳定
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index):
        """
        执行LightGCN的消息传播过程。

        Args:
            edge_index (torch.LongTensor): 图的边索引，格式为 (2, num_edges)。
                                            包含了所有异构边。

        Returns:
            torch.Tensor: 所有节点的最终学习到的Embedding矩阵，形状为 (num_nodes, embedding_dim)。
        """
        # 1. 添加自循环并标准化邻接矩阵 (LightGCN的A_hat = A + I)
        # add_self_loops 函数会向 edge_index 添加自循环，并返回新的边索引。
        edge_index_norm, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # 计算度矩阵的逆平方根，用于归一化
        row, col = edge_index_norm
        deg = degree(col, self.num_nodes, dtype=row.dtype) # 计算每个节点的度
        deg_inv_sqrt = deg.pow(-0.5) # 度^-0.5
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # 避免对度为0的节点进行无穷大操作

        # 计算归一化因子 (D_hat)^(-1/2) * (D_hat)^(-1/2)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 构建稀疏邻接矩阵。PyTorch Geometric在内部会更高效地处理图结构。
        # 这里手动构建是为了更清晰地展示LightGCN的核心传播步骤。
        adj_matrix = torch.sparse_coo_tensor(
            edge_index_norm, norm, (self.num_nodes, self.num_nodes), 
            dtype=torch.float32, device=self.embedding.weight.device
        )
        
        # 2. 消息传播
        # all_embeddings 列表存储每一层传播后的节点Embedding
        all_embeddings = [self.embedding.weight] # L0 层的Embedding就是初始Embedding

        for layer in range(self.num_layers):
            # E^(l+1) = (Normalized_Adj_Matrix) * E^(l)
            # 即每个节点从其邻居聚合信息
            current_embeddings = torch.sparse.mm(adj_matrix, all_embeddings[-1])
            all_embeddings.append(current_embeddings)

        # 3. 最终Embedding聚合 (所有层级Embedding的平均)
        # LightGCN的最终Embedding是所有层（包括初始层）Embedding的平均。
        # torch.stack 将列表中的张量沿新维度堆叠起来 (num_layers+1, num_nodes, embedding_dim)
        # torch.mean 沿第一个维度（层维度）取平均，得到 (num_nodes, embedding_dim)
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        
        return final_embeddings

    # 辅助方法：在评估阶段，可以根据全局ID范围获取特定类型的Embedding
    # 通常不需要直接调用，因为评估函数直接通过全局ID索引 final_embeddings
    def get_user_course_embeddings(self, final_embeddings, user_ids_global_range, course_ids_global_range):
        user_embeddings = final_embeddings[user_ids_global_range[0]:user_ids_global_range[1]]
        course_embeddings = final_embeddings[course_ids_global_range[0]:course_ids_global_range[1]]
        return user_embeddings, course_embeddings