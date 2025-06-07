# 基于知识图谱的课程推荐系统

## 项目概述

本项目旨在利用LightGCN推荐算法，并结合MOOCCube数据集中的多源知识图谱信息，构建一个能够为学生推荐课程的智能系统。通过融合用户-课程交互数据与课程、知识点、学校、老师、专业等实体之间的知识图谱关系，我们旨在提升推荐系统的准确性和召回率。

## 数据集说明

数据集位于 `data/` 目录下，包含以下文件：

*   `train.csv`: 训练集的用户-课程交互数据。格式：`user,course`
*   `test.csv`: 测试集的用户-课程交互数据。格式：`user,course`
*   `knowledge-major.csv`: 知识点与专业之间的关系。格式：`knowledge,major`
*   `course-knowledge.csv`: 课程与知识点之间的关系。格式：`course,knowledge`
*   `school-course.csv`: 学校与课程之间的关系。格式：`school,course`
*   `teacher-course.csv`: 老师与课程之间的关系。格式：`teacher,course`
*   `prerequisite_relations.csv`: 知识点之间的先修关系。格式：`knowledge1,knowledge2`

**注意：** 数据中所有实体（用户、课程、知识点、学校、老师、专业）均以唯一ID字符串形式表示（例如：`U_123`, `C_course-v1:...`, `K_数学_微积分`, `S_清华大学`, `T_张三`, `M_计算机科学`）。

## 环境配置

建议使用 `conda` 创建并管理项目环境。

1.  **创建Conda环境**：
    ```bash
    conda env create -f environment.yml
    ```
2.  **激活环境**：
    ```bash
    conda activate course_recommender
    ```
3.  **验证GPU支持** (可选，如果遇到问题可以跳过):
    ```bash
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
    ```

## 代码结构

```
.
├── data/                         # 存放原始数据文件
│   ├── train.csv
│   ├── test.csv
│   ├── knowledge-major.csv
│   ├── course-knowledge.csv
│   ├── school-course.csv
│   ├── teacher-course.csv
│   └── prerequisite_relations.csv
├── src/                          # 核心代码模块
│   ├── __init__.py
│   ├── data_loader.py            # 数据加载、ID映射、图构建
│   ├── model.py                  # LightGCN模型定义
│   └── trainer.py                # 训练循环、BPR损失、评估指标计算
├── main.py                       # 项目主入口，运行消融实验
├── environment.yml               # Conda环境配置文件
├── requirements.txt              # Pip安装的包列表（备用或作为参考）
├── README.md                     # 项目说明文件
└── report/                       # 存放大作业报告及相关图片
    ├── course_recommendation_report.pdf
    └── images/
```

## 如何运行

在激活 `course_recommender` Conda 环境后，在项目根目录下运行 `main.py` 脚本：

```bash
python main.py
```

脚本将依次运行多个消融实验组，并打印每个实验组的HR@10, HR@20, NDCG@10, NDCG@20指标。

## 消融实验

本项目设计了详细的消融实验，以量化不同知识图谱组件对推荐性能的贡献。实验组包括：

1.  **Baseline LightGCN**: 仅使用用户-课程交互数据。
2.  **+ Course-Knowledge**: 在基线基础上，加入课程与知识点关系。
3.  **+ Course-Knowledge + School-Course**: 在上一组基础上，加入学校与课程关系。
4.  **+ Course-Knowledge + School-Course + Teacher-Course**: 在上一组基础上，加入老师与课程关系。
5.  **+ Course-Knowledge + School-Course + Teacher-Course + Knowledge-Major**: 在上一组基础上，加入知识点与专业关系。
6.  **Full KG Model**: 包含所有知识图谱关系 (课程-知识点、学校-课程、老师-课程、知识点先修、知识点-专业)。

实验结果将在控制台输出

## 评价指标

*   **HR@K (Hit Ratio @ K)**: 推荐列表中是否包含用户实际交互过的物品。
*   **NDCG@K (Normalized Discounted Cumulative Gain @ K)**: 衡量推荐列表中命中物品的位置，位置越靠前得分越高。



