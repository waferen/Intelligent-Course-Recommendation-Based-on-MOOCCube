import pandas as pd
import torch
import random
import os

# 定义实体类型前缀，用于ID映射和区分
ENTITY_TYPES = {
    'user': 'U_',
    'course': 'C_',
    'knowledge': 'K_', # 知识点现在用 'knowledge'
    'school': 'S_',
    'teacher': 'T_',
    'major': 'M_'
}

def load_and_process_data_for_experiment(
    data_dir, # 数据文件所在的目录
    # 由于prerequisite_relations.csv现在直接是关系，不再有概率阈值
    # prereq_prob_threshold=0.7, 
    
    # 消融实验参数
    include_kg_course_knowledge=True, 
    include_kg_school_course=True,
    include_kg_teacher_course=True,
    include_kg_kp_prereq=True, # 对应 prerequisite_relations.csv
    include_kg_knowledge_major=True
):
    """
    加载所有原始数据，进行全局ID映射，划分训练/测试交互，并根据配置构建图。
    根据新的文件结构和命名约定进行调整。
    """
    # --- 1. 加载所有原始数据 ---
    # 用户-课程交互数据
    df_train_uc = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test_uc = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # 知识图谱关系数据
    df_knowledge_major = pd.read_csv(os.path.join(data_dir, 'knowledge-major.csv'))
    df_course_knowledge = pd.read_csv(os.path.join(data_dir, 'course-knowledge.csv'))
    df_school_course = pd.read_csv(os.path.join(data_dir, 'school-course.csv'))
    df_teacher_course = pd.read_csv(os.path.join(data_dir, 'teacher-course.csv'))
    
    # 知识点先修关系 (现在是CSV文件)
    df_prerequisite_relations = pd.read_csv(os.path.join(data_dir, 'prerequisite_relations.csv'))

    # --- 2. 收集所有唯一实体并分配全局ID ---
    entity_to_id = {}
    id_counter = 0

    def get_global_id(entity_str):
        nonlocal id_counter
        if entity_str not in entity_to_id:
            entity_to_id[entity_str] = id_counter
            id_counter += 1
        return entity_to_id[entity_str]

    # 2.1 遍历所有文件，收集并ID化所有实体
    # 用户ID (user)
    for u in pd.concat([df_train_uc['user'], df_test_uc['user']]).unique():
        get_global_id(u)
    # 课程ID (course)
    for c in pd.concat([df_train_uc['course'], df_test_uc['course'], 
                         df_course_knowledge['course'], df_school_course['course'], 
                         df_teacher_course['course']]).unique():
        get_global_id(c)
    # 知识点ID (knowledge)
    for k_node in pd.concat([df_knowledge_major['knowledge'], df_course_knowledge['knowledge'], 
                             df_prerequisite_relations['knowledge1'], df_prerequisite_relations['knowledge2']]).unique():
        get_global_id(k_node)
    # 学校ID (school)
    for s in df_school_course['school'].unique():
        get_global_id(s)
    # 老师ID (teacher)
    for t in df_teacher_course['teacher'].unique():
        get_global_id(t)
    # 专业ID (major)
    for m in df_knowledge_major['major'].unique():
        get_global_id(m)

    num_nodes = id_counter # 总节点数

    # 记录用户和课程的全局ID范围（用于评估）
    user_ids = sorted([v for k,v in entity_to_id.items() if k.startswith(ENTITY_TYPES['user'])])
    course_ids = sorted([v for k,v in entity_to_id.items() if k.startswith(ENTITY_TYPES['course'])])

    user_ids_global_range = (min(user_ids), max(user_ids) + 1) if user_ids else (0,0)
    course_ids_global_range = (min(course_ids), max(course_ids) + 1) if course_ids else (0,0)

    # --- 3. 构建训练集和测试集的用户-课程交互（已ID化） ---
    train_interactions = []
    user_positive_items_train = {}
    for _, row in df_train_uc.iterrows():
        u_id = get_global_id(row['user'])
        c_id = get_global_id(row['course'])
        train_interactions.append((u_id, c_id))
        user_positive_items_train.setdefault(u_id, set()).add(c_id)

    test_interactions = []
    user_positive_items_test = {}
    for _, row in df_test_uc.iterrows():
        u_id = get_global_id(row['user'])
        c_id = get_global_id(row['course'])
        test_interactions.append((u_id, c_id))
        user_positive_items_test.setdefault(u_id, []).append(c_id)

    # --- 4. 构建用于LightGCN的图的 edge_index ---
    edges = []
    # 4.1 添加训练集的用户-课程交互边 (双向)
    for u_id, c_id in train_interactions:
        edges.append((u_id, c_id))
        edges.append((c_id, u_id))

    # 4.2 根据消融实验配置添加知识图谱边 (双向)
    if include_kg_course_knowledge: # course-knowledge.csv
        for _, row in df_course_knowledge.iterrows():
            c_id = get_global_id(row['course'])
            k_id = get_global_id(row['knowledge'])
            edges.append((c_id, k_id))
            edges.append((k_id, c_id))

    if include_kg_school_course: # school-course.csv
        for _, row in df_school_course.iterrows():
            s_id = get_global_id(row['school'])
            c_id = get_global_id(row['course'])
            edges.append((s_id, c_id))
            edges.append((c_id, s_id))

    if include_kg_teacher_course: # teacher-course.csv
        for _, row in df_teacher_course.iterrows():
            t_id = get_global_id(row['teacher'])
            c_id = get_global_id(row['course'])
            edges.append((t_id, c_id))
            edges.append((c_id, t_id))

    if include_kg_kp_prereq: # prerequisite_relations.csv (新的)
        for _, row in df_prerequisite_relations.iterrows():
            kp1_id = get_global_id(row['knowledge1'])
            kp2_id = get_global_id(row['knowledge2'])
            edges.append((kp1_id, kp2_id))
            edges.append((kp2_id, kp1_id)) # 默认仍为无向边

    if include_kg_knowledge_major: # knowledge-major.csv
        for _, row in df_knowledge_major.iterrows():
            k_id = get_global_id(row['knowledge'])
            m_id = get_global_id(row['major'])
            edges.append((k_id, m_id))
            edges.append((m_id, k_id))

    src_nodes, dst_nodes = zip(*edges)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    return num_nodes, edge_index, train_interactions, test_interactions, \
           user_ids_global_range, course_ids_global_range, \
           user_positive_items_train, user_positive_items_test, entity_to_id