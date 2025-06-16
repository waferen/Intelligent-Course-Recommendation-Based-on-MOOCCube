import pandas as pd
import torch
import random
import os
from tqdm import tqdm # 导入 tqdm 库

# 定义实体类型前缀，用于ID映射和区分
ENTITY_TYPES = {
    'user': 'U_',
    'course': 'C_',
    'knowledge': 'K_',
    'school': 'S_',
    'teacher': 'T_',
    'major': 'M_'
}

def load_and_process_data_for_experiment(
    data_dir, 
    include_kg_course_knowledge=True, 
    include_kg_school_course=True,
    include_kg_teacher_course=True,
    include_kg_kp_prereq=True,
    include_kg_knowledge_major=True
):
    """
    【修正版】加载数据，严格分离训练和测试，避免数据泄露。
    """
    # --- 1. 加载所有原始数据 ---
    print("Step 1: Loading raw data files...")
    df_train_uc = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test_uc = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_knowledge_major = pd.read_csv(os.path.join(data_dir, 'knowledge-major.csv'))
    df_course_knowledge = pd.read_csv(os.path.join(data_dir, 'course-knowledge.csv'))
    df_school_course = pd.read_csv(os.path.join(data_dir, 'school-course.csv'))
    df_teacher_course = pd.read_csv(os.path.join(data_dir, 'teacher-course.csv'))
    df_prerequisite_relations = pd.read_csv(os.path.join(data_dir, 'prerequisite_relations.csv'))
    print("Data files loaded successfully.")

    # --- 2. 【核心修正】只基于训练集和KG数据构建全局ID映射 ---
    print("\nStep 2: Building global entity-to-ID mapping (Train+KG only)...")
    entity_to_id = {}
    id_counter = 0

    def get_global_id(entity_str):
        nonlocal id_counter
        if entity_str not in entity_to_id:
            entity_to_id[entity_str] = id_counter
            id_counter += 1
        return entity_to_id[entity_str]

    # 【修正】只使用训练集和KG数据来收集实体
    all_entities = [
        ("Users (from Train)", df_train_uc['user'].unique()),
        ("Courses (from Train & KG)", pd.concat([df_train_uc['course'], 
                                                 df_course_knowledge['course'], df_school_course['course'], 
                                                 df_teacher_course['course']]).unique()),
        ("Knowledge Points", pd.concat([df_knowledge_major['knowledge'], df_course_knowledge['knowledge'], 
                                        df_prerequisite_relations['knowledge1'], df_prerequisite_relations['knowledge2']]).unique()),
        ("Schools", df_school_course['school'].unique()),
        ("Teachers", df_teacher_course['teacher'].unique()),
        ("Majors", df_knowledge_major['major'].unique())
    ]
    
    for name, entities in all_entities:
        for entity in tqdm(entities, desc=f"  Processing {name}"):
            get_global_id(entity)

    num_nodes = id_counter
    print(f"Global ID mapping built. Total unique nodes in Train+KG: {num_nodes}")

    # 【修正】记录ID范围的逻辑不变，但现在只基于训练集和KG实体
    user_ids = sorted([v for k,v in entity_to_id.items() if k.startswith(ENTITY_TYPES['user'])])
    course_ids = sorted([v for k,v in entity_to_id.items() if k.startswith(ENTITY_TYPES['course'])])
    user_ids_global_range = (min(user_ids), max(user_ids) + 1) if user_ids else (0,0)
    course_ids_global_range = (min(course_ids), max(course_ids) + 1) if course_ids else (0,0)

    # --- 3. 【核心修正】处理训练和测试交互，并过滤掉测试集中的未知实体 ---
    print("\nStep 3: Processing train and test interactions...")
    # 处理训练集
    train_interactions = []
    user_positive_items_train = {}
    for _, row in tqdm(df_train_uc.iterrows(), total=df_train_uc.shape[0], desc="  Processing train.csv"):
        u_id = get_global_id(row['user'])
        c_id = get_global_id(row['course'])
        train_interactions.append((u_id, c_id))
        user_positive_items_train.setdefault(u_id, set()).add(c_id)

    # 处理测试集，并过滤未知实体
    test_interactions = []
    user_positive_items_test = {}
    original_test_count = df_test_uc.shape[0]
    for _, row in tqdm(df_test_uc.iterrows(), total=original_test_count, desc="  Processing test.csv (filtering unseen)"):
        # 只有当用户和课程都存在于ID映射中（即在训练集或KG中出现过）时，才保留该测试交互
        if row['user'] in entity_to_id and row['course'] in entity_to_id:
            u_id = get_global_id(row['user'])
            c_id = get_global_id(row['course'])
            test_interactions.append((u_id, c_id))
            user_positive_items_test.setdefault(u_id, []).append(c_id)
            
    filtered_test_count = len(test_interactions)
    print(f"Interactions processed. Original test interactions: {original_test_count}, Filtered test interactions: {filtered_test_count}")

    # --- 4. 构建用于LightGCN的图的 edge_index ---
    print("\nStep 4: Building graph edges...")
    edges = []
    # 4.1 添加训练集的用户-课程交互边 (双向)
    for u_id, c_id in tqdm(train_interactions, desc="  Adding user-course edges"):
        edges.append((u_id, c_id))
        edges.append((c_id, u_id))

    # 4.2 根据消融实验配置添加知识图谱边 (双向)
    if include_kg_course_knowledge:
        for _, row in tqdm(df_course_knowledge.iterrows(), total=df_course_knowledge.shape[0], desc="  Adding course-knowledge edges"):
            c_id = get_global_id(row['course'])
            k_id = get_global_id(row['knowledge'])
            edges.append((c_id, k_id))
            edges.append((k_id, c_id))

    if include_kg_school_course:
        for _, row in tqdm(df_school_course.iterrows(), total=df_school_course.shape[0], desc="  Adding school-course edges"):
            s_id = get_global_id(row['school'])
            c_id = get_global_id(row['course'])
            edges.append((s_id, c_id))
            edges.append((c_id, s_id))

    if include_kg_teacher_course:
        for _, row in tqdm(df_teacher_course.iterrows(), total=df_teacher_course.shape[0], desc="  Adding teacher-course edges"):
            t_id = get_global_id(row['teacher'])
            c_id = get_global_id(row['course'])
            edges.append((t_id, c_id))
            edges.append((c_id, t_id))

    if include_kg_kp_prereq:
        for _, row in tqdm(df_prerequisite_relations.iterrows(), total=df_prerequisite_relations.shape[0], desc="  Adding prerequisite edges"):
            kp1_id = get_global_id(row['knowledge1'])
            kp2_id = get_global_id(row['knowledge2'])
            edges.append((kp1_id, kp2_id))
            edges.append((kp2_id, kp1_id))

    if include_kg_knowledge_major:
        for _, row in tqdm(df_knowledge_major.iterrows(), total=df_knowledge_major.shape[0], desc="  Adding knowledge-major edges"):
            k_id = get_global_id(row['knowledge'])
            m_id = get_global_id(row['major'])
            edges.append((k_id, m_id))
            edges.append((m_id, k_id))
            
    print("Graph edges built. Converting to tensor...")
    src_nodes, dst_nodes = zip(*edges)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    print("Data loading and processing complete.")

    return num_nodes, edge_index, train_interactions, test_interactions, \
           user_ids_global_range, course_ids_global_range, \
           user_positive_items_train, user_positive_items_test, entity_to_id

