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
    data_dir, # 数据文件所在的目录
    # 消融实验参数
    include_kg_course_knowledge=True, 
    include_kg_school_course=True,
    include_kg_teacher_course=True,
    include_kg_kp_prereq=True,
    include_kg_knowledge_major=True
):
    """
    加载所有原始数据，进行全局ID映射，划分训练/测试交互，并根据配置构建图。
    在耗时部分添加了tqdm进度条。
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

    # --- 2. 收集所有唯一实体并分配全局ID ---
    print("\nStep 2: Building global entity-to-ID mapping...")
    entity_to_id = {}
    id_counter = 0

    def get_global_id(entity_str):
        nonlocal id_counter
        if entity_str not in entity_to_id:
            entity_to_id[entity_str] = id_counter
            id_counter += 1
        return entity_to_id[entity_str]

    # 使用tqdm来显示实体收集的进度
    all_entities = [
        ("Users", pd.concat([df_train_uc['user'], df_test_uc['user']]).unique()),
        ("Courses", pd.concat([df_train_uc['course'], df_test_uc['course'], 
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

    num_nodes = id_counter # 总节点数
    print(f"Global ID mapping built. Total unique nodes: {num_nodes}")

    # 记录用户和课程的全局ID范围
    user_ids = sorted([v for k,v in entity_to_id.items() if k.startswith(ENTITY_TYPES['user'])])
    course_ids = sorted([v for k,v in entity_to_id.items() if k.startswith(ENTITY_TYPES['course'])])
    user_ids_global_range = (min(user_ids), max(user_ids) + 1) if user_ids else (0,0)
    course_ids_global_range = (min(course_ids), max(course_ids) + 1) if course_ids else (0,0)

    # --- 3. 构建训练集和测试集的用户-课程交互（已ID化） ---
    print("\nStep 3: Processing train and test interactions...")
    train_interactions = []
    user_positive_items_train = {}
    for _, row in tqdm(df_train_uc.iterrows(), total=df_train_uc.shape[0], desc="  Processing train.csv"):
        u_id = get_global_id(row['user'])
        c_id = get_global_id(row['course'])
        train_interactions.append((u_id, c_id))
        user_positive_items_train.setdefault(u_id, set()).add(c_id)

    test_interactions = []
    user_positive_items_test = {}
    for _, row in tqdm(df_test_uc.iterrows(), total=df_test_uc.shape[0], desc="  Processing test.csv"):
        u_id = get_global_id(row['user'])
        c_id = get_global_id(row['course'])
        test_interactions.append((u_id, c_id))
        user_positive_items_test.setdefault(u_id, []).append(c_id)
    print("Interactions processed.")

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