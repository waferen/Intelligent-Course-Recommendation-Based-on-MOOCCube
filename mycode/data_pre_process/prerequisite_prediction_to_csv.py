import json
import csv

# 输入和输出文件路径
json_path = 'MOOCCube/重要数据/prerequisite_prediction.json'
csv_path = 'data/prerequisite_relations.csv'

relations = []

# 读取json文件
with open(json_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # 只保留label为1的先修关系
        if data['label'] == 1:
            relations.append([data['c1'], data['c2']])

# 写入csv文件
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['knowledge1', 'knowledge2'])
    writer.writerows(relations)

print(f'已输出{len(relations)}条label=1的先修关系到{csv_path}')
