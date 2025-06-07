import csv

input_file = 'MOOCCube/重要数据/course-concept.json'
output_file = 'data/knowledge-major.csv'

seen = set()

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['knowledge', 'major'])
    for line in infile:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            concept = parts[1]
            if concept.startswith('K_'):
                try:
                    _, knowledge, major = concept.split('_', 2)
                    key = (knowledge, major)
                    if key not in seen:
                        writer.writerow([knowledge, major])
                        seen.add(key)
                except ValueError:
                    continue
