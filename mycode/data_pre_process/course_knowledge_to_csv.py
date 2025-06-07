import csv

input_file = 'MOOCCube/重要数据/course-concept.json'
output_file = 'data/course-knowledge.csv'

seen = set()

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['course_id', 'knowledge'])
    for line in infile:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            course_id = parts[0]
            concept = parts[1]
            if concept.startswith('K_'):
                try:
                    _, knowledge, _ = concept.split('_', 2)
                    key = (course_id, knowledge)
                    if key not in seen:
                        writer.writerow([course_id, knowledge])
                        seen.add(key)
                except ValueError:
                    continue
