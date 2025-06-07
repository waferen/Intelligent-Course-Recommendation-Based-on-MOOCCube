import csv

input_file = 'MOOCCube\重要数据\\teacher-course.json'
output_file = 'data/teacher-course.csv'

seen = set()

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['teacher', 'course_id'])
    for line in infile:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            teacher, course_id = parts
            key = (teacher, course_id)
            if key not in seen:
                writer.writerow([teacher, course_id])
                seen.add(key) 