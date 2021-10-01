import csv
from collections import defaultdict

d = defaultdict(list)

with open("/home/longlh/hard_2/PycharmProjects/roi_segmentation/testing_data_lesion_metadata.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        d[row[0]].append(int(row[3]))

small = []
medium = []
large = []


for k, v in d.items():
    for e in v:
        if e > 500:
            large.append(k)
            break
    if k not in large:
        for e in v:
            if e > 50:
                medium.append(k)
                break
    if k not in medium and k not in large:
        small.append(k)

print(len(small))
print(len(medium))
print(len(large))

with open('/home/longlh/hard_2/PycharmProjects/roi_segmentation/small_lesion_dataset.csv',
          mode='w') as lesion_file:
    lesion_writer = csv.writer(lesion_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for e in small:
        lesion_writer.writerow([e, 2])

with open('/home/longlh/hard_2/PycharmProjects/roi_segmentation/medium_lesion_dataset.csv',
          mode='w') as lesion_file:
    lesion_writer = csv.writer(lesion_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for e in medium:
        lesion_writer.writerow([e, 2])

with open('/home/longlh/hard_2/PycharmProjects/roi_segmentation/large_lesion_dataset.csv',
          mode='w') as lesion_file:
    lesion_writer = csv.writer(lesion_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for e in large:
        lesion_writer.writerow([e, 2])
# for e in small:
#     print(f"small: {e}")
# for e in medium:
#     print(f"medium: {e}")
# for e in large:
#     print(f"large: {e}")