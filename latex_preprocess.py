import numpy as np
import csv

from utils import transform_to_new_range

"""
SCRIPT TO PREPROCESS DATA FOR LATEX PARALLEL COORDINATES PLOT
FOR NOW WORKS ONLY FOR THREE AXES
"""


dropout = []
lr = []
acc = []

with open('/Users/fryderykkogl/Downloads/wandb_export_2022-06-14T10_02_10.893-04_00.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    headers = None
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            headers = row.copy()
            continue
        dropout.append(float(row[0]))
        lr.append(float(row[1]))
        acc.append(float(row[2]))

print(f"{headers[0]}: {dropout}")
print(f"{headers[1]}: {lr}")
print(f"{headers[2]}: {acc}")


acc = np.asarray(acc)
dropout = np.asarray(dropout)
lr = np.asarray(lr)

acc = transform_to_new_range(acc, 0.84, 0.91)
dropout = transform_to_new_range(dropout, 0.25, 0.7)
lr = transform_to_new_range(lr, 0.0, 0.004)

# 5.00/0.58/0.00,0.51/0.00/4.00,0.00/5.00/5.00
# drop val lr

command = ""
for i in range(len(acc)):
    command += f"{dropout[i]:.2f}/{lr[i]:.2f}/{acc[i]:.2f},"

print(command)
# 0.01/3.26/4.78,7.63/0.90/1.56,0.25/3.44/9.06,2.22/0.95/0.63,0.73/2.47/9.09,2.22/0.86/5.12,2.46/3.56/9.41,8.98/2.08/5.07,4.97/1.08/2.75,4.50/9.64/9.35,
