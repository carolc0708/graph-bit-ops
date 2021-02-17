# Python program to demonstrate
# writing to CSV
import csv
import random

# data rows of csv file
mat_len = 32
rows = []
# for i in range(mat_len):
#     temp = []
#     for j in range(mat_len):
#         temp += ['1' if j%2 == 0 else '0']
#     rows.append(temp)

# print(rows)

# better keep this as 1D
for i in range(mat_len):
    for j in range(mat_len):
        rows += ['1' if j == 0 else '0'] # rows += [str(random.randrange(0, 2))]

print(rows)
# name of csv file
filename = "matrix.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the data rows
    csvwriter.writerows(rows)