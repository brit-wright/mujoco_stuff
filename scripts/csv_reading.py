import csv
path = []
path_nodes_path = '/home/brittany/IRIS_env_drake/maze2_results.csv'
with open(path_nodes_path) as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        path.append([float(row[0]), float(row[1])])

print(path)