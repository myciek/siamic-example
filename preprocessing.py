import csv
import os
from numpy import mean
from numpy import std


def preprocess_data(input_path, output_path):
    for file_name in (x for x in os.listdir(input_path) if not x[0] == '.'):
        with open(os.path.join(input_path, file_name)) as file:
            csv_reader = csv.reader(file)
            data = []
            for row in csv_reader:
                data.append(float(row[0]))
        data_mean, data_std = mean(data), std(data)
        cut_off = data_std * 3
        data_without_outliers = [x for x in data if x >= data_mean - cut_off and x <= data_mean + cut_off]
        with open(os.path.join(output_path, file_name), 'w') as output:
            csv_writer = csv.writer(output)
            # for item in data[(len(data) // 2 - 1000):(len(data) // 2 + 1000)]:
            for item in data_without_outliers[:3000]:
                csv_writer.writerow([item])
