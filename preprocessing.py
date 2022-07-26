import csv
import os


def preprocess_data(input_path, output_path):
    for file_name in (x for x in os.listdir(input_path) if not x[0] == '.'):
        with open(os.path.join(input_path, file_name)) as file:
            csv_reader = csv.reader(file)
            data = []
            for row in csv_reader:
                data.append(float(row[0]))

        data = sorted(data)
        with open(os.path.join(output_path, file_name), 'w') as output:
            csv_writer = csv.writer(output)
            for item in data[(len(data) // 2 - 50):(len(data) // 2 + 50)]:
                csv_writer.writerow([item])
