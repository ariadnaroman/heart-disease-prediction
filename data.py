from csv import reader
from sklearn import preprocessing


# Read CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r', encoding='utf-8-sig') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert attribute (dataset column) from string to float
def string_to_float(dataset, column_index):
    for row in dataset:
        row[column_index] = float(row[column_index].strip())


# Normalize dataset
def normalize(dataset):
    input_features = [row[:-1] for row in dataset]  # remove target column
    normalized_dataset = list(preprocessing.normalize(input_features, axis=0))
    normalized_dataset = [list(row) for row in normalized_dataset]
    for i, row in enumerate(dataset):
        normalized_dataset[i].append(dataset[i][-1])  # add target column back
    return normalized_dataset


# Convert target column from string to integer
def string_to_int(dataset, column_index):
    class_values = [row[column_index] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column_index] = lookup[row[column_index]]
    return lookup


# Preprocess input data for entering the algorithm
def preprocess_data(filename):
    dataset = load_csv(filename)
    # convert string attributes to floats
    for i in range(0, len(dataset[0]) - 1):
        string_to_float(dataset, i)
    # convert class column to integers
    string_to_int(dataset, len(dataset[0]) - 1)
    # normalize dataset
    dataset = normalize(dataset)
    return dataset
