import decisiontree
import evaluation
import random


# Select a random subsample from the dataset
def subsample(dataset, ratio):
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)
    sample = list()
    n_sample = round(len(dataset_copy) * ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dataset_copy))
        sample.append(dataset_copy.pop(index))
    return sample, dataset_copy


# Make a prediction based on the vote of a set of trees
def predict(trees, row):
    predictions = [decisiontree.predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
# @param train - array of floats - the train data
# @param test - array of floats - the test data
# @param min_size - integer - the minimum number of observations in a node for splitting
# @param sample_size - float between 0 and 1 - the dimension of the random subsample for building a tree
# @param n_trees - integer - the number of trees
# @param n_features - integer - the number of features that are selected randomly when building a tree
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    variable_importance = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    oob_error_rate = 0
    for i in range(n_trees):
        sample, oob_data = subsample(train, sample_size)
        tree = decisiontree.build_tree(sample, max_depth, min_size, n_features, variable_importance)
        oob_error_rate += evaluation.oob_error_rate(tree, oob_data)
        trees.append(tree)
    predictions = [predict(trees, row) for row in test]
    for i in range(len(variable_importance)):
        variable_importance[i] /= n_trees
    oob_error_rate /= n_trees
    return (predictions, variable_importance, oob_error_rate)
