from random import randrange


# Split a dataset based on an feature and its split value
def split_dataset_by_feature(feature_index, feature_split_value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[feature_index] < feature_split_value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(clusters, classes):
    # count all instances at split point
    n_instances = float(sum([len(cluster) for cluster in clusters]))
    # sum weighted Gini index for each cluster
    gini = 0.0
    for cluster in clusters:
        size = float(len(cluster))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the cluster based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in cluster].count(class_val) / size
            score += p * p
        # weight the cluster score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def select_best_feature(dataset, n_features, current_gini, variable_importance):
    class_values = list(set(row[-1] for row in dataset))
    feature_index, feature_split_value, feature_gini_score, feature_clusters = 999, 999, 999, None  # best feature
    features = list()
    # pick n random features to test in order to select the one for the node
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    # find the (feature, feature_split_value) pair that result in the lowest gini index
    for index in features:  # for each feature
        for row in dataset:  # for each value of that feature
            # compute the clusters that result from the split
            clusters = split_dataset_by_feature(index, row[index],dataset)
            # compute the gini index for the clusters
            gini = gini_index(clusters, class_values)
            variable_importance[index] += current_gini - gini
            # replace the selected feature if a better one was found
            if gini < feature_gini_score:
                feature_index, feature_split_value, feature_gini_score, feature_clusters = index, row[index], gini, clusters
    return {'index': feature_index, 'value': feature_split_value, 'gini': feature_gini_score, 'clusters': feature_clusters}


# Create a terminal node
def create_terminal_node(cluster):
    outcomes = [row[-1] for row in cluster]
    return max(set(outcomes), key=outcomes.count)


# Recursive function that splits the dataset while building the tree, until getting to terminal nodes
def split_node(node, max_depth, min_size, n_features, depth, variable_importance):
    left, right = node['clusters']
    del (node['clusters'])
    # check if any of the clusters is empty
    if not left or not right:
        node['left'] = node['right'] = create_terminal_node(left + right)
        return
    # check if the max_depth of the tree has been reached
    if depth >= max_depth:
        node['left'], node['right'] = create_terminal_node(left), create_terminal_node(right)
        return
    # check if the min_size of a node has been reached (for left child)
    if len(left) <= min_size:
        node['left'] = create_terminal_node(left)
    else:  # compute the next split
        node['left'] = select_best_feature(left, n_features, node['gini'], variable_importance)
        split_node(node['left'], max_depth, min_size, n_features, depth + 1, variable_importance)
    # check if the min_size of a node has been reached (for right child)
    if len(right) <= min_size:
        node['right'] = create_terminal_node(right)
    else:  # compute the next split
        node['right'] = select_best_feature(right, n_features, node['gini'], variable_importance)
        split_node(node['right'], max_depth, min_size, n_features, depth + 1, variable_importance)


# Build a decision tree
def build_tree(train_data, max_depth, min_size, n_features, variable_importance):
    class_values = list(set(row[-1] for row in train_data))
    current_gini = gini_index([train_data], class_values)
    root = select_best_feature(train_data, n_features, current_gini, variable_importance)
    split_node(root, max_depth, min_size, n_features, 1, variable_importance)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
