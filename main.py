import data
import randomforest
import evaluation
from math import sqrt
from random import seed

# Test the random forest algorithm
seed(2)
# load and prepare data
dataset = data.preprocess_data('../heart.csv')
# evaluate algorithm
n_folds = 10
max_depth = 6
min_size = 1
sample_size = 0.7
n_features = int(sqrt(len(dataset[0]) - 1))
for n_trees in [10, 50, 100]:
    scores = evaluation.evaluate_algorithm(dataset, randomforest.random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Number of trees: %d' % n_trees)
    print('Mean error rate: %.3f%%' % scores['error_rate'])
    print('Mean oob error rate: %.3f%%' % scores['oob_error_rate'])
    print('Mean accuracy: %.3f%%' % scores['accuracy'])
    print('Mean precision: %.3f%%' % scores['precision'])
    print('Mean sensitivity: %.3f%%' % scores['sensitivity'])
    print('Mean specificity: %.3f%%' % scores['specificity'])
    print('Mean f1_score: %.3f%%' % scores['f1_score'])
    print('Mean variable importance:')
    print(scores['variable_importance'])
