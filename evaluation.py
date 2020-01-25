import random
import decisiontree


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate error rate based on confusion matrix
def error_rate(cm):
    return (cm['false_positive'] + cm['false_negative']) / float(cm['true_positive'] + cm['true_negative'] + cm['false_negative'] + cm['false_positive']) * 100.0


# Calculate accuracy percentage based on confusion matrix
def accuracy(cm):
    return (cm['true_positive'] + cm['true_negative']) / float(cm['true_positive'] + cm['true_negative'] + cm['false_negative'] + cm['false_positive']) * 100.0


# Calculate precision percentage based on confusion matrix
def precision(cm):
    return cm['true_positive'] / float(cm['true_positive'] + cm['false_positive']) * 100.0


# Calculate sensitivity (recall) based on confusion matrix
def sensitivity(cm):
    return cm['true_positive'] / float(cm['true_positive'] + cm['false_negative']) * 100.0


# Calculate specificity based on confusion matrix
def specificity(cm):
    return cm['true_negative'] / float(cm['true_negative'] + cm['false_positive']) * 100.0


# Calculate F1 score based on confusion matrix
def f1_score(cm):
    return (2 * cm['true_positive']) / float(2 * cm['true_positive'] + cm['false_positive'] + cm['false_negative']) * 100.0


# Calculate confusion matrix
def confusion_matrix(actual, predicted):
    tp = 0  # true positive
    tn = 0  # true negative
    fp = 0  # false positive
    fn = 0  # false negative
    for i in range(len(actual)):
        if actual[i] == predicted[i] & actual[i] == 1:
            tp += 1
        elif actual[i] == predicted[i] & actual[i] == 0:
            tn += 1
        elif actual[i] != predicted[i] & actual[i] == 1:
            fn += 1
        elif actual[i] != predicted[i] & actual[i] == 0:
            fp += 1
    return {'true_positive': tp,
            'true_negative': tn,
            'false_negative': fn,
            'false_positive': fp}


def oob_error_rate(tree, oob_data):
    predicted = [decisiontree.predict(tree, row) for row in oob_data]
    actual = [row[-1] for row in oob_data]
    cm = confusion_matrix(actual, predicted)
    return error_rate(cm)


# Evaluate an algorithm using k-fold cross validation
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    mean_err = 0
    mean_acc = 0
    mean_prec = 0
    mean_sens = 0
    mean_spec = 0
    mean_f1 = 0
    mean_oob_err = 0
    mean_variable_importance = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, variable_importance_fold, oob_error = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        cm = confusion_matrix(actual, predicted)
        mean_err += error_rate(cm)
        mean_acc += accuracy(cm)
        mean_prec += precision(cm)
        mean_sens += sensitivity(cm)
        mean_spec += specificity(cm)
        mean_f1 += f1_score(cm)
        mean_oob_err += oob_error
        for i in range(len(mean_variable_importance)):
            mean_variable_importance[i] += variable_importance_fold[i]
    for i in range(len(mean_variable_importance)):
        mean_variable_importance[i] /= n_folds
    return {
        'error_rate': mean_err / float(len(folds)),
        'oob_error_rate': mean_oob_err / float(len(folds)),
        'accuracy': mean_acc / float(len(folds)),
        'precision': mean_prec / float(len(folds)),
        'sensitivity': mean_sens / float(len(folds)),
        'specificity': mean_spec / float(len(folds)),
        'f1_score': mean_f1 / float(len(folds)),
        'variable_importance': mean_variable_importance
    }
