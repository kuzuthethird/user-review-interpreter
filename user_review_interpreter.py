import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt
from helper import *

def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    #linear kernel
    if degree == 1:
        #l2 norm with hinge loss
        if penalty == 'l2':
            return SVC(kernel='linear', C=c, class_weight=class_weight)
    #l1 norm with squared hinge loss
    elif penalty == 'l1':
        return LinearSVC(penalty='l1', dual=False, C=c, class_weight=class_weight)
    #quadratic kernel
    elif degree == 2:
        return SVC(kernel='poly', degree=2, C=c, coef0=r, class_weight=class_weight)

def performance(y_true, y_pred, metric='accuracy', labels=[-1, 1]):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
            other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
            and 'specificity')
    Returns:
        the performance as an np.float64
    """
    avg = 'binary' if len(labels ) == 2 else 'macro'

    if(metric == 'accuracy'):
        #(tp + tn) / n
        return metrics.accuracy_score(y_true, y_pred, labels)
    elif(metric == 'f1_score'):
        #2 * (precision * sensitivity ) / (precision + sensitivity)
        return metrics.f1_score(y_true, y_pred, labels, average=avg)
    elif(metric == 'auroc'):
        #area under receiver operating curve
        #tells how good we are at guessing true positive vs false positive
        return metrics.roc_auc_score(y_true, y_pred)
    elif(metric == 'precision'):
        #tp / (tp + fp)
        return metrics.precision_score(y_true, y_pred, labels)
    elif(metric == 'sensitivity'):
        #sensitivity and recall reference the same performance metric
        #tp / (tp + fn)
        return metrics.recall_score(y_true, y_pred, labels)
    else:
        #if(metric == "specificity"):
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels).ravel()
        #tn / (tn + fp)
        return np.divide(tn, np.add(tn, fp, dtype=np.float64), dtype=np.float64)

def cv_performance(clf, X, y, k=5, metric='accuracy', labels=[-1, 1]):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
        other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
        and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    #create the kfolds cross validator object
    kfold = StratifiedKFold(n_splits=k)

    #array to keep track of each performance score
    scores = np.zeros(k, np.float64)

    #keep track of index in score array
    score_index = 0

    #iterate through each training and test index in the kfold object
    for train_index, test_index in kfold.split(X, y):
        #get the current training fold
        X_train_fold = X[train_index]
        y_train_fold = y[train_index]

        #get the current test fold
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]

        #find an appropriate fit given training fold
        clf.fit(X_train_fold, y_train_fold)

        #auroc uses decision_function, every other metric uses predict
        y_pred = clf.decision_function(X_test_fold) if metric == 'auroc' else clf.predict(X_test_fold)

        #save the appropriate score in the scores array
        scores[score_index] = performance(y_test_fold, y_pred, metric, labels)

        #increment to the next index
        score_index += 1

    #and return the average performance across all fold splits
    return scores.mean()

def select_param_linear(X, y, k=5, metric='accuracy', C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of labels {1, 0, -1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
            other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
            and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    #keep track the best C and the max performance it generates for the specified metric
    best_c = C_range[0]
    max_perf = np.float64('-inf')

    #labels for multiclass classification
    labels=[-1, 0, 1]

    #go through all C values
    for c in C_range:
        #create our linear SVC with new c
        clf = select_classifier(penalty=penalty, c=c, degree=1, r=0.0, class_weight='balanced')

        #test performance of specified metric with SVC that has new c
        current_perf = cv_performance(clf, X, y, k=5, metric=metric, labels=labels)

        #check if the current performance is better than the max performance
        if(current_perf > max_perf):
            #save the new C that causes better performance
            best_c = c
            max_perf = current_perf

    return best_c, max_perf

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
            and d is the number of features
        y: (n,) array of labels {1, 0, -1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
        other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
            and 'specificity')
        parameter_values: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter value(s) for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance
    """
    #keep track the best C and the max performance it generates for the specified metric
    best_params = (param_range[0][0], param_range[0][1])
    max_perf = np.float64('-inf')

    #labels for multiclass classification
    labels=[-1, 0, 1];

    #go through each c, r pairing
    for params in param_range:
        c = params[0]
        r = params[1]

        #create quadratic kernel classifier and test the performance for chosen metric
        clf = select_classifier(c=c, degree=2, r=r, class_weight='balanced')
        current_perf = cv_performance(clf, X, y, k=5, metric=metric, labels=labels)

        #assign max performance appropriately
        if(current_perf > max_perf):
            best_params = (c, r)
            max_perf = current_perf

    return best_params, max_perf

def select_param_rbf(X, y):
    """
    Sweeps different settings for the hyperparameters of an rbf-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y. This function has some
    preset variables that I describe right below ---v

    NOTE: After researching SVMs and kernels, I found that the the rbf kernel
    is the most flexible kernel and usually better-performing than others with multiclass
    classification. The rbf kernel get even stronger with one-vs-one method of multiclass
    classification, as compared to one-vs-all, when dealing multiple labels. Also, I found
    out that f1_score is a good metric to use in this situation as well. And 5-fold cv takes
    a while to run, so I'm going with 3-fold cv.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
            and d is the number of features
        y: (n,) array of labels {1, 0, -1}
    Returns:
        The parameter value(s) for a quadratic-kernel SVM that maximize
        the average 3-fold CV performance and the best performnce
    """
    #our c and gamma ranges to test
    C_range = np.float_power(np.repeat(np.float(10 ), 4), range(-1, 3), dtype=np.float64)
    gamma_range = np.float_power(np.repeat(np.float(10 ), 3), range(-2, 1), dtype=np.float64)

    #the best parameteres and their performance
    best_params = (C_range[0], gamma_range[0])
    best_perf = np.float64('-inf')

    #labels for multiclass classification
    labels=[-1, 0, 1]

    #go through each c value
    for c in C_range:
        #go through each gamma value
        for gamma in gamma_range:
            #create our classifier with the according c and gamma values, test it's 3fold f1_score performance
            clf = SVC(kernel='rbf', C=c, gamma=gamma, decision_function_shape='ovo')
            current_perf = cv_performance(clf, X, y, k=3, metric='f1_score', labels=labels )

            #save best performances accordingly
            if(current_perf > best_perf ):
                best_params = (c, gamma );best_perf = current_perf

    return best_perf, best_params

def plot_weight(X, y, penalty, C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """
    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2-penalty, degree=1 SVM to the data (X, y)

    for c in C_range:
        clf = select_classifier(penalty=penalty, c=c, degree=1, r=0.0, class_weight='balanced')
        clf.fit(X, y)
        l0_norm = np.count_nonzero(clf.coef_)
        norm0.append(l0_norm )

    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-' + penalty + '_penalty.png')
    plt.savefig('Norm-' + penalty + '_penalty.png')
    plt.close()

def main():
    #list of metrics
    #perf_metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity'];

    # Read multiclass data
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    #create the SVC classifier
    clf = SVC(kernel='rbf', C=np.float64(1.0), gamma=np.float64(1.0), decision_function_shape='ovo')

    #let the SVC classifier learn from the data
    clf.fit(multiclass_features, multiclass_labels)

    #make our predictions
    heldout_pred = clf.predict(heldout_features)

    #save our predictions to a file
    generate_challenge_labels(heldout_pred, 'test')

if __name__ == '__main__':
    main()
