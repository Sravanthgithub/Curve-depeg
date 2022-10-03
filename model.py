from data import prepare_price
from plot import plot_predictions, plot_results_confusion_matrix
from utils import cv_split, sort_results
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import f1_score

plt.ion()


def search_cls(min_f1=0.8, threshold=1.):
    """
    {'name': 'logistic_regression',
    'win': 80,
    'f1': 0.8571428571428572,
    'acc': 0.957983193277311}
    """
    cv_ids = [0, 1, 4]
    wins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 70]
    results = []
    for win in wins:
        print(f'window length: {win}')
        X, Y, _ = prepare_price(win=win, threshold=threshold)

        learners = {'logistic_regression': linear_model.LogisticRegression,
                    'naive_bayes': GaussianNB,
                    'SVM': SVC,
                    'decision_tree': DecisionTreeClassifier,
                    'random_forest': ensemble.RandomForestClassifier,
                    'gradient_boosting': ensemble.HistGradientBoostingClassifier}

        for name, learner in learners.items():
            cls = learner()
            y_pred = []
            for cv_i in cv_ids:
                X_valid, y_valid = X[cv_i], Y[cv_i]
                X_train = [x for i, x in enumerate(X) if i != cv_i]
                y_train = [y for i, y in enumerate(Y) if i != cv_i]
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)

                cls.fit(X_train, y_train)
                y_pred.append(cls.predict(X_valid))

            y_pred = np.concatenate(y_pred)
            y_valid = []
            for cv_i in cv_ids:
                y_valid.append(Y[cv_i])
            y_valid = np.concatenate(y_valid)
            f1 = f1_score(y_valid, y_pred > .5)
            acc = (y_valid == (y_pred > .5)).mean()

            print(f'{name}\n\tf1: {f1:.3f} accuracy: {acc:.3f}', f1 > 1/2)

            if f1 > min_f1:
                results.append({'name': name,
                                'win': win,
                                'f1': f1,
                                'acc': acc})

    best, results = sort_results(results)
    return best, results


def fit_predict(threshold=1., plot=True):
    """

    {'name': 'gradient_boosting',
      'win': 3,
      'f1': 0.9045226130653267,
      'acc': 0.9836206896551725},
    """
    learner = 'gradient_boosting'
    win = 3
    cv_ids = [0, 1, 4]

    X, Y, pool_names = prepare_price(win=win, threshold=threshold)

    # cls = ensemble.RandomForestClassifier(n_estimators=n_estimators,
    #                                       criterion=criterion)

    # cls = DecisionTreeClassifier()
    cls = ensemble.HistGradientBoostingClassifier()
    # cls = ensemble.RandomForestClassifier()
    # cls = linear_model.LogisticRegression()
    Y_pred, Y_prob, Y_valid = [], [], []
    for cv_i in cv_ids:
        X_train, y_train, X_valid, y_valid = cv_split(X, Y, cv_i)
        cls.fit(X_train, y_train)
        Y_pred.append(cls.predict(X_valid))
        Y_prob.append(cls.predict_proba(X_valid))
        Y_valid.append(y_valid)

    f1 = f1_score(np.concatenate(Y_valid), np.concatenate(Y_pred))
    acc = (np.concatenate(Y_valid) == np.concatenate(Y_pred)).mean()

    print(f'f1: {f1:.3f}, acc: {acc:.3f}')

    if plot:
        plot_results_confusion_matrix(Y_valid, Y_pred, Y_prob,
                                      threshold, learner, win)
        plot_predictions(Y_valid, Y_pred, Y_prob, threshold,
                         learner, win, pool_names)

    return Y_valid, Y_pred, Y_prob
