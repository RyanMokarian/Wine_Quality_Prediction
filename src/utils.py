import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm, preprocessing


def read_data_to_dataframe(path):
    print(os.path.abspath(__file__))
    return pd.read_csv(path)


@staticmethod
def split_data(df):
    data = df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def apply_one_hot(df, column_name, prefix):
    return pd.concat([df, pd.get_dummies(df[column_name], prefix=prefix)], axis=1)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def classifier_kfold_validation(df, clf):
    """
    RandomForestClassifier
    :param df:
    :param clf:
    :return:
    """
    data = df.to_numpy()
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    scores = cross_val_score(clf, X, y, cv=5)
    print("Scores: " + str(scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def classifier_kfoldn(df_gender):
    """
    RandomForestClassifier
    Scores: [0.75892401 0.76454945 0.76577682 0.76639051 0.76237725]
    Accuracy: 0.76 (+/- 0.01)

    red wine RandomForestClassifier(max_depth=40, n_estimators=100, verbose=True) test acc: 0.675
    white wine RandomForestClassifier(max_depth=30, n_estimators=25, verbose=True) test acc: 0.6612244897959184

    red wine GradientBoostingClassifier(learning_rate=0.4, n_estimators=300, verbose=True) test acc: 0.640625
    white wine GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, verbose=True) test acc: 0.6387755102040816

    :param df_gender:
    :param clf:
    :return:
    """
    data = df_gender.to_numpy()
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Error rate: " + str(accuracy_score(y_test, y_pred)))


def run_exhaustive_search(clf, df, parameter_space):
    """
    used the code from:
    https://datascience.stackexchange.com/questions/36049/
    how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
    :param clf:
    :param df:
    :param parameter_space:
    :return:
    """
    data = df.to_numpy()
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_true, y_pred = y_test, clf.predict(X_test)

    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_true, y_pred))



def classifier_learn(X, y):
<<<<<<< HEAD
    x, x_test, y, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, train_size=0.75)
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))
=======
    x, x_test, y, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=0)
    # clf = RandomForestClassifier(max_depth=30, n_estimators=25, verbose=True)
    # clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # print(accuracy_score(y_test, y_pred))
>>>>>>> d80ec5f481d8629d91a4289e494b026a478d9bf3

    test_acc = []
    train_acc = []
    for e in [0.01, 0.1, 1, 10, 100, 1000]:
        # Kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        # Cs = [0.01, 0.1, 1, 10, 100, 1000]
        # SVM default parameters: C=1.0, Kernel= 'rbf'
        clf = svm.SVC(kernel='rbf', C=e)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_cv)
        train_pred = clf.predict(x_train)
        test_acc.append(accuracy_score(y_cv, y_pred))
        train_acc.append(accuracy_score(y_train, train_pred))
    print("Test Acc rate: " + str(test_acc))
    print("Train Acc rate: " + str(train_acc))
    return train_acc, test_acc

def svm_estimation(df):
    data = df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("SVM acc: ", accuracy_score(y_test, y_pred))