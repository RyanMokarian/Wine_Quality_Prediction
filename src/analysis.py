from src.utils import read_data_to_dataframe
import seaborn as sns
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, preprocessing
from src.utils import read_data_to_dataframe, apply_one_hot, classifier_learn, classifier_kfold_validation
from src.utils import run_exhaustive_search
from src.utils import svm_estimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def plot_corr(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns, annot=True, ax=ax,
                linewidths=.5)
    # ax.set(xticks=np.arange(data.shape[1]) + .5,
    #        yticks=np.arange(data.shape[0]) + .5)
    ax.set_ylim(len(data.columns) + 0.5, -0.5)
    plt.show()

def Histograms(df_red_wine, df_white_wine):
    for i in range(len(df_red_wine.columns)):
        df_red_wine.iloc[:,i].hist(alpha=0.5)
        df_white_wine.iloc[:,i].hist(alpha=0.5)
        plt.title(df_red_wine.columns[i], fontsize=18)
        plt.show()

def radar_chart(labels, values, title):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    ax.set_title(title)
    ax.grid(True)
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.show()



if __name__ == "__main__":
# (1) Loading
    df_red_wine =read_data_to_dataframe("../data/winequality-red.csv")
    df_white_wine =read_data_to_dataframe("../data/winequality-white.csv")
# (2) Pre-processing
# Description
    # print(df_red_wine.describe())
    # print(df_white_wine.describe())
# Histogram
    # Histograms(df_red_wine, df_white_wine)
# Min-Max scaling and Radar Chart
    X_redwine = df_red_wine.iloc[:,:-1]
    y_redwine = df_red_wine.iloc[:,-1]

    X_whitewine = df_white_wine.iloc[:,:-1]
    y_whitewine = df_white_wine.iloc[:,-1]
    df_red_wine_scaled = (X_redwine - X_redwine.min()) / (X_redwine.max() - X_redwine.min())
    df_white_wine_scaled = (X_whitewine - X_whitewine.min()) / (X_whitewine.max() - X_whitewine.min())
    # labels = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides',
    #           'free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','quality']
    # radar_chart(labels, df_red_wine_scaled.mean(), 'Red Wine Mean Radar Chart (0-1)')
    # radar_chart(labels, df_white_wine_scaled.mean(), 'White Wine Mean Radar Chart (0-1)')
# Heat map
#     plot_corr(pd.concat([df_red_wine_scaled, y_redwine], axis=1))
    plot_corr(pd.concat([df_white_wine_scaled, y_whitewine], axis=1))

# (3) Analyzing/Experiment
# Random Forest
#     clf = RandomForestClassifier()
#     classifier_kfold_validation(df_red_wine, clf)
#     classifier_kfold_validation(df_white_wine, clf)

# Support Vector Machine
#     clf = svm_estimation(df_red_wine)
#     clf = svm.SVC(kernel='rbf', gamma=0.01, C=1000)
#     classifier_kfold_validation(df_red_wine, clf)
    # classifier_kfold_validation(df_white_wine, clf)

# Gradient Boosting
#     run_exhaustive_search(clf, df, 1, parameter_space)
#     train_acc, test_acc =classifier_learn(df_red_wine)
#
#     data = [1, 5, 10, 25, 35, 50, 75, 100, 150, 175, 200]
#     from matplotlib import pyplot as plt
#
#     plt.plot(data, train_acc)
#     plt.plot(data, test_acc)
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('number of rounds')
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.show()

    # gavazn=""

    # classifier_kfold_validation(df, clf)
    # train_acc, test_acc = classifier_learn(df_white_wine_scaled, y_whitewine)
    #
    # import matplotlib.pyplot as plt
    # C = [0.01, 0.1, 1, 10, 100, 1000]
    # # Kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    # plt.plot(C, train_acc)
    # plt.plot(C, test_acc)
    # plt.title('red wine model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('SVM Kernel Parameter with C=1.0')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
