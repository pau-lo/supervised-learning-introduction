# loan repayment prediction

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


class main():
    # load dataset
    df = pd.read_csv('./data/Loan-data.csv')
    # df.drop([])
    # in case want to see dastaset
    print("Dataset shape: ", df.shape)
    print(df.dtypes)
    # print(df.head())
    # print(df.columns)

    # split into train/test
    X = df.iloc[:, 1:24]
    # print(X)
    y = df.iloc[:, 24]
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,

                                                        random_state=0)

    # load the classifier
    clf = DecisionTreeClassifier(
        criterion="entropy", random_state=0, max_depth=3, min_samples_leaf=5)

    # fit and predict the train data in the model
    clf.fit(X_train, y_train)
    # predictions
    # y_pred = clf.predict(X_test)
    # print(y_pred)
    # score = accuracy_score(y_test, y_pred)
    # print('Accuracy score: {0:f}'.format(score))
    # # Evaluate model
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix: \n", cm)
    # # f1 score measures the test's accuracy considering both
    # # precision and recall
    # f_score = f1_score(y_test, y_pred)
    # print("f1 score: {0:f}".format(f_score))
