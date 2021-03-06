import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class main():
    # load dataset
    df = pd.read_csv('./data/diabetes.csv')
    # in case want to see dastaset
    # print("Dataset shape: ", df.shape)
    # print(df.head())

    # prepraring data
    # replacing zeroes in datasets with Nan and the filled them with the mean
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.NaN)
        mean = int(df[column].mean(skipna=True))
        df[column] = df[column].replace(np.NaN, mean)
    # print(df.head(10))

    # checking for NaN values
    # print(df.isnull().sum())
    # split into train/test
    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)

    # feature scaling 
    # rule of thumb:  any algorithm that computes distance or assumes normality
    # scale features.. exclude tree classifiers.
    # set data from -1 to 1
    scaling_X = StandardScaler()
    # fitting only the train set
    X_train = scaling_X.fit_transform(X_train)
    # yet, make sure the x going in it's transform
    X_test = scaling_X.transform(X_test)

    # load the classifier
    # N_neighbors = K
    # print(math.sqrt(len(y_test))) # this gives ~ 12.4 we'll use 11
    # p == power parameter to define the metric 2 classes
    clf = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

    # fit and predict the train data in the model
    clf.fit(X_train, y_train)
    # predictions
    y_pred = clf.predict(X_test)
    # print(y_pred)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy score: {0:f}'.format(score))
    # Evaluate model
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: \n", cm)
    # f1 score measures the test's accuracy considering both 
    # precision and recall
    f_score = f1_score(y_test, y_pred)
    print("f1 score: {0:f}".format(f_score))
