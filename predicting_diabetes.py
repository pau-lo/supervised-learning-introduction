import pandas as pd
import numpy as np

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
    #                            
    # split into train/test
    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)

    # feature scaling 
    # rule of thumb:  any algorithm that computes distance or assumes normality
    # scale features.. exclude tree classifiers.
    scaling_X = StandardScaler() # set data from -1 to 1
    # fitting only the train set
    X_train = scaling_X.fit_transform(X_train)
    # yet, make sure the x going in it's transform
    X_test = scaling_X.transform(X_test)

    # load the classifer
    clf = KNeighborsClassifier()

    # fit and predict
    # clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    # score = accuracy_score(y_test, predictions)
    # print('Accuracy: {0:f}'.format(score))

