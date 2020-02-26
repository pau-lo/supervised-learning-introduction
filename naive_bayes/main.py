# Use case text-classification
# performing text classification on news headlines and classify
# news into different topics for news website

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class main():
    # load dataset
    df = fetch_20newsgroups()
    data = df.target_names
    print(data)

    # defining all the categories
    categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                  'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                  'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                  'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
                  'talk.politics.misc', 'talk.religion.misc']

    # training data on these categories
    train = fetch_20newsgroups(subset='train', categories=categories)

    # Testing the data for these categories
    test = fetch_20newsgroups(subset='test', categories=categories)

    # printing the trainning data
    # print(train.data[5])

    # creating model
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Trainning the model with the train data
    model.fit(train.data, train.target)

    # creating labels for the test data
    labels = model.predict(test.data)

    # creating confusion matrix and heat map
    cm = confusion_matrix(test.target, labels)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=train.target_names, yticklabels=train.target_names)

    # plotting heatmap of confusion matrix
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    # predicting category on new data on trained model
    def predict_category(s, train=train, model=model):
        pred = model.predict(s)
        return train.target_names[pred[0]]

