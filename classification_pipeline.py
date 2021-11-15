#### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# standard classification pipeline class inherited by a specific model to use
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

class ClassificationPipeline:
    def __init__(self):
        self.cm = []
        self.cmap = 'Blues'
        self.score = 0
        self.f1 = 0
        self.le = preprocessing.LabelEncoder()
        self.y_map = dict()
        self.classes_str = []
        self.standardscaler = preprocessing.StandardScaler()

    def classify(self,X,y,category, plot=1):
        self.predict(X,y)
        if plot:
            self.plotConfusionMatrix(category)
        return self.score, self.f1, self.cm, self.classes_str

    def predict(self,X,Ystr):
        nf = 5
        y = self.encodeLabels(Ystr)
        self.cm = np.zeros(shape=(len(self.classes_str),len(self.classes_str)))
        X = np.array(X)
        y = np.array(y)
        kfold = KFold(n_splits = nf, shuffle=True, random_state=0)
        for train_index, test_index in kfold.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            x_train, x_test = self.preprocess(x_train, x_test)
            self.model.fit(x_train, y_train)
            predictions = self.model.predict(x_test)
            self.score += self.model.score(x_test, y_test)
            self.f1 += metrics.f1_score(y_test, predictions, average='weighted')
            self.cm += np.array(metrics.confusion_matrix(y_test, predictions, normalize='true'))
        self.score = self.score / nf
        self.f1 = self.f1 / nf
        self.cm = self.cm/nf

    def preprocess(self,x_train,x_test):
        scaler = self.standardscaler.fit(x_train)
        scaler.transform(x_train)
        scaler.transform(x_test)
        return x_train, x_test

    def encodeLabels(self,Ystr):
        self.le.fit(Ystr)
        y = self.le.transform(Ystr)
        self.y_map = dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))
        self.classes_str = self.le.classes_
        #print(self.y_map)
        return y


    def plotConfusionMatrix(self,category):
        plt.figure(figsize=(8,8))
        sns.heatmap(100*self.cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = self.cmap,
                    xticklabels=self.classes_str, yticklabels=self.classes_str);
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = '{0} Accuracy = {1:.1f}, F1 = {2:.2f}'.format(category,100*self.score, self.f1)
        plt.title(all_sample_title, size = 15);
