#!/usr/bin/env python
# Created by "Thieu" at 00:06, 22/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVM
from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix, accuracy_score

import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    # class for defining the evaluation metrics
    def __init__(self, train_X, test_X, train_Y, test_Y, solution=None, classifier=None, draw_confusion_matrix=False, average="weighted"):
        self.solution = solution
        if self.solution is None:
            self.solution = np.ones(train_X.shape[1])

        # store the train and test features and labels
        cols = np.flatnonzero(self.solution)
        self.X_train = train_X[:, cols]
        self.X_test = test_X[:, cols]
        self.y_train = train_Y
        self.y_test = test_Y

        # set the classifier type
        self.classifier = classifier
        if self.classifier.lower() == 'knn':
            self.clf = KNN()
        elif self.classifier.lower() == 'rf':
            self.clf = RF()
        elif self.classifier.lower() == 'svm':
            self.clf = SVM()
        else:
            self.clf = None
            print('\n[Error!] We don\'t currently support {} classifier...\n'.format(classifier))
            exit(0)

        # get the unique labels
        self.n_labels = len(np.unique(train_Y))
        self.average = "binary" if self.n_labels == 2 else average
        self.draw_confusion_matrix = draw_confusion_matrix

    def get_metrics(self):
        ## Train on training set
        self.clf.fit(self.X_train, self.y_train)

        ## Test and get accuracy on testing set
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=self.average, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=self.average, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=self.average)

        ## Save confusion matrix
        if self.draw_confusion_matrix:
            plot_confusion_matrix(self.clf, self.X_test, self.y_test)
            plt.savefig('confusion_matrix.png')
            plt.title('Confusion Matrix')
            plt.show()

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
