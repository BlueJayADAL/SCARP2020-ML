import time
import numpy as np

from sklearn.linear_model import LogisticRegression

from utils.helper import collect_statistics
from models.ModelLoader import ModelLoader


class LR:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Prep data for use in training / testing
        self.prep_data()

    def prep_data(self):
        """
        Sets X_train, X_test, y_train, and y_test variables for training / testing.
        Run this method to reset values
        """
        # Reshape labels
        self.data = np.array(self.data)
        self.labels = np.array(self.labels).reshape((len(self.labels), 1))

        # Setup train / test data
        dataLen = len(self.data)
        mark = 0.8
        upperBound = int(dataLen * mark)

        self.X_train = self.data[0:upperBound]
        self.y_train = self.labels[0:upperBound].flatten()
        self.X_test = self.data[upperBound:]
        self.y_test = self.labels[upperBound:].flatten()

    def train(self,
              save_model=True):

        nClasses = 2

        # begin train timing
        startTime = time.time()

        # Create Logistic Regression Classifier
        logreg = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=1000000, verbose=1)

        # Train model
        logreg.fit(self.X_train, self.y_train)

        y_train_pred = logreg.predict(self.X_train)
        y_test_pred = logreg.predict(self.X_test)

        # End train timing
        endTime = time.time()

        # Collect statistics
        train_tpr, train_far, train_accu, train_report = collect_statistics(self.y_train, y_train_pred)
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test, y_test_pred)

        print("Training and testing (Logistic Regression) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Training Results ---")
        print("Train accuracy: ", train_accu)
        print("TPR: ", train_tpr)
        print("FAR: ", train_far)
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")
        return test_accu, test_tpr, test_far, test_report

        if save_model:
            ml = ModelLoader('model_LR', logreg)
            ml.save_sk_daal_model()

    def load_saved_model(self, loaded_model):
        # Begin test timing
        startTime = time.time()

        # Make predictions
        y_pred = loaded_model.predict(self.X_test)

        # End test timing
        endTime = time.time()

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test, y_pred)

        print("Test (Logistic Regression) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")
        return test_accu, test_tpr, test_far, test_report
