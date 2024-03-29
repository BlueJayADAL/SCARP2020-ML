import time
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from utils.helper import collect_statistics
from utils.ModelLoader import ModelLoader


class kNN:
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

    def train_model(self,
                    save_model=True):
        nClasses = 2

        # Begin train timing
        startTime = time.time()

        # Create kNN Classifier
        knn = KNeighborsClassifier()

        # Train model
        knn.fit(self.X_train, self.y_train)

        y_train_pred = knn.predict(self.X_train)
        y_test_pred = knn.predict(self.X_test)

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

        if save_model:
            ml = ModelLoader('model_knn', knn)
            ml.save_sk_daal_model()

        return test_accu, test_tpr, test_far, test_report

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
