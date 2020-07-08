import time
import numpy as np

from _daal4py import logistic_regression_training, logistic_regression_prediction

from matt.models.ModelLoader import ModelLoader


class daal_LR:
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
        self.labels = np.array(self.labels).reshape((len(self.labels), 1))

        # Setup train / test data
        dataLen = len(self.data)
        mark = 0.8
        upperBound = int(dataLen * mark)

        self.X_train = self.data[0:upperBound]
        self.y_train = self.labels[0:upperBound]
        self.X_test = self.data[upperBound:]
        self.y_test = self.labels[upperBound:]

    def train(self,
              save_model=True):

        nClasses = 2

        # begin train timing
        startTime = time.time()

        # Create Logistic Regression Classifier
        trainAlg = logistic_regression_training(nClasses=nClasses, interceptFlag=True)

        # Train model
        trainResult = trainAlg.compute(self.X_train,
                                       self.y_train)
        # Create prediction classes 0.
        predictAlgTrain = logistic_regression_prediction(nClasses=nClasses)
        predictAlgTest = logistic_regression_prediction(nClasses=nClasses)
        # Make train and test predictions
        predictResultTrain = predictAlgTrain.compute(self.X_train, trainResult.model)
        predictResultTest = predictAlgTest.compute(self.X_test, trainResult.model)

        # End train timing
        endTime = time.time()

        # Sum prediction results
        correctTrain = np.sum(self.y_train.flatten() == predictResultTrain.prediction.flatten())
        correctTest = np.sum(self.y_test.flatten() == predictResultTest.prediction.flatten())
        # Compare train predictions
        trainAccu = float(correctTrain) / len(self.y_train) * 100
        # Compare test predictions
        testAccu = float(correctTest) / len(self.y_test) * 100

        print("Training and test (Logistic Regression) elapsed in %.3f seconds" % (endTime - startTime))
        print("Train accuracy: ", trainAccu)
        print("Test accuracy: ", testAccu)

        if save_model:
            ml = ModelLoader('daal_LR', trainResult.model)
            ml.save_daal_model()

        ml = ModelLoader('daal_LR', None)
        loaded_model = ml.load_daal_model()
        self.load_saved_model(loaded_model)

    def load_saved_model(self, loaded_model):
        startTime = time.time()
        # create prediction class
        predictAlg = logistic_regression_prediction(nClasses=2)
        # make predictions
        predictResultTest = predictAlg.compute(self.X_test, loaded_model)
        endTime = time.time()
        print("Test (Logistic Regression) elapsed in %.3f seconds" % (endTime - startTime))
        # assess accuracy
        count = 0
        for i in range(0, len(self.y_test)):
            if self.y_test[i] == predictResultTest.prediction[i]:
                count += 1
        print("Test (Logistic Regression) has a test accuracy of ", float(count) / len(self.y_test) * 100)
        # return float(count) / len(self.y_test) * 100
