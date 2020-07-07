import argparse
import time

from _daal4py import logistic_regression_training, logistic_regression_prediction, \
    decision_forest_classification_prediction, decision_forest_classification_training, kernel_function_linear, \
    svm_training, svm_prediction
import numpy as np
from tensorflow import keras

from matt.models.ModelLoader import ModelLoader
from utils.helper import read_csv_dataset


def main():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline",
                                     add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset")
    parser.add_argument('-a', '--anno', action="store", help="Annoation level: {top, mid, fine}")
    parser.add_argument('-m', '--model', action="store")

    args = parser.parse_args()

    if args.dataset is None or args.dataset not in ["NetML", "CICIDS"] or args.anno is None:
        print("No valid dataset set or annotations found!")
        return
    elif args.anno is None or args.anno not in ["top", "mid", "fine"]:
        print("Please select one of these for annotations: {top, mid, fine}. e.g. --anno top")
        return
    elif args.anno == "mid" and (args.dataset.find("NetML") > 0 or args.dataset.find("CICIDS2017") > 0):
        print(
            "NetML and CICIDS2017 datasets cannot be trained with mid-level annotations. Please use either top or fine.")
        return
    elif args.model not in ["lr", "df", "svm", "ann"]:
        print("Please select one of these for model: {lr, df, svm, ann}. e.g. --model lr")
        return
    else:
        if args.dataset is "NetML":
            args.dataset = "./data/NetML_enc_filtered_top50.csv"
        else:
            args.dataset = "./data/CICIDS2017_enc_filtered_top50.csv"

        data, labels = read_csv_dataset(args.dataset)

        # Train respective model
        if args.model == 'lr':
            lr_daal = LR(data, labels)
            lr_daal.train()
        elif args.model == 'df':
            df_daal = DF(data, labels)
            df_daal.train()
        elif args.model == 'svm':
            svm_daal = SVM(data, labels)
            svm_daal.train()
        elif args.model == 'ann':
            ann_daal = ANN(data, labels)
            ann_daal.train()

        # python daalModels.py --dataset NetML --anno top --model lr

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


class DF:
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
        self.y_train = self.labels[0:upperBound]
        self.X_test = self.data[upperBound:]
        self.y_test = self.labels[upperBound:]

    def train(self,
              save_model=True):
        # Begin train timing
        startTime = time.time()

        # Decision Forest
        trainAlg = decision_forest_classification_training(2, nTrees=100, maxTreeDepth=0)

        # Train model
        trainResult = trainAlg.compute(self.X_train, self.y_train)

        # Create prediction classes
        predictAlgTrain = decision_forest_classification_prediction(2)
        predictAlgTest = decision_forest_classification_prediction(2)

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

        print("Training and test (Decision Forest) elapsed in %.3f seconds" % (endTime - startTime))
        print("Train accuracy: ", trainAccu)
        print("Test accuracy: ", testAccu)

        if save_model:
            ml = ModelLoader('daal_DF', trainResult.model)
            ml.save_daal_model()

        ml = ModelLoader('daal_DF', None)
        loaded_model = ml.load_daal_model()
        self.load_saved_model(loaded_model)

    def load_saved_model(self, loaded_model):
        # Begin test timing
        startTime = time.time()

        # Create prediction class
        predictAlg = decision_forest_classification_prediction(2)

        # make predictions
        predictResultTest = predictAlg.compute(self.X_test, loaded_model)

        # End test timing
        endTime = time.time()

        # Assess accuracy
        correctTest = np.sum(self.y_test == predictResultTest.prediction.flatten())
        testAccu = float(correctTest) / len(self.y_test) * 100

        print("Test (Decision Forest) elapsed in %.3f seconds" % (endTime - startTime))
        print("Test accuracy: ", testAccu)


class SVM:
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
        # make 0 values -1
        self.labels = [-1 if i == 0 else 1 for i in self.labels]

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
        # Begin train timing
        startTime = time.time()

        # Support Vector Machine
        kern = kernel_function_linear(method='defaultDense')
        trainAlg = svm_training(nClasses=2, C=1e+6, maxIterations=1e+7, cacheSize=2000, kernel=kern,
                                accuracyThreshold=1e-3, doShrinking=True)
        # Train model
        trainResult = trainAlg.compute(self.X_train, self.y_train)

        # Create prediction classes
        predictAlgTrain = svm_prediction(nClasses=2, kernel=kern)
        predictAlgTest = svm_prediction(nClasses=2, kernel=kern)

        # Make train and test predictions
        predictResultTrain = predictAlgTrain.compute(self.X_train, trainResult.model)
        predictResultTest = predictAlgTest.compute(self.X_test, trainResult.model)

        # End train timing
        endTime = time.time()

        # Compare train predictions
        predictionsTrain = predictResultTrain.prediction.flatten()
        trainLabel = self.y_train.flatten()
        correctTrain = np.sum(np.logical_or(np.logical_and(trainLabel > 0, predictionsTrain > 0),
                                            np.logical_and(trainLabel < 0, predictionsTrain < 0)))
        trainAccu = float(correctTrain) / len(trainLabel) * 100

        # Compare test predictions
        predictionsTest = predictResultTest.prediction.flatten()
        testLabel = self.y_test.flatten()
        correctTest = np.sum(np.logical_or(np.logical_and(testLabel > 0, predictionsTest > 0),
                                           np.logical_and(testLabel < 0, predictionsTest < 0)))
        testAccu = float(correctTest) / len(testLabel) * 100

        print("Training and test (Support Vector Machine) elapsed in %.3f seconds" % (endTime - startTime))
        print("Train accuracy: ", trainAccu)
        print("Test accuracy: ", testAccu)

        if save_model:
            ml = ModelLoader('daal_SVM', trainResult.model)
            ml.save_daal_model()

        ml = ModelLoader('daal_SVM', None)
        loaded_model = ml.load_daal_model()
        self.load_saved_model(loaded_model)

    def load_saved_model(self, loaded_model):
        # Begin test timing
        startTime = time.time()

        # create prediction class
        kern = kernel_function_linear(method='defaultDense')
        predictAlg = svm_prediction(nClasses=2, kernel=kern)

        # make predictions
        predictResultTest = predictAlg.compute(self.X_test, loaded_model)

        # End test timing
        endTime = time.time()

        # assess accuracy
        predictions = predictResultTest.prediction.flatten()
        correctTest = np.sum(
            np.logical_or(np.logical_and(self.y_test > 0, predictions > 0), np.logical_and(self.y_test < 0, predictions < 0)))
        testAccu = float(correctTest) / len(self.y_test) * 100

        print("Test (Support Vector Machine) elapsed in %.3f seconds" % (endTime - startTime))
        print("Test accuracy: ", testAccu)


main()
