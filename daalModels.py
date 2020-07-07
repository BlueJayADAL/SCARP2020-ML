import argparse
import os
import time

import joblib
from _daal4py import logistic_regression_training, logistic_regression_prediction, \
    decision_forest_classification_prediction, decision_forest_classification_training
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from matt.models.ModelLoader import ModelLoader
from matt.utils.helper import get_training_data


def main():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline",
                                     add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset path")
    parser.add_argument('-a', '--anno', action="store", help="Annoation level: {top, mid, fine}")
    parser.add_argument('-s', '--submit', action="store",
                        help="{test-std, test-challenge, both} Select which set to submit")
    parser.add_argument('-m', '--model', action="store")

    args = parser.parse_args()

    if args.dataset is None or not os.path.isdir(args.dataset) or args.anno is None:
        print("No valid dataset set or annotations found!")
        return
    elif args.submit is not None and args.submit not in ["test-std", "test-challenge", "both"]:
        print("Please select which set to submit: {test-std, test-challenge, both}")
        return
    elif args.anno not in ["top", "mid", "fine"]:
        print("Please select one of these for annotations: {top, mid, fine}. e.g. --anno top")
        return
    elif args.anno == "mid" and (args.dataset.find("NetML") > 0 or args.dataset.find("CICIDS2017") > 0):
        print(
            "NetML and CICIDS2017 datasets cannot be trained with mid-level annotations. Please use either top or fine.")
        return
    elif args.model not in ["lr", "df"]:
        print("Please select one of these for model: {lr, df}. e.g. --model lr")
        return
    else:
        training_set = args.dataset + "/2_training_set"
        training_anno_file = args.dataset + "/2_training_annotations/2_training_anno_" + args.anno + ".json.gz"
        test_set = args.dataset + "/1_test-std_set"
        challenge_set = args.dataset + "/0_test-challenge_set"

        # Train respective model
        if args.model == 'lr':
            lr_daal = LR(training_set, training_anno_file, test_set)
            lr_daal.train()
        elif args.model == 'df':
            df_daal = DF(training_set, training_anno_file, test_set)
            df_daal.train()

        # python daalModels.py --dataset ./matt/data/NetML --anno top --submit test-std --model lr

class LR:
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

        # Prep data for use in training / testing
        self.prep_data()

    def prep_data(self):
        """
        Sets X_train_scaled, X_test_scaled, y_train, and y_test variables for training / testing.
        Run this method to reset values
        """
        # Get training data in np.array format
        X_train, y_train, class_label_pair, X_train_ids = get_training_data(self.training_set, self.training_anno_file)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            stratify=y_train)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reshape y_train and y_test
        self.y_train = y_train.reshape(y_train.shape[0], 1)
        self.y_test = y_test.reshape(y_test.shape[0], 1)

        # Set variables for use in training / testing later
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

    def train(self,
              save_model=True):


        nClasses = 2

        # begin train timing
        # print("Beginning train timing...")
        # saga, sgd, adagrad, lbfgs (default sgd)
        trainAlg = logistic_regression_training(nClasses=nClasses, interceptFlag=True)
        # train model
        # print("Training model...")
        trainResult = trainAlg.compute(self.X_train_scaled,
                                       self.y_train)
        # create prediction classes 0.
        predictAlgTrain = logistic_regression_prediction(nClasses=nClasses)
        predictAlgTest = logistic_regression_prediction(nClasses=nClasses)
        # make train and test predictions
        predictResultTrain = predictAlgTrain.compute(self.X_train_scaled, trainResult.model)
        predictResultTest = predictAlgTest.compute(self.X_test_scaled, trainResult.model)
        # sum prediction results
        correctTrain = np.sum(self.y_train.flatten() == predictResultTrain.prediction.flatten())
        correctTest = np.sum(self.y_test.flatten() == predictResultTest.prediction.flatten())
        # compare train predictions
        trainAccu = float(correctTrain) / len(self.y_train) * 100
        # compare test predictions
        testAccu = float(correctTest) / len(self.y_test) * 100

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
        predictResultTest = predictAlg.compute(self.X_test_scaled, loaded_model)
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
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

        # Prep data for use in training / testing
        self.prep_data()

    def prep_data(self):
        """
        Sets X_train_scaled, X_test_scaled, y_train, and y_test variables for training / testing.
        Run this method to reset values
        """
        # Get training data in np.array format
        X_train, y_train, class_label_pair, X_train_ids = get_training_data(self.training_set, self.training_anno_file)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            stratify=y_train)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reshape y_train and y_test
        self.y_train = y_train.reshape(y_train.shape[0], 1)
        self.y_test = y_test.reshape(y_test.shape[0], 1)

        # Set variables for use in training / testing later
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

    def train(self,
              save_model=True):
        # Begin train timing
        startTime = time.time()

        # Decision Forest
        trainAlg = decision_forest_classification_training(2, nTrees=100, maxTreeDepth=0)
        # Train model
        trainResult = trainAlg.compute(self.X_train_scaled, self.y_train)
        # Create prediction classes
        predictAlgTrain = decision_forest_classification_prediction(2)
        predictAlgTest = decision_forest_classification_prediction(2)
        # Make train and test predictions
        predictResultTrain = predictAlgTrain.compute(self.X_train_scaled, trainResult.model)
        predictResultTest = predictAlgTest.compute(self.X_test_scaled, trainResult.model)

        # End train timing
        endTime = time.time()

        # Sum prediction results
        correctTrain = np.sum(self.X_train_scaled.flatten() == predictResultTrain.prediction.flatten())
        correctTest = np.sum(self.X_test_scaled.flatten() == predictResultTest.prediction.flatten())

        # compare train predictions
        trainAccu = float(correctTrain) / len(self.y_train) * 100
        # compare test predictions
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
        startTime = time.time()
        # Create prediction class
        predictAlg = decision_forest_classification_prediction(2)
        # Make predictions
        predictResultTest = predictAlg.compute(self.X_test_scaled, loaded_model)
        endTime = time.time()
        print("Test (Decision Forest) elapsed in %.3f seconds" % (endTime - startTime))
        correctTest = np.sum(self.y_test == predictResultTest.prediction.flatten())
        print("Loaded Test accuracy: ", float(correctTest) / len(self.y_test) * 100)


main()
