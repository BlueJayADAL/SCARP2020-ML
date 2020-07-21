import time

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Dense, Flatten

from utils.helper import collect_statistics
from models.ModelLoader import ModelLoader


class ANN:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.prep_data()

    def prep_data(self):
        # Split validation set from training data
        # Reshape labels
        self.labels = np.array(self.labels).reshape((len(self.labels), 1))

        # Convert from dataframe to np.array
        self.data = self.data.values

        # Setup train / test data
        dataLen = len(self.data)
        mark = 0.8
        upperBound = int(dataLen * mark)

        self.X_train = self.data[0:upperBound]
        self.y_train = self.labels[0:upperBound]
        self.X_test = self.data[upperBound:]
        self.y_test = self.labels[upperBound:]

    def train_model(self,
                    save_model=True):
        # Begin train timing
        startTime = time.time()

        # Create ANN classifier
        model = tf.keras.models.Sequential()

        # Add input layer required for interpretation from OpenVINO
        model.add(Dense(64, input_shape=(self.X_train.shape[1],), activation='relu'))

        model.add(Flatten())
        model.add(Dense(64, activation=tf.nn.relu))
        model.add(Dense(32, activation=tf.nn.relu))
        model.add(Dense(1, activation='sigmoid'))  # Probability distribution

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_split=0.1)

        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # End train timing
        endTime = time.time()

        # Collect statistics
        train_tpr, train_far, train_accu, train_report = collect_statistics(self.y_train, y_train_pred)
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test, y_test_pred)

        print("Training and testing (Artificial Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
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
            # Save model to disk
            ml = ModelLoader('model_ann', model)
            ml.save_keras_model()

    def load_saved_model(self, loaded_model):
        """
        Compiles loaded model and tests for accuracy
        """
        # Begin test timing
        startTime = time.time()

        loaded_model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        score = loaded_model.evaluate(self.X_test, self.y_test, verbose=0)

        testAccu = loaded_model.metrics_names[1], score[1]*100

        # End test timing
        endTime = time.time()

        y_pred = loaded_model.predict_classes(self.X_test)

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test.flatten(), y_pred.flatten())

        print("Test (ANN) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", testAccu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")
