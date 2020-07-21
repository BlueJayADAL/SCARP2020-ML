import time

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Embedding, LSTM, Dropout, Dense

from utils.helper import collect_statistics, convertToDefault
from models.ModelLoader import ModelLoader


class RNN:
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

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])

    def train_model(self,
              save_model=True):
        print(self.X_train.shape)
        print(self.y_train.shape)

        # Create ANN classifier
        model = tf.keras.models.Sequential()

        # Add input layer required for interpretation from OpenVINO
        model.add(LSTM(units=50, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=False))

        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, validation_split=0.1)

        y_train_pred = convertToDefault(model.predict(self.X_train))
        y_test_pred = convertToDefault(model.predict(self.X_test))
        print(y_train_pred.shape)
        print(self.y_train.shape)

        train_tpr, train_far, train_accu, train_report = collect_statistics(self.y_train.flatten(), y_train_pred)
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test.flatten(), y_test_pred)

        print("Training and testing (Recurrent Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
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
            ml = ModelLoader('model_rnn', model)
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

        # End test timing
        endTime = time.time()

        y_pred = loaded_model.predict_classes(self.X_test)

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test.flatten(), y_pred.flatten())

        print("Test (Recurrent Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")
