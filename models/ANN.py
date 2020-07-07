import numpy as np
import tensorflow as tf

from matt.models.ModelLoader import ModelLoader


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

        self.data = self.data.values


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
        # Create ANN classifier
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Probability distribution

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_split=0.1)

        if save_model:
            # Save model to disk
            ml = ModelLoader('ann_model', model)
            ml.save_keras_model()

            # Load model from disk and test
            loaded_model = ml.load_keras_model()
            self.load_saved_model(loaded_model)

    def load_saved_model(self, loaded_model):
        """
        Compiles loaded model and tests for accuracy
        """
        loaded_model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        score = loaded_model.evaluate(self.X_test, self.y_test, verbose=0)

        print('%s: %.2f%%' % (loaded_model.metrics_names[1], score[1]*100))

        return loaded_model
