import numpy as np
import tensorflow as tf

from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from matt.utils.helper import get_training_data


class ANN:
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set

    def prep_training_data(self):
        # Get training data in np.array format
        X_train, y_train, class_label_pair, X_train_ids = get_training_data(self.training_set, self.training_anno_file)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            stratify=y_train)

        # Preprocess the data
        #X_train = tf.keras.utils.normalize(X_train, axis=1)
        #X_test = tf.keras.utils.normalize(X_train, axis=1)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prep_training_data()

        # Create MLP classifier
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Probability distribution

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

        return model

    def load_saved_model(self, loaded_model):
        X_train, X_test, y_train, y_test = self.prep_training_data()

        loaded_model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        score = loaded_model.evaluate(X_test, y_test, verbose=0)

        print('%s: %.2f%%' % (loaded_model.metrics_names[1], score[1]*100))

        return loaded_model
