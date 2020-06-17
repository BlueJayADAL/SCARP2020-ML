import numpy as np
import tensorflow as tf

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

from matt.utils.helper import get_training_data


class ANN:
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set

    def train_model(self):
        # Get training data in np.array format
        X_train, y_train, class_label_pair, X_train_ids = get_training_data(self.training_set, self.training_anno_file)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            stratify=y_train)

        # Preprocess the data
        X_train = tf.keras.utils.normalize(X_train, axis=1)
        X_test = tf.keras.utils.normalize(X_train, axis=1)

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

        y_pred = model.predict(X_test)

        #cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        #detectionRate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        #falseAlarmRate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        #print("TPR: \t\t\t{:.5f}".format(detectionRate))
        #print("FAR: \t\t\t{:.5f}".format(falseAlarmRate))

        #print("Mean ACC: \t\t{:.5f}".format(clf.score(X_test_scaled, y_test)))
