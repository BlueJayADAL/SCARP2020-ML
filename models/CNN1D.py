import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Model

from models.ModelLoader import ModelLoader
from utils.helper import encode_label
from utils.helper2 import read_dataset, one_hot


class CNN1D:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train_1D = None
        self.X_test_1D = None
        self.y_train = None
        self.y_test = None

        self.top_class_names = None
        self.n_classes_top = None

        self.prep_data()

    def prep_data(self):
        """
        Sets X_train_1D, X_test_1D, y_train, y_test, and  variables for training / testing.
        Run this method to reset values
        """
        # Load label array and class label pairs
        label_array, class_label_pairs = encode_label(self.labels)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(self.data, label_array,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=self.labels)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_test)

        # Get name of each class to display in confusion matrix
        self.top_class_names = list(sorted(class_label_pairs.keys()))

        self.n_classes_top = len(self.top_class_names)

        # Reshape and assign training and testing variables
        self.X_train_1D = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
        self.X_test_1D = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1)

        self.y_train = y_train#np.array(y_train).reshape((len(y_train), 1))
        self.y_test = y_test#np.array(y_test).reshape((len(y_test), 1))

    def train_model(self,
                    save_model=True):


        # Default Training Hyper-parameters
        # n_classes_fine = len(fine_class_names)
        learning_rate = 1e-3
        decay_rate = 1e-5
        dropout_rate = 0.5
        n_batch = 64
        n_epochs = 10  # Loop 10 times on the dataset
        filters = 128
        kernel_size = 4
        strides = 1
        CNN_layers = 2
        clf_reg = 1e-5

        OUTPUT = {anno: self.n_classes_top}

        # Model Definition
        OUTPUTS = []
        # raw_inputs = Input(shape=(X_train.shape[1],))
        raw_inputs = Input(shape=(self.X_train_1D.shape[1], 1,))
        xcnn = Conv1D(filters, kernel_size,
                      input_shape=(self.X_train_1D.shape[1], 1),
                      padding='same',
                      activation='relu',
                      strides=strides)(raw_inputs)

        xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv1D(filters,
                          kernel_size,
                          padding='same',
                          activation='relu',
                          strides=strides)(xcnn)

            xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

            if dropout_rate != 0:
                xcnn = Dropout(dropout_rate)(xcnn)

                # we flatten for dense layer
        xcnn = Flatten()(xcnn)

        xcnn = Dense(128, activation='relu',
                     name='FC1_layer')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        xcnn = Dense(128, activation='relu',
                     name='FC2_layer')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        # Use top level predictions because csv dataset is based on top level annotations
        top_level_predictions = Dense(2, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                      name='top_level_output')(xcnn)
        OUTPUTS.append(top_level_predictions)

        model = Model(inputs=raw_inputs, outputs=OUTPUTS)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])

        print(model.summary())  # summarize layers

        # Train the model
        history = model.fit(self.X_train_1D, one_hot(self.y_train, self.n_classes_top),
                            batch_size=n_batch,
                            epochs=n_epochs,
                            validation_data=(self.X_test_1D, one_hot(self.y_test, self.n_classes_top)))

        score = model.evaluate(self.X_test_1D, one_hot(self.y_test, self.n_classes_top), verbose=0)
        print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))

        if save_model:
            ml = ModelLoader('model_cnn1d', model)
            ml.save_keras_model()

    def load_saved_model(self, loaded_model):
        # Base settings for learning
        learning_rate = 1e-3
        decay_rate = 1e-5

        loaded_model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])

        score = loaded_model.evaluate(self.X_test_1D, one_hot(self.y_test, self.n_classes_top), verbose=0)

        print('%s: %.2f%%' % (loaded_model.metrics_names[1], score[1]*100))

        return loaded_model
