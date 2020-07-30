import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.autograph.pyct import anno

from utils.helper2 import read_dataset, one_hot


class CNN2D:
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set

    def prep_training_data(self):
        # TLS, DNS, HTTP features included?
        TLS, DNS, HTTP = {}, {}, {}
        TLS['tlsOnly'] = False  # returns
        TLS['use'] = True
        TLS['n_common_client'] = 10
        TLS['n_common_server'] = 5
        #
        DNS['use'] = False
        ##
        ##
        #
        HTTP['use'] = False
        ##
        ##

        feature_names, ids, Xtrain, ytrain, class_label_pairs = read_dataset(self.training_set,
                                                                             annotationFileName=self.training_anno_file,
                                                                             TLS=TLS, class_label_pairs=None)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=ytrain)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, class_label_pairs

    def create_model(self):
        X_train_scaled, X_test_scaled, y_train, y_test, class_label_pairs = self.prep_training_data()

        # Get name of each class to display in confusion matrix
        top_class_names = list(sorted(class_label_pairs.keys()))
        # fine_class_names = list(sorted(class_label_pairs_list[1].keys()))

        # Default Training Hyperparameters
        n_classes_top = len(top_class_names)
        # n_classes_fine = len(fine_class_names)
        learning_rate = 1e-3
        decay_rate = 1e-5
        dropout_rate = 0.5
        n_batch = 64
        n_epochs = 1  # Loop 500 times on the dataset
        filters = 128
        kernel_size = 4
        strides = 1
        CNN_layers = 2
        clf_reg = 1e-5

        # Model Definition
        X_train_2D = X_train_scaled.reshape(-1, 9, 19, 1)  # reshape 171 to 9x19
        X_val_2D = X_test_scaled.reshape(-1, 9, 19, 1)  # reshape 171 to 9x19
        model = self.train_model(X_train_2D, {anno: n_classes_top},
                              filters=filters, kernel_size=kernel_size,
                              strides=strides, dropout_rate=dropout_rate,
                              CNN_layers=CNN_layers, clf_reg=clf_reg)

        print(model.summary())  # summarize layers
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])
        # Train the model
        history = model.fit(X_train_2D, one_hot(y_train, n_classes_top), batch_size=n_batch, epochs=n_epochs,
                            validation_data=(X_val_2D, one_hot(y_test, n_classes_top)))

    def train_model(self, X_train, OUTPUT,
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    dropout_rate=0.,
                    CNN_layers=2,
                    clf_reg=1e-4,
                    learning_rate=1e-3,
                    decay_rate=1e-5):
        # Model Definition
        OUTPUTS = []
        # raw_inputs = Input(shape=(X_train.shape[1],))
        raw_inputs = Input(shape=(X_train.shape[1], X_train.shape[2], 1))
        # xcnn = Embedding(1000, 50, input_length=X_train.shape[1])(raw_inputs)
        xcnn = Conv2D(filters, (kernel_size, kernel_size),
                      input_shape=(X_train.shape[1], X_train.shape[2], 1),
                      padding='same',
                      activation='relu',
                      strides=strides)(raw_inputs)

        xcnn = MaxPooling2D(pool_size=(2, 2), padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv2D(filters,
                          (kernel_size, kernel_size),
                          padding='same',
                          activation='relu',
                          strides=strides)(xcnn)

            xcnn = MaxPooling2D(pool_size=(2, 2), padding='same')(xcnn)

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

        if 'top' in OUTPUT:
            top_level_predictions = Dense(OUTPUT['top'], activation='softmax',
                                          kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                          bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                          activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                          name='top_level_output')(xcnn)
            OUTPUTS.append(top_level_predictions)

        if 'mid' in OUTPUT:
            mid_level_predictions = Dense(OUTPUT['mid'], activation='softmax',
                                          kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                          bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                          activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                          name='mid_level_output')(xcnn)
            OUTPUTS.append(mid_level_predictions)

        if 'fine' in OUTPUT:
            fine_grained_predictions = Dense(OUTPUT['fine'], activation='softmax',
                                             kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                             bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                             activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                             name='fine_grained_output')(xcnn)
            OUTPUTS.append(fine_grained_predictions)

        model = Model(inputs=raw_inputs, outputs=OUTPUTS)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])

        return model


    def load_saved_model(self, loaded_model):
        X_train_scaled, X_test_scaled, y_train, y_test, class_label_pairs = self.prep_training_data()

        # Base settings for learning
        learning_rate = 1e-3
        decay_rate = 1e-5

        loaded_model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])

        score = loaded_model.evaluate(X_test_scaled, y_test, verbose=0)

        print('%s: %.2f%%' % (loaded_model.metrics_names[1], score[1]*100))

        return loaded_model
