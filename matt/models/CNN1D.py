import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow_core.python.autograph.pyct import anno
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow_core.python.keras.models import Model

from utils.helper2 import read_dataset, one_hot


class CNN1D:
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set

    def create_model(self):
        dataset = "./data/NetML"  # "./data/non-vpn2016" or "./data/CICIDS2017" or "./data/non-vpn2016"
        anno = "top"  # or "mid" or "fine"
        # submit = "both" # or "test-std" or "test-challenge"

        # Assign variables
        training_set = dataset + "/2_training_set"
        training_anno_file = dataset + "/2_training_annotations/2_training_anno_" + anno + ".json.gz"
        test_set = dataset + "/1_test-std_set"
        challenge_set = dataset + "/0_test-challenge_set"

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

        # Get training data in np.array format
        # Xtrain, ytrain, class_label_pair, _ = get_training_data(training_set, training_anno_file)
        # annotationFileName = [dataset+"/2_training_annotations/2_training_anno_top.json.gz", dataset+"/2_training_annotations/2_training_anno_fine.json.gz"]
        annotationFileName = dataset + "/2_training_annotations/2_training_anno_top.json.gz"
        feature_names, ids, Xtrain, ytrain, class_label_pairs = read_dataset(training_set,
                                                                             annotationFileName=annotationFileName,
                                                                             TLS=TLS, class_label_pairs=None)

        # Drop flows with total_num_pkts < 2
        df = pd.DataFrame(data=Xtrain, columns=feature_names)
        df['total_num_pkts'] = df['num_pkts_in'] + df['num_pkts_out']
        df['label'] = ytrain
        df = df[~(df['total_num_pkts'] < 2)]
        y = df.pop('label')
        df.pop('total_num_pkts')

        # Split validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=ytrain)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Get name of each class to display in confusion matrix
        top_class_names = list(sorted(class_label_pairs.keys()))
        # fine_class_names = list(sorted(class_label_pairs_list[1].keys()))

        # Get model
        # Default Training Hyperparameters

        n_classes_top = len(top_class_names)
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

        # Model Definition
        X_train_1D = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
        X_val_1D = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1)
        model = self.train_model(X_train_1D, {anno: n_classes_top},
                                 filters=filters, kernel_size=kernel_size,
                                 strides=strides, dropout_rate=dropout_rate,
                                 CNN_layers=CNN_layers, clf_reg=clf_reg)

        print(model.summary())  # summarize layers
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])
        # Train the model
        history = model.fit(X_train_1D, one_hot(y_train, n_classes_top),
                            batch_size=n_batch,
                            epochs=n_epochs,
                            validation_data=(X_val_1D, one_hot(y_val, n_classes_top)))

    def train_model(self, X_train, OUTPUT,
                      filters=250,
                      kernel_size=3,
                      strides=1,
                      dropout_rate=0.,
                      CNN_layers=2,
                      clf_reg=1e-4):
        # Model Definition
        OUTPUTS = []
        # raw_inputs = Input(shape=(X_train.shape[1],))
        raw_inputs = Input(shape=(X_train.shape[1], 1,))
        xcnn = Conv1D(filters, (kernel_size),
                      input_shape=(X_train.shape[1], 1),
                      padding='same',
                      activation='relu',
                      strides=strides)(raw_inputs)

        xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv1D(filters,
                          (kernel_size),
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

        if 'top' in OUTPUT:
            print('top found in OUTPUT')
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

        return model


    def load_saved_model(self, loaded_model):
        X_train, X_test, y_train, y_test = self.prep_training_data()

        loaded_model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        score = loaded_model.evaluate(X_test, y_test, verbose=0)

        print('%s: %.2f%%' % (loaded_model.metrics_names[1], score[1]*100))

        return loaded_model
