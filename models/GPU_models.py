import numpy as np
import tensorflow as tf 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Conv2D,\
    MaxPooling2D

def one_hot(y_, n_classes=None):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    if n_classes is None:
        n_classes = int(int(max(y_))+1)
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


class ANN:
    """docstring for ANN"""

    def __init__(self, input_shape,
                 n_classes,
                 dense_units=128,
                 dropout_rate=0.,
                 clf_reg=1e-4):
        # Model Definition
        # raw_inputs = Input(shape=(X_train.shape[1],1,))
        raw_inputs = Input(shape=input_shape)

        xann = Dense(dense_units, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                     bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                     activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                     name='FC1_layer')(raw_inputs)

        if dropout_rate != 0:
            xann = Dropout(dropout_rate)(xann)

        # we flatten for dense layer
        xann = Flatten()(xann)

        top_level_predictions = Dense(n_classes, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                      name='top_level_output')(xann)

        model = Model(inputs=raw_inputs, outputs=top_level_predictions)
        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        print(self.model.summary())  # summarize layers
        plot_model(self.model, to_file=save_dir + '/model.png')  # plot graph
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                           metrics=['accuracy'])
        # Train the model
        return self.model.fit(X_train, one_hot(y_train, self.n_classes),
                              batch_size=n_batch,
                              epochs=n_epochs,
                              validation_data=(X_val, one_hot(y_val, self.n_classes)))

    def classify(self, data):
        if len(data.shape) < 3:
            X_test_1D = data.reshape(-1, data.shape[1], 1)
        else:
            X_test_1D = data

        return self.model.predict(X_test_1D)


class CNN_1D:
    """docstring for CNN_1D"""
    def __init__(self, input_shape, 
                    n_classes,                  
                    filters=250, 
                    kernel_size=3,
                    strides=1,
                    dense_units=128,
                    dropout_rate=0., 
                    CNN_layers=2, 
                    clf_reg=1e-4):
        # Model Definition
        #raw_inputs = Input(shape=(X_train.shape[1],1,))
        raw_inputs = Input(shape=input_shape)
        print("INPUT SHAPE:", input_shape)
        xcnn = Conv1D(filters, 
                        (kernel_size),
                        padding='same',
                        activation='relu',
                        strides=strides,
                        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                        name='Conv1D_1')(raw_inputs)

        xcnn = BatchNormalization()(xcnn)                 
        xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv1D(filters,
                        (kernel_size),
                        padding='same',
                        activation='relu',
                        strides=strides,
                        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                        name='Conv1D_'+str(i+1))(xcnn)

            xcnn = BatchNormalization()(xcnn)                 
            xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

            if dropout_rate != 0:
                xcnn = Dropout(dropout_rate)(xcnn)  

        # we flatten for dense layer
        xcnn = Flatten()(xcnn)
        
        xcnn = Dense(dense_units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                name='FC1_layer')(xcnn)
        
        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn) 
        
        xcnn = Dense(dense_units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                name='FC2_layer')(xcnn)            
         
        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn) 


        top_level_predictions = Dense(n_classes, activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='top_level_output')(xcnn)


        model = Model(inputs=raw_inputs, outputs=top_level_predictions)
        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        if len(X_train.shape) < 3:
            X_train_1D = X_train.reshape(-1,X_train.shape[1],1)
            X_val_1D = X_val.reshape(-1,X_val.shape[1],1)
        else:
            X_train_1D = X_train
            X_val_1D = X_val
        print(self.model.summary()) # summarize layers
        plot_model(self.model, to_file=save_dir+'/model.png') # plot graph
        self.model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                        metrics=['accuracy'])
        # Train the model
        return self.model.fit(X_train_1D, one_hot(y_train, self.n_classes), 
                            batch_size=n_batch, 
                            epochs=n_epochs, 
                            validation_data=(X_val_1D, one_hot(y_val, self.n_classes)))

    

    def classify(self, data):
        if len(data.shape) < 3:
            X_test_1D = data.reshape(-1,data.shape[1],1)
        else:
            X_test_1D = data        

        return self.model.predict(X_test_1D)


class CNN_2D:
    """docstring for CNN_2D"""
    def __init__(self, input_shape, 
                    n_classes,                  
                    filters=250, 
                    kernel_size=3,
                    strides=1,
                    dense_units=128,
                    dropout_rate=0., 
                    CNN_layers=2, 
                    clf_reg=1e-4):
        # Model Definition
        #raw_inputs = Input(shape=(X_train.shape[1],1,))
        raw_inputs = Input(shape=input_shape)
        xcnn = Conv2D(filters, 
                        (kernel_size),
                        padding='same',
                        activation='relu',
                        strides=strides,
                        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                        name='Conv2D_1')(raw_inputs)

        xcnn = BatchNormalization()(xcnn)                 
        xcnn = MaxPooling2D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv2D(filters,
                        (kernel_size),
                        padding='same',
                        activation='relu',
                        strides=strides,
                        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                        name='Conv2D_'+str(i+1))(xcnn)

            xcnn = BatchNormalization()(xcnn)                 
            xcnn = MaxPooling2D(pool_size=2, padding='same')(xcnn)

            if dropout_rate != 0:
                xcnn = Dropout(dropout_rate)(xcnn)  

        # we flatten for dense layer
        xcnn = Flatten()(xcnn)
        
        xcnn = Dense(dense_units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                name='FC1_layer')(xcnn)
        
        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn) 
        
        xcnn = Dense(dense_units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                name='FC2_layer')(xcnn)            
         
        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn) 


        top_level_predictions = Dense(n_classes, activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='top_level_output')(xcnn)


        model = Model(inputs=raw_inputs, outputs=top_level_predictions)
        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
            X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)
        print(self.model.summary()) # summarize layers
        plot_model(self.model, to_file=save_dir+'/model.png') # plot graph
        self.model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                        metrics=['accuracy'])
        # Train the model
        return self.model.fit(X_train, one_hot(y_train, self.n_classes), 
                            batch_size=n_batch, 
                            epochs=n_epochs, 
                            validation_data=(X_val, one_hot(y_val, self.n_classes)))

    

    def classify(self, data):
        if len(data.shape) > 2:
            return self.model.predict(data.reshape(-1, data.shape[1], data.shape[2], 1))
        else:
            return self.model.predict(data)

class LSTM:
    """docstring for LSTM"""
    def __init__(self, input_shape, 
                    n_classes,                  
                    dense_units=128,
                    dropout_rate=0., 
                    LSTM_layers=2,
                    LSTM_units=128,
                    lstm_reg=1e-4, 
                    clf_reg=1e-4):
        # Model Definition
        #raw_inputs = Input(shape=(X_train.shape[1],1,))
        raw_inputs = Input(shape=input_shape)
        if LSTM_layers == 1:
            xlstm = tf.keras.layers.LSTM(LSTM_units, return_sequences=False,
                        kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
            if dropout_rate != 0:
                xlstm = Dropout(dropout_rate)(xlstm)        
        else:   
            xlstm = tf.keras.layers.LSTM(LSTM_units, return_sequences=True,
                        kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(raw_inputs)
            if dropout_rate != 0:
                xlstm = Dropout(dropout_rate)(xlstm)    

            for i in range(1, LSTM_layers-1):
                xlstm = tf.keras.layers.LSTM(LSTM_units, return_sequences=True,
                            kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)    
                if dropout_rate != 0:
                    xlstm = Dropout(dropout_rate)(xlstm)

            xlstm = tf.keras.layers.LSTM(LSTM_units, return_sequences=False,
                        kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                        activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xlstm)
            if dropout_rate != 0:
                xlstm = Dropout(dropout_rate)(xlstm)


        top_level_predictions = Dense(n_classes, activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='top_level_output')(xlstm)


        model = Model(inputs=raw_inputs, outputs=top_level_predictions)
        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        print(self.model.summary()) # summarize layers
        plot_model(self.model, to_file=save_dir+'/model.png') # plot graph
        self.model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                        metrics=['accuracy'])
        # Train the model
        return self.model.fit(X_train, one_hot(y_train, self.n_classes), 
                            batch_size=n_batch, 
                            epochs=n_epochs, 
                            validation_data=(X_val, one_hot(y_val, self.n_classes)))

    

    def classify(self, data):
        return self.model.predict(data)


class CNN_LSTM:
    """docstring for 1D_CNN_LSTM"""
    def __init__(self, input_shape, 
                    n_classes,                  
                    filters=32, 
                    kernel_size=5,
                    strides=1,
                    dense_units=200,
                    dropout_rate=0., 
                    LSTM_units=200,
                    lstm_reg=1e-4, 
                    clf_reg=1e-4):
        # Model Definition
        #raw_inputs = Input(shape=(X_train.shape[1],1,))
        raw_inputs = Input(shape=input_shape)
        xcnn = Conv1D(filters, 
                        (kernel_size),
                        padding='same',
                        activation='relu',
                        strides=strides,
                        kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                        bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                        activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                        name='Conv1D_1')(raw_inputs)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        xcnn = Conv1D(filters,
                    (kernel_size),
                    padding='same',
                    activation='relu',
                    strides=strides,
                    kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                    bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                    activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                    name='Conv1D_2')(xcnn)

        xcnn = BatchNormalization()(xcnn)                 
        xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)  

        xlstm = tf.keras.layers.LSTM(LSTM_units, return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                    recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                    bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                    activity_regularizer=tf.keras.regularizers.l1(lstm_reg))(xcnn)
        
        if dropout_rate != 0:
            xlstm = Dropout(dropout_rate)(xlstm)        


        # we flatten for dense layer
        xlstm = Flatten()(xlstm)
        
        xlstm = Dense(dense_units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                name='FC1_layer')(xlstm)
        
        if dropout_rate != 0:
            xlstm = Dropout(dropout_rate)(xlstm) 
        
        xlstm = Dense(dense_units, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                name='FC2_layer')(xlstm)            
         
        if dropout_rate != 0:
            xlstm = Dropout(dropout_rate)(xlstm) 


        top_level_predictions = Dense(n_classes, activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
            bias_regularizer=tf.keras.regularizers.l2(clf_reg),
            activity_regularizer=tf.keras.regularizers.l1(clf_reg),
            name='top_level_output')(xlstm)


        model = Model(inputs=raw_inputs, outputs=top_level_predictions)
        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        if len(X_train.shape) < 3:
            X_train_1D = X_train.reshape(-1,X_train.shape[1],1)
            X_val_1D = X_val.reshape(-1,X_val.shape[1],1)
        else:
            X_train_1D = X_train
            X_val_1D = X_val
        print(self.model.summary()) # summarize layers
        plot_model(self.model, to_file=save_dir+'/model.png') # plot graph
        self.model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                        metrics=['accuracy'])
        # Train the model
        return self.model.fit(X_train_1D, one_hot(y_train, self.n_classes), 
                            batch_size=n_batch, 
                            epochs=n_epochs, 
                            validation_data=(X_val_1D, one_hot(y_val, self.n_classes)))

    

    def classify(self, data):
        if len(data.shape) < 3:
            X_test_1D = data.reshape(-1,data.shape[1],1)
        else:
            X_test_1D = data
        return self.model.predict(X_test_1D)
