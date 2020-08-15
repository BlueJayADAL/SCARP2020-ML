import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Activation

from utils.helper import one_hot


class AlexNet:
    """Incomplete implementation of AlexNet
        Unable to find a way to adjust base input size"""

    def __init__(self, input_shape,
                 n_classes,
                 filters=250,
                 kernel_size=3,
                 strides=1,
                 dense_units=128,
                 dropout_rate=0.,
                 clf_reg=1e-4):

        # (3) Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(input_shape[0] * input_shape[1] * input_shape[2],)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))

        self.model = model
        self.n_classes = n_classes

    def train(self, X_train, y_train, X_val, y_val, n_batch, n_epochs, learning_rate, decay_rate, save_dir):
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
            X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)
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
        if len(data.shape) > 2:
            return self.model.predict(data.reshape(-1, data.shape[1], data.shape[2], 1))
        else:
            return self.model.predict(data)
