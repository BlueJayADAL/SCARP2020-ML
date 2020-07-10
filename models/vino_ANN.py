import os
import time
import tensorflow as tf

import numpy as np
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

from models.ModelLoader import ModelLoader


class vino_ANN:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        # Must run `pycharm-community` command from terminal to initialize openvino in pycharm

    def train(self,
              work_dir='models/saved/'):
        """
        Loads an ANN model to be used for OpenVINO
        """
        batch = len(self.data)
        dWidth = self.data.shape[0]

        # Start train timing
        startTime = time.time()

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model_ann', None)
        loaded_model = ml.load_keras_model()
        ml.save_keras_as_vino()

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %svino_ann.pb --input_shape [%d,%d] --output_dir %s" % (
        work_dir, batch, dWidth, work_dir)

        print(generateCommand)
        os.system(generateCommand)

        # Make predictions
        predictResultTest = loaded_model.predict_classes(self.data)

        # End train timing
        endTime = time.time()

        print("Test (VINO Accelerated Artificial Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
        correctTest = np.sum(self.labels == predictResultTest.flatten())
        print("Test accuracy: ", float(correctTest)/len(self.labels)*100)


    def load_saved_model(self, loaded_model):
        # TODO load saved model
        pass