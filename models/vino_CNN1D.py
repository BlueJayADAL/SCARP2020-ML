import os
import time

import numpy as np

from models.CNN1D import CNN1D
from models.ModelLoader import ModelLoader
from openvino.inference_engine import IECore, IENetwork, IEPlugin

from utils.helper import convertToOneHot
from utils.helper2 import one_hot


class vino_CNN1D:
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

        # Must run `pycharm-community` command from terminal to initialize openvino in pycharm

    def prep_data(self):
        """
        Sets X_train_1D, X_test_1D, y_train, y_test, and  variables for training / testing.
        Run this method to reset values
        """

        # Create cnn1d model to pull data from
        cnn1d_data = CNN1D(self.data, self.labels)

        # Clone data from cnn1d model
        self.X_train_1D = cnn1d_data.X_train_1D
        self.X_test_1D = cnn1d_data.X_test_1D
        self.y_train = cnn1d_data.y_train
        self.y_test = cnn1d_data.y_test

        self.top_class_names = cnn1d_data.top_class_names
        self.n_classes_top = cnn1d_data.n_classes_top

    def train_model(self,
                    work_dir='models/saved/'):
        """
        Loads a CNN1D model to be used for OpenVINO
        """
        len = self.X_test_1D.shape

        # Start train timing
        startTime = time.time()

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model_cnn1d', None)
        loaded_model = ml.load_keras_model()
        ml.save_keras_as_vino()

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %svino_cnn1d.pb --input_shape [%d,%d,%d] --output_dir %s" % (
        work_dir, len[0], len[1], len[2], work_dir)

        print(generateCommand)
        os.system(generateCommand)

        # End train timing
        endTime = time.time()

        print("Test (VINO Convolutional 2D Neural Network) elapsed in %.3f seconds" % (endTime - startTime))

        self.load_saved_model(None)


    def load_saved_model(self, loaded_model,
                         work_dir='models/saved/'):
        modelXML = work_dir + "vino_cnn1d.xml"
        modelBin = work_dir + "vino_cnn1d.bin"

        ie = IECore()
        net = ie.read_network(model=modelXML, weights=modelBin)
        execNet = ie.load_network(network=net, device_name="CPU")

        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        res = execNet.infer(inputs={input_blob: self.X_test_1D})

        # Gets the output layer from neural network
        res = res['top_level_output/Softmax']

        res = convertToOneHot(res)

        correctTest = np.sum(one_hot(self.y_test, self.n_classes_top) == res) / float(self.n_classes_top)

        testAccu = float(correctTest) / len(self.y_test) * 100
        print("VINO CNN1D Test Accuracy: %.3f%%" % testAccu)
