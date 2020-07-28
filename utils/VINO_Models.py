import os

from models.ModelLoader import ModelLoader
from openvino.inference_engine import IECore, IENetwork

from utils.helper import convertToDefault


class vino_CNN_1D:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir

    def train(self,
                    work_dir='models/saved/'):
        """
        Loads a CNN1D model to be used for OpenVINO
        """

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader("model", None)
        loaded_model = ml.load_keras_model(load_dir=self.load_dir)
        ml.save_keras_as_vino(save_dir=self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d,%d] --output_dir %s" % (
            self.save_dir,
            self.input_shape[0], self.input_shape[1], self.input_shape[2],
            self.save_dir)

        # Run vino model creation command
        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Reshape data for input
        data = data.reshape(data.shape[0], data.shape[1], 1)

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: data})

        # Get prediction results
        res = res[list(res.keys())[0]]

        return convertToDefault(res)

class vino_CNN_2D:
    def __init__(self, input_shape, save_dir, load_dir):
        self.input_shape = input_shape
        self.save_dir = save_dir
        self.load_dir = load_dir


    def train(self):
        """
        Loads a CNN2D model to be used for OpenVINO
        """

        # Clone data from cnn2d model
        self.X_train_2D = cnn2d_data.X_train_2D
        self.X_test_2D = cnn2d_data.X_test_2D.reshape(cnn2d_data.X_test_2D.shape[0], 1, 5, 10)
        self.y_train = cnn2d_data.y_train
        self.y_test = cnn2d_data.y_test


    def train_model(self):
        """
        Loads an CNN2D model to be used for OpenVINO
        """

        # Convert ANN model to binary .pb file and save
        ml = ModelLoader('model', None)
        loaded_model = ml.load_keras_model(self.load_dir)
        ml.save_keras_as_vino(self.save_dir)

        # Run OpenVINO's Model Optimizer TensorFlow script (Have script [mo_tf.py] in main directory with DAAL scripts)
        generateCommand = "mo_tf.py --input_model %smodel.pb --input_shape [%d,%d,%d,%d] --output_dir %s" % (
            self.save_dir,
            self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3],
            self.save_dir)
        # 22880, 5, 10, 1

        print(generateCommand)
        os.system(generateCommand)

    def classify(self, data):
        # Load vino model
        ml = ModelLoader('model', None)
        net, execNet = ml.load_vino_model(load_dir=self.save_dir)

        # Get input and outputs of model
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))

        # Input data into model for predicting
        res = execNet.infer(inputs={input_blob: self.X_test_2D})

        # Get prediction results
        res = res['top_level_output/Softmax']

        return convertToDefault(res)
