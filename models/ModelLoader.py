import joblib
import tensorflow as tf
from tensorflow.python.data.experimental.ops.optimization import model

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from utils.helper import freeze_session, wrap_frozen_graph


class ModelLoader:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model

    def save_keras_model(self,
                         save_dir='models/saved/'):
        """
        Saves a keras model to disk memory
        """
        model_json = self.model.to_json()
        with open(save_dir + self.filename + '.json', 'w') as json_file:
            json_file.write(model_json)

        # Save weights into model
        self.model.save_weights(save_dir + self.filename + '.h5')

        print('Successfully saved model to disk as ' + self.filename + '.json!')

    def load_keras_model(self,
                         load_dir='models/saved/'):
        """
        Loads a keras model from disk memory
        """
        json_file = open(load_dir + self.filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # Load weights into model
        loaded_model.load_weights(load_dir + self.filename + '.h5')

        print('Successfully loaded model ' + self.filename + '.json from disk!')

        self.model = loaded_model

        return self.model

    def save_daal_model(self,
                        save_dir='models/saved/'):
        """
        Saves a DAAL model to disk memory
        """
        outPKL = "%s%s.pkl" % (save_dir, self.filename)
        joblib.dump(self.model, outPKL)

    def load_daal_model(self,
                        load_dir='models/saved/'):
        """
        Loads a DAAL model from disk memory
        """
        inPKL = "%s%s.pkl" % (load_dir, self.filename)
        self.model = joblib.load(inPKL)

        return self.model

    def save_keras_as_vino(self,
                           save_dir='models/saved/',):
        session = K.get_session

        self.model.compile(optimizer='adam',
                             loss='binary_crossentropy',  # Tries to minimize loss
                             metrics=['accuracy'])

        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(x=(tf.TensorSpec(self.model.inputs[0].shape, self.model.inputs[0].dtype),
                                                         tf.TensorSpec(self.model.inputs[1].shape, self.model.inputs[1].dtype),
                                                         tf.TensorSpec(self.model.inputs[2].shape, self.model.inputs[2].dtype)))

        frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir="./frozen_models",
                          name="complex_frozen_graph.pb",
                          as_text=False)

        # Load frozen graph using TensorFlow 1.x functions
        with tf.io.gfile.GFile("./frozen_models/complex_frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["x:0", "x_1:0", "x_2:0"],
                                        outputs=["Identity:0", "Identity_1:0"],
                                        print_graph=True)

        # frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in self.model.outputs])
        tf.io.write_graph(frozen_func, save_dir, "vino_ann.pb", as_text=False)
