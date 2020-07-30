import joblib
from tensorflow.python.keras.models import model_from_json


class ModelLoader:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model

    def save_keras_model(self,
                         save_dir='matt/models/saved/'):
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
                         load_dir='matt/models/saved/'):
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

        return loaded_model

    def save_daal_model(self,
                        save_dir='matt/models/saved/'):
        """
        Saves a DAAL model to disk memory
        """
        outPKL = "%s%s.pkl" % (save_dir, self.filename)
        joblib.dump(self.model, outPKL)

    def load_daal_model(self,
                        load_dir='matt/models/saved/'):
        """
        Loads a DAAL model from disk memory
        """
        inPKL = "%s%s.pkl" % (load_dir, self.filename)
        loaded_model = joblib.load(inPKL)

        return loaded_model
