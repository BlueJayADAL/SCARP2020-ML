from tensorflow.python.keras.models import model_from_json


class ModelLoader:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model

    def save_keras_model(self):
        """
        Saves a keras model to disk memory
        """
        model_json = self.model.to_json()
        with open('models/saved/' + self.filename + '.json', 'w') as json_file:
            json_file.write(model_json)

        # Save weights into model
        self.model.save_weights('models/saved/' + self.filename + '.h5')

        print('Successfully saved model to disk as ' + self.filename + '.json!')

    def load_keras_model(self):
        """
        Loads a keras model from disk memory
        """
        json_file = open('models/saved/' + self.filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # Load weights into model
        loaded_model.load_weights('models/saved/' + self.filename + '.h5')

        print('Successfully loaded model ' + self.filename + '.json from disk!')

        return loaded_model
