from sklearn import preprocessing, metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from matt.utils.helper import get_training_data


class MLP:
    def __init__(self, training_set, training_anno_file, test_set):
        self.training_set = training_set
        self.training_anno_file = training_anno_file
        self.test_set = test_set

    def train_model(self):
        # Get training data in np.array format
        X_train, y_train, class_label_pair, X_train_ids = get_training_data(self.training_set, self.training_anno_file)

        X, y = make_classification(n_samples=100, random_state=1)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            stratify=y_train)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create MLP classifier
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)

        cm = metrics.confusion_matrix(y_test, y_pred)
        detectionRate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        falseAlarmRate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        print("TPR: \t\t\t{:.5f}".format(detectionRate))
        print("FAR: \t\t\t{:.5f}".format(falseAlarmRate))

        print("Mean ACC: \t\t{:.5f}".format(clf.score(X_test_scaled, y_test)))
