import numpy as np
import pandas as pd
from sklearn import metrics


def read_csv_dataset(fileName):
    # Read dataset from csv and shuffle it into random order
    data = pd.read_csv(fileName).sample(frac=1)
    labels = data['label']

    data.drop('label', axis=1, inplace=True)

    return data, labels


def collect_statistics(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    detectionRate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    falseAlarmRate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    correct = np.sum(y_true == y_pred)
    accu = float(correct) / len(y_true) * 100

    return detectionRate, falseAlarmRate, accu
