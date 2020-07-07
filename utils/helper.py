import numpy as np
import pandas as pd

def read_csv_dataset(fileName):
    # Read dataset from csv and shuffle it into random order
    data = pd.read_csv(fileName).sample(frac=1)
    labels = data['label']

    data.drop('label', axis=1, inplace=True)

    return data, labels
