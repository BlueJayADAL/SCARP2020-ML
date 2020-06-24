import argparse
import sys
import os.path
import time

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matt.models.ANN import ANN
from models.CNN import CNN
from matt.models.MLP import MLP
from matt.models.ModelLoader import ModelLoader
from matt.utils.helper import *


def main():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline",
                                     add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset path")
    parser.add_argument('-a', '--anno', action="store", help="Annoation level: {top, mid, fine}")
    parser.add_argument('-s', '--submit', action="store",
                        help="{test-std, test-challenge, both} Select which set to submit")

    args = parser.parse_args()

    if args.dataset is None or not os.path.isdir(args.dataset) or args.anno is None:
        print("No valid dataset set or annotations found!")
        return
    elif args.submit is not None and args.submit not in ["test-std", "test-challenge", "both"]:
        print("Please select which set to submit: {test-std, test-challenge, both}")
        return
    elif args.anno not in ["top", "mid", "fine"]:
        print("Please select one of these for annotations: {top, mid, fine}. e.g. --anno top")
        return
    elif args.anno == "mid" and (args.dataset.find("NetML") > 0 or args.dataset.find("CICIDS2017") > 0):
        print(
            "NetML and CICIDS2017 datasets cannot be trained with mid-level annotations. Please use either top or fine.")
        return
    else:
        training_set = args.dataset + "/2_training_set"
        training_anno_file = args.dataset + "/2_training_annotations/2_training_anno_" + args.anno + ".json.gz"
        test_set = args.dataset + "/1_test-std_set"
        challenge_set = args.dataset + "/0_test-challenge_set"

    #mlp = MLP(training_set, training_anno_file, test_set)
    #mlp.train_model()

    # Controls debug mode
    load = True
    type = 'ann'

    start_time = time.time()

    ann = ANN(training_set, training_anno_file, test_set)
    cnn = CNN(training_set, training_anno_file, test_set)
    mlp = MLP(training_set, training_anno_file, test_set)

    if load:
        if type is 'ann':
            ml = ModelLoader('ann_model', None)
            loaded_model = ml.load_keras_model()
            ann_model = ann.load_saved_model(loaded_model)
        elif type is 'cnn':
            pass
        elif type is 'mlp':
            pass
    else:
        if type is 'ann':
            ann_model = ann.train_model()
            ml = ModelLoader('ann_model', ann_model)
            ml.save_keras_model()
        elif type is 'cnn':
            cnn_model = cnn.train_model()
        elif type is 'mlp':
            mlp_model = mlp.train_model()

    end_time = time.time()

    print("Total program execution took " + str(round(end_time - start_time, 3)) + " seconds!")


if __name__ == "__main__":
    main()
