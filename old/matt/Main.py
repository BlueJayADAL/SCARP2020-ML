import argparse
import os.path
import time

from old.matt.models.CNN1D import CNN1D
from old.matt.models.CNN2D import CNN2D
from old.matt.models.ANN import ANN
from old.matt.models.MLP import MLP
from old.matt.models.ModelLoader import ModelLoader
from old.matt.utils.helper import *


def main():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline",
                                     add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset path")
    parser.add_argument('-a', '--anno', action="store", help="Annoation level: {top, mid, fine}")
    parser.add_argument('-s', '--submit', action="store",
                        help="{test-std, test-challenge, both} Select which set to submit")

    args = parser.parse_args()

    if args.dataset is None or not os.path.isdir("matt/" + args.dataset) or args.anno is None:
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
        training_set = "matt/" + args.dataset + "/2_training_set"
        training_anno_file = "matt/" + args.dataset + "/2_training_annotations/2_training_anno_" + args.anno + ".json.gz"
        test_set = "matt/" + args.dataset + "/1_test-std_set"
        challenge_set = "matt/" + args.dataset + "/0_test-challenge_set"

    #python3 matt/Main.py --dataset ./data/NetML --anno top --submit test-std

    #mlp = MLP(training_set, training_anno_file, test_set)
    #mlp.train_model()

    # Controls debug mode
    load = False
    type = 'cnn1d'

    start_time = time.time()

    ann = ANN(training_set, training_anno_file, test_set)
    cnn1d = CNN1D(training_set, training_anno_file, test_set)
    cnn2d = CNN2D(training_set, training_anno_file, test_set)
    mlp = MLP(training_set, training_anno_file, test_set)

    if load:
        if type is 'mlp':
            pass
            ml = ModelLoader('ann_model', None)
            loaded_model = ml.load_keras_model()
            ann_model = ann.load_saved_model(loaded_model)
        elif type is 'cnn1d':
            ml = ModelLoader('cnn1d_model', None)
            loaded_model = ml.load_keras_model()
            cnn1d_model = cnn1d.load_saved_model(loaded_model)
        elif type is 'cnn2d':
            ml = ModelLoader('cnn2d_model', None)
            loaded_model = ml.load_keras_model()
            cnn2d_model = cnn2d.load_saved_model(loaded_model)
        elif type is 'mlp':
            pass
    else:
        if type is 'ann':
            ann_model = ann.train_model()
            ml = ModelLoader('ann_model', ann_model)
            ml.save_keras_model()
        elif type is 'cnn1d':
            cnn1d_model = cnn1d.create_model()
            ml = ModelLoader('ann_model', cnn1d_model)
            ml.save_keras_model()
        elif type is 'cnn2d':
            cnn2d_model = cnn2d.create_model()
            ml = ModelLoader('ann_model', cnn2d_model)
            ml.save_keras_model()
        elif type is 'mlp':
            mlp_model = mlp.train_model()

    end_time = time.time()

    print("Total program execution took " + str(round(end_time - start_time, 3)) + " seconds!")


if __name__ == "__main__":
    main()
