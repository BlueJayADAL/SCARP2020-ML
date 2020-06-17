import argparse

from matt.MLP import MLP
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

    mlp = MLP(training_set, training_anno_file, test_set)
    mlp.train_model()


if __name__ == "__main__":
    main()
