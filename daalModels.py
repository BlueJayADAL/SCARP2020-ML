import argparse

from models.ModelLoader import ModelLoader
from models.daal_DF import daal_DF
from models.daal_LR import daal_LR
from models.daal_SVM import daal_SVM
from models.ANN import ANN
from utils.helper import read_csv_dataset
from distutils import util


def main():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline",
                                     add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset")
    parser.add_argument('-m', '--model', action="store")
    parser.add_argument('-l', '--load', action="store")
    parser.add_argument('-r', '--runs', action="store", help="how many runs?")
    args = parser.parse_args()

    if args.dataset is None or args.dataset not in ["NetML", "CICIDS"]:
        print("No valid dataset set or annotations found!")
        return
    elif args.model not in ["lr", "df", "svm", "ann"]:
        print("Please select one of these for model: {lr, df, svm, ann}. e.g. --model lr")
        return
    elif args.load not in ["true", "false"]:
        args.load = False
    else:
        # run_num is amout of times model is going to be tested
        run_num = args.runs
        run_num = int(run_num)
        if args.load is not False:
            # Convert load argument to boolean
            args.load = bool(util.strtobool(args.load))
        # If all arguments are succsessful

        # Choose correct dataset via paths below
        if args.dataset == "NetML":
            args.dataset = "./data/NetML_enc_filtered_top50.csv"
        else:
            args.dataset = "./data/CICIDS2017_enc_filtered_top50.csv"

        # Setup data and labels from csv file
        data, labels = read_csv_dataset(args.dataset)

        # Train respective model
        if args.model == 'lr':
            # Setup LR model
            lr_daal = daal_LR(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_LR', None)
                loaded_model = ml.load_daal_model()
                lr_daal.load_saved_model(loaded_model)
            else:
                lr_daal.train()
        elif args.model == 'df':
            # Setup DF model
            df_daal = daal_DF(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_DF', None)
                loaded_model = ml.load_daal_model()
                df_daal.load_saved_model(loaded_model)
            else:
                df_daal.train()
        elif args.model == 'svm':
            # Setup SVM model
            svm_daal = daal_SVM(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_SVM', None)
                loaded_model = ml.load_daal_model()
                svm_daal.load_saved_model(loaded_model)
            else:
                svm_daal.train()

        # Non DAAL models can be found below

        elif args.model == 'ann':
            # Setup ANN model
            ann_model = ANN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_ann', None)
                loaded_model = ml.load_keras_model()
                ann_model.load_saved_model(loaded_model)
            else:
                ann_model.train()

        # python daalModels.py --dataset NetML --model lr --load false --runs 1


main()
