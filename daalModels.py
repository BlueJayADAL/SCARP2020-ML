import argparse

from models.daal_DF import daal_DF
from models.daal_LR import daal_LR
from models.daal_SVM import daal_SVM
from models.ANN import ANN
from utils.helper import read_csv_dataset


def main():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline",
                                     add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset")
    parser.add_argument('-m', '--model', action="store")

    args = parser.parse_args()

    if args.dataset is None or args.dataset not in ["NetML", "CICIDS"]:
        print("No valid dataset set or annotations found!")
        return
    elif args.model not in ["lr", "df", "svm", "ann"]:
        print("Please select one of these for model: {lr, df, svm, ann}. e.g. --model lr")
        return
    else:
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
            lr_daal = daal_LR(data, labels)
            lr_daal.train()
        elif args.model == 'df':
            df_daal = daal_DF(data, labels)
            df_daal.train()
        elif args.model == 'svm':
            svm_daal = daal_SVM(data, labels)
            svm_daal.train()

        # Non DAAL models can be found below

        elif args.model == 'ann':
            ann_model = ANN(data, labels)
            ann_model.train()

        # python daalModels.py --dataset NetML --model lr


main()
