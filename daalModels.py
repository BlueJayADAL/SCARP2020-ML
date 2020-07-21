import argparse
import psutil

from models.CNN1D import CNN1D
from models.CNN2D import CNN2D
from models.LR import LR
from models.ModelLoader import ModelLoader
from models.RNN import RNN
from models.daal_DF import daal_DF
from models.daal_LR import daal_LR
from models.daal_SVM import daal_SVM
from models.daal_kNN import daal_kNN
from models.ANN import ANN
from models.vino_ANN import vino_ANN

from models.kNN import kNN
from models.vino_CNN1D import vino_CNN1D
from models.vino_CNN2D import vino_CNN2D
from models.vino_RNN import vino_RNN
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
    elif args.model not in ["lr", "df", "svm", "knn", "daallr", "daaldf", "daalsvm", "daalknn", "ann", "rnn", "cnn1d", "cnn2d", "vinoann", "vinornn", "vinocnn1d", "vinocnn2d"]:
        print("Please select one of these for model: {lr, df, svm, knn, daallr, daaldf, daalsvm, daalknn, ann, rnn, cnn1d, cnn2d, vinoann, vinornn, vinocnn1d, vinocnn2d}. e.g. --model lr")
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

        cpu_reads = []
        p = psutil.Process()
        cpu_reads.append(p.cpu_percent(interval=None))


        # Setup data and labels from csv file
        data, labels = read_csv_dataset(args.dataset)
        cpu_reads.append(p.cpu_percent(interval=None))

        # Train respective model based on arguments provided from command

        # Handle LR Model:
        if args.model == 'lr':
            # Setup LR model
            lr = LR(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_LR', None)
                loaded_model = ml.load_sk_daal_model()
                lr.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report=lr.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()
        #hendles regular knn
        elif args.model == 'knn':
            # Setup knn model
            knn = kNN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_knn', None)
                loaded_model = ml.load_sk_daal_model()
                knn.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report=knn.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()

        # Handle SVM Model
        elif args.model == 'svm':
            # Setup SVM model
            svm = daal_SVM(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_SVM', None)
                loaded_model = ml.load_sk_daal_model()
                svm.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report=svm.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()

        # Handle DAAL LR Model
        elif args.model == 'daallr':
            # Setup LR model
            lr_daal = daal_LR(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_LR', None)
                loaded_model = ml.load_sk_daal_model()
                lr_daal.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report= lr_daal.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()

        # Handle DAAL DF Model
        elif args.model == 'daaldf':
            # Setup DF model
            df_daal = daal_DF(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_DF', None)
                loaded_model = ml.load_sk_daal_model()
                df_daal.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report=df_daal.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()

        # Handle DAAL SVM Model
        elif args.model == 'daalsvm':
            # Setup SVM model
            svm_daal = daal_SVM(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_SVM', None)
                loaded_model = ml.load_sk_daal_model()
                svm_daal.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report=svm_daal.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()

        # Handle DAAL KNN Model
        elif args.model == 'daalknn':
            knn_daal = daal_kNN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('daal_kNN', None)
                loaded_model = ml.load_sk_daal_model()
                knn_daal.load_saved_model(loaded_model)
            else:
                acc,tpr,far,report=knn_daal.train()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()


        # Non DAAL models can be found below

        # Handle ANN Model
        elif args.model == 'ann':
            # Setup ANN model
            ann_model = ANN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_ann', None)
                loaded_model = ml.load_keras_model()
                ann_model.load_saved_model(loaded_model)
            else:
                acc, tpr, far, report = ann_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nCPU Mean: ", str(cpu_mean),
                           "\nCPU Max: ", str(cpu_max), "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()

        # Handle RNN Model
        elif args.model == 'rnn':
            # Setup RNN model
            rnn_model = RNN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_ann', None)
                loaded_model = ml.load_keras_model()
                rnn_model.load_saved_model(loaded_model)
            else:
                rnn_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # Handle 1D-CNN Model
        elif args.model == 'cnn1d':
            # Setup 1D-CNN model
            cnn1d_model = CNN1D(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_cnn1d', None)
                loaded_model = ml.load_keras_model()
                acc, tpr, far, report = cnn1d_model.load_saved_model(loaded_model)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nAccuracy: ", str(acc), "\nTPR: ", str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()
            else:
                cnn1d_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # Handle 2D-CNN Model
        elif args.model == 'cnn2d':
            # Setup 2D-CNN model
            cnn2d_model = CNN2D(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('model_cnn2d', None)
                loaded_model = ml.load_keras_model()
                acc, tpr, far, report = cnn2d_model.load_saved_model(loaded_model)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nAccuracy: ", str(acc), "\nTPR: ",
                           str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()
            else:
                cnn2d_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # Handle VINO ANN Model
        elif args.model == 'vinoann':
            # Setup VINO ANN model
            vino_ann_model = vino_ANN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('vino_ann', None)
                net, execNet = ml.load_vino_model()
                vino_ann_model.load_saved_model(net, execNet)
            else:
                acc, tpr, far, report = vino_ann_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # Handle VINO RNN Model
        elif args.model == 'vinornn':
            # Setup VINO ANN model
            vino_rnn_model = vino_RNN(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('vino_rnn', None)
                net, execNet = ml.load_vino_model()
                vino_rnn_model.load_saved_model(net, execNet)
            else:
                vino_rnn_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # Handle VINO 1D-CNN Model
        elif args.model == 'vinocnn1d':
            # Setup VINO CNN1D model
            vino_cnn1d_model = vino_CNN1D(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('vino_cnn1d', None)
                net, execNet = ml.load_vino_model()
                acc, tpr, far, report = vino_cnn1d_model.load_saved_model(net, execNet)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nAccuracy: ", str(acc), "\nTPR: ",
                           str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()
            else:
                vino_cnn1d_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # Handle VINO 2D-CNN Model
        elif args.model == 'vinocnn2d':
            # Setup VINO CNN2D model
            vino_cnn2d_model = vino_CNN2D(data, labels)

            # Handle training / loading of model
            if args.load:
                ml = ModelLoader('vino_cnn2d', None)
                net, execNet = ml.load_vino_model()
                acc, tpr, far, report = vino_cnn2d_model.load_saved_model(net, execNet)
                results = open("results.txt", "a")
                outputs = ["Model: ", args.model, "\nDataset: ", args.dataset, "\nAccuracy: ", str(acc), "\nTPR: ",
                           str(tpr), "\nFAR: ",
                           str(far), "\n", str(report), "\n\n\n\n"]
                results.writelines(outputs)
                results.close()
            else:
                vino_cnn2d_model.train_model()
                cpu_reads.append(p.cpu_percent(interval=None))
                cpu_mean = sum(cpu_reads) / len(cpu_reads[1:])
                cpu_max = max(cpu_reads)
                print("Cpu Mean:", cpu_mean)
                print("Cpu Max:", cpu_max)

        # python daalModels.py --dataset NetML --model cnn2d --load false --runs 1


main()
