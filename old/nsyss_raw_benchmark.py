import os
import time as t
import numpy as np

from memory_profiler import memory_usage
import psutil

from sklearn.model_selection import StratifiedKFold

from utils.helper import plot_confusion_matrix, plotLoss, saveModel, read_raw_from_csv
from models.GPU_models import CNN_1D, CNN_2D, LSTM, CNN_LSTM
from models.sklearn_models import LR, RF, SVM, MLP, kNN


# Set hyperparameters Globally
learning_rate = 1e-3
decay_rate = 1e-5
dropout_rate = 0.5
n_batch = 100
n_epochs = 1  # Loop 1000 times on the dataset
filters = 128
kernel_size = 4
strides = 1
CNN_layers = 2
clf_reg = 1e-5
# ML parameters
n_neighbors = 5     # kNN
n_estimators = 100  # RF
max_depth = 10      # RF
C = 1.0             # SVM
svm_kernel = 'rbf'  # SVM
mlp_solver = 'adam'     # MLP
mlp_hidden_units = 128  # MLP

#@profile
def profile(dataset, modelname, save_dict, save_dir, N, M, num_folds=10):
    # Performance data
    Performance = {}
    Performance["t_train"] = []
    Performance["t_classify"] = []
    Performance["acc"] = []
    Performance["tpr"] = []
    Performance["far"] = []

    cpu_reads = []
    p = psutil.Process()
    cpu_reads.append(p.cpu_percent(interval=None))

    t_prep = t.time()

    # Read data
    #dataset = "NetML" # NetML or CICIDS2017
    print("Reading raw data from csv ...")
    X, y, class_label_pair = read_raw_from_csv(dataset, anno="top")

    # Truncate to selected values of N and M
    X = X[:,:N,:M]

    # Normalize the data
    X /= 255
    Performance["preprocessing time"] = t.time()-t_prep
    

    # Arrange folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 0
    for train_index, test_index in skf.split(X, y):
        cpu_reads.append(p.cpu_percent(interval=None))
        fold_no += 1
        save_dir_k = save_dir + '/{}'.format(fold_no)
        os.makedirs(save_dir_k)
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Init the model
        if modelname == "1D_CNN":
            model = CNN_1D(input_shape=(X_train.shape[1],X_train.shape[2]), 
                    n_classes=2,                  
                    filters=128, 
                    kernel_size=3,
                    strides=1,
                    dense_units=128,
                    dropout_rate=0.5, 
                    CNN_layers=2, 
                    clf_reg=1e-4)

        elif modelname == "2D_CNN":
            model = CNN_2D(input_shape=(X_train.shape[1],X_train.shape[2],1), 
                    n_classes=2,                  
                    filters=128, 
                    kernel_size=3,
                    strides=1,
                    dense_units=128,
                    dropout_rate=0.5, 
                    CNN_layers=2, 
                    clf_reg=1e-4)

        elif modelname == "LSTM":
            model = LSTM(input_shape=(X_train.shape[1],X_train.shape[2]), 
                    n_classes=2,                  
                    dense_units=128,
                    dropout_rate=0.5, 
                    LSTM_layers=2,
                    LSTM_units=128,
                    lstm_reg=1e-4, 
                    clf_reg=1e-4)
        
        elif modelname == "CNN+LSTM":
            # Reference to model : https://www.ieee-security.org/TC/SPW2019/DLS/doc/06-Marin.pdf
            model = CNN_LSTM(input_shape=(X_train.shape[1],X_train.shape[2]), # Model of "Deep in the Dark paper"
                    n_classes=2,                  
                    dropout_rate=0.5, 
                    lstm_reg=1e-4,
                    clf_reg=1e-4)

        elif modelname == "LR":
            model = LR()

        elif modelname == "kNN":
            model = kNN(n=n_neighbors)

        elif modelname == "RF":
            model = RF(n=n_estimators, m=max_depth)

        elif modelname == "SVM":
            model = SVM(C=C, kernel=svm_kernel)
        
        elif modelname == "MLP":
            model = MLP(solver=mlp_solver, hidden_units=mlp_hidden_units)

        else:
            return

        # Train the model
        if modelname in ["1D_CNN", "2D_CNN", "LSTM", "CNN+LSTM"]:
            t_train_0 = t.time()
            history=model.train(X_train, y_train, X_test, y_test,
                                n_batch, 
                                n_epochs,
                                learning_rate,
                                decay_rate,
                                save_dir_k)
            Performance["t_train"].append(t.time()-t_train_0)
            # Output accuracy of classifier
            print("Training Score: \t{:.5f}".format(history.history['acc'][-1]))
            print("Validation Score: \t{:.5f}".format(history.history['val_acc'][-1]))

            # Print Confusion Matrix
            t_classify_0 = t.time()
            ypred = model.classify(X_test)
            Performance["t_classify"].append(t.time()-t_classify_0)

            Performance["acc"].append(history.history['val_acc'][-1])
            
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            _,_, results = plot_confusion_matrix(directory=save_dir_k, y_true=y_test, y_pred=ypred.argmax(1), 
                                    classes=['benign', 'malware'], 
                                    normalize=False)

            Performance["tpr"].append(results["TPR"])
            Performance["far"].append(results["FAR"])

            for k,v in results.items():
                save_dict[k] = v
            # Plot loss and accuracy
            plotLoss(save_dir_k, history)

            # Save the trained model and its hyperparameters
            saveModel(save_dir_k, model.model, save_dict, history)

        else: # ML model
            t_train_0 = t.time()
            model.train(X_train, y_train)
            Performance["t_train"].append(t.time()-t_train_0)
            
            t_classify_0 = t.time()
            ypred = model.classify(X_test)
            Performance["t_classify"].append(t.time()-t_classify_0)
            Performance["acc"].append(model.model.score(X_test, y_test))
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            _,_, results = plot_confusion_matrix(directory=save_dir_k, y_true=y_test, y_pred=ypred, 
                                    classes=['benign', 'malware'], 
                                    normalize=False)

            Performance["tpr"].append(results["TPR"])
            Performance["far"].append(results["FAR"])

            # No loss plot

            # No model saving

            # Save some stuff
            with open(save_dir_k + '/'+ modelname+'.txt', 'w') as file:
                for k,v in sorted(save_dict.items()):
                    file.write("{} \t: {}\n".format(k,v))
                file.write("Train Accuracy \t: {:.5f} \n".format(model.model.score(X_train, y_train)))
                file.write("Validation Accuracy \t: {:.5f} \n".format(model.model.score(X_test, y_test)))

    Performance["t_train_mean"] = sum(Performance["t_train"])/len(Performance["t_train"])
    Performance["t_classify_mean"] = sum(Performance["t_classify"])/len(Performance["t_classify"])
    Performance["acc_mean"] = sum(Performance["acc"])/len(Performance["acc"])
    Performance["tpr_mean"] = sum(Performance["tpr"])/len(Performance["tpr"])
    Performance["far_mean"] = sum(Performance["far"])/len(Performance["far"])
    cpu_reads.append(p.cpu_percent(interval=None))
    Performance["cpu_mean"] = sum(cpu_reads)/len(cpu_reads[1:]) # Exclude first element coz it's 0
    Performance["cpu_max"] = max(cpu_reads)

    with open(save_dir+'/performace.txt', 'w') as fp:
        for k,v in sorted(Performance.items()):
            fp.write("{},{}\n".format(k,v))


def main():

    save_dict = {}
    save_dict['CNN_layers'] = CNN_layers
    save_dict['filters'] = filters
    save_dict['kernel_size'] = kernel_size
    save_dict['strides'] = strides
    save_dict['clf_reg'] = clf_reg
    save_dict['dropout_rate'] = dropout_rate
    save_dict['learning_rate'] = learning_rate
    save_dict['decay_rate'] = decay_rate
    save_dict['n_batch'] = n_batch
    save_dict['n_epochs'] = n_epochs
    save_dict['n_neighbors'] = n_neighbors
    save_dict['n_estimators'] = n_estimators
    save_dict['max_depth'] = max_depth
    save_dict['C'] = C
    save_dict['svm_kernel'] = svm_kernel
    save_dict['mlp_solver'] = mlp_solver
    save_dict['mlp_hidden_units'] = mlp_hidden_units
    
    mem_by_model = {}

    modelnames = ["1D_CNN", "2D_CNN", "LSTM", "CNN+LSTM"] # 1D_CNN 2D_CNN LSTM CNN+LSTM

    dataset = "CICIDS2017_encrypted_noFilter_T1800_N15_M1000_multilabels.csv" # full path to csv of NetML or CICIDS2017
    N = 10      # Max=15
    M = 100     # Max=1000
    # IPv6+UDP (no IPaddr) = 16 (MINIMUM)
    # IPv4+TCP (no IPaddr, both with full options field) = 112 (MAX of hdr len) 


    for modelname in modelnames:
        # Create folder for the results
        time_ = t.strftime("%Y%m%d-%H%M%S")
        save_dir = os.getcwd() + '/results/' + dataset.split('_')[0] + '/raw-bytes_experiments/' + modelname + '_' + time_
        os.makedirs(save_dir)
        with open(save_dir + "/memory_usage.txt", "w") as fp:
            mem_by_model[modelname] = memory_usage((profile, (dataset, modelname, save_dict, save_dir, N, M, 2)))
            fp.write("AverageMem,{}MB\n".format(sum(mem_by_model[modelname])/len(mem_by_model[modelname])))
            fp.write("MaxMem,{}MB\n".format(max(mem_by_model[modelname])))


if __name__ == '__main__':
    main()