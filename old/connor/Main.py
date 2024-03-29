import os
import json
import argparse
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing, linear_model, svm, neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from utils.helper import *

def submit(clf, test_set, scaler, class_label_pair, filepath):
    Xtest, ids = get_submission_data(test_set)
    X_test_scaled = scaler.transform(Xtest)
    print("Predicting on {} ...".format(test_set.split('/')[-1]))
    predictions = clf.predict(X_test_scaled)
    make_submission(predictions, ids, class_label_pair, filepath)   

def main ():
    parser = argparse.ArgumentParser(description="NetML Challenge 2020 Random Forest Classifier Baseline", add_help=True)
    parser.add_argument('-d', '--dataset', action="store", help="Dataset path")
    parser.add_argument('-a', '--anno', action="store", help="Annoation level: {top, mid, fine}")
    parser.add_argument('-s', '--submit', action="store", help="{test-std, test-challenge, both} Select which set to submit")
    parser.add_argument('-m', '--model', action="store",help="RF LR SVM or MLP")
    parser.add_argument('-r', '--runs', action="store", help="how many runs?")

    args = parser.parse_args()

    if args.dataset == None or not os.path.isdir(args.dataset) or args.anno == None:
        print ("No valid dataset set or annotations found!")
        return
    elif args.submit is not None and args.submit not in ["test-std", "test-challenge", "both"]:
        print("Please select which set to submit: {test-std, test-challenge, both}")
        return
    elif args.anno not in ["top", "mid", "fine"]:
        print("Please select one of these for annotations: {top, mid, fine}. e.g. --anno top")
        return
    elif args.anno == "mid" and (args.dataset.find("NetML") > 0 or args.dataset.find("CICIDS2017") > 0):
        print("NetML and CICIDS2017 datasets cannot be trained with mid-level annotations. Please use either top or fine.")
        return
    elif args.model not in["RF", "SVM", "MLP", "LR"]:
        print("Please make sure your model is one of these: { RF, LR, SVM, MLP} and capital e.g. --model RF")
        return
    elif args.runs == None:
        print("Please enter how many times you want to test this program")
        return
    else:
        training_set = args.dataset+"/2_training_set"
        training_anno_file = args.dataset+"/2_training_annotations/2_training_anno_"+args.anno+".json.gz"
        test_set = args.dataset+"/1_test-std_set"
        challenge_set = args.dataset+"/0_test-challenge_set"
        model_type = args.model
        run_num = args.runs
        run_num = int(run_num)

    if run_num <=0:
        print("Please make your run number above 0!")

    for x in range(run_num):
        run = x+1
        print("This is run number: " ,(x+1))
        # Create folder for the results
        time_ = t.strftime("%Y%m%d-%H%M%S")

        save_dir = os.getcwd() + '/results/' + model_type+ time_
        os.makedirs(save_dir)

        # Get training data in np.array format
        Xtrain, ytrain, class_label_pair, Xtrain_ids = get_training_data(training_set, training_anno_file)

        # Split validation set from training data
        X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=ytrain)

        # Get name of each class to display in confusion matrix
        class_names = list(sorted(class_label_pair.keys()))

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)


        # Train RF Model
        train_timeS = t.time()
        if model_type == 'RF':
            print("Training the model ...")
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, max_features="auto")
            clf.fit(X_train_scaled, y_train)
        elif model_type == 'SVM':
            print("Training the model ...")
            clf = svm.SVC(C=1e+6, max_iter=1e+7, kernel='rbf', degree=2, gamma='scale', tol=1e-3, cache_size=2000,
                          verbose=True)
            clf.fit(X_train_scaled, y_train)
        elif model_type == 'MLP':
            print("Training the model ...")
            clf = neural_network.MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(23, 13, 11, 7, 5),
                                               batch_size=1000, alpha=1e-5, max_iter=300, tol=1e-4, n_iter_no_change=1000,
                                               verbose=True)
            clf.fit(X_train_scaled, y_train)
        elif model_type == 'LR':
            clf = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', max_iter=1000000, verbose=1)
            clf.fit(X_train_scaled, y_train)

        # Output accuracy of classifier
        print("Training Score: \t{:.5f}".format(clf.score(X_train_scaled, y_train)))
        print("Validation Score: \t{:.5f}".format(clf.score(X_val_scaled, y_val)))

        train_timeF =t.time()
        print("Total training time took " + str(round(train_timeF - train_timeS, 3)) + " seconds!")
        matrix_timeS = t.time()
        #Print Confusion Matrix
        ypred = clf.predict(X_val_scaled)
        np.set_printoptions(precision=2)
        # Plot normalized confusion matrix
        plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred,
                                classes=class_names,
                                normalize=False)
        matrix_timeF = t.time()
        print("Plotting normalized confusion matrix took " + str(round(matrix_timeF - matrix_timeS, 3)) + " seconds!")
       #Work in progress below
       #f = open("myfile.txt", "w")
        #f.write(str(plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred,
                 #               classes=class_names,
                       #         normalize=False)))
       #f.close()

            # Make submission with JSON format
        if args.submit == "test-std" or args.submit == "both":
            submit(clf, test_set, scaler, class_label_pair, save_dir+"/submission_test-std.json")
        if args.submit == "test-challenge" or args.submit == "both":
            submit(clf, challenge_set, scaler, class_label_pair, save_dir+"/submission_test-challenge.json")

if __name__ == "__main__":
    main()
