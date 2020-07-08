import os
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from memory_profiler import memory_usage
import psutil

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from utils.helper2 import plot_confusion_matrix, plotLoss, saveModel
from utils.GPU_models import CNN_1D, CNN_2D, LSTM, CNN_LSTM

def matrix_to3D(X_train, X_test):
	dim1 = X_train.shape[1]
	divs = [i for i in range(1,dim1+1) if (dim1%i == 0)]
	if len(divs) == 2: # i.e. prime number
		# Add zeros column
		X_train = np.concatenate((X_train, np.zeros((X_train.shape[0],1))), axis=1)
		X_test = np.concatenate((X_test, np.zeros((X_test.shape[0],1))), axis=1)
		dim1 = X_train.shape[1]
		divs = [i for i in range(1,dim1+1) if (dim1%i == 0)]		
	mid_idx = len(divs)//2

	return X_train.reshape(-1, divs[mid_idx], int(dim1/divs[mid_idx]), 1), X_test.reshape(-1, divs[mid_idx], int(dim1/divs[mid_idx]), 1)

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

#@profile
def profile(dataset, modelname, save_dict, save_dir, num_folds=10):
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
	df = pd.read_csv(dataset+"_enc_filtered_top50.csv")

	# Standardize the data
	y = df.pop('label').values
	scaler = preprocessing.StandardScaler()
	X = scaler.fit_transform(df.values)

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

		# Init the model
		if modelname == "1D_CNN":
			model = CNN_1D(input_shape=(X_train.shape[1],1,), 
					n_classes=2,                  
					filters=128, 
                  	kernel_size=3,
                  	strides=1,
                  	dense_units=128,
                  	dropout_rate=0.5, 
                  	CNN_layers=2, 
                  	clf_reg=1e-4)

		elif modelname == "2D_CNN":
			X_train, X_test = matrix_to3D(X_train, X_test)
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
			X_train, X_test = matrix_to3D(X_train, X_test)
			X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
			X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

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
			model = CNN_LSTM(input_shape=(X_train.shape[1],1,), # Model of "Deep in the Dark paper"
					n_classes=2,                  
                  	dropout_rate=0.5, 
                  	lstm_reg=1e-4,
                  	clf_reg=1e-4)
		else:
			return

		# Train the model
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
	
	mem_by_model = {}

	modelnames = ["CNN+LSTM"] # 1D_CNN 2D_CNN LSTM CNN+LSTM

	dataset = "NetML" # NetML or CICIDS2017

	

	for modelname in modelnames:
		# Create folder for the results
		time_ = t.strftime("%Y%m%d-%H%M%S")
		save_dir = os.getcwd() + '/results/' + modelname + '_' + time_
		os.makedirs(save_dir)
		with open(save_dir + "/memory_usage.txt", "w") as fp:
			mem_by_model[modelname] = memory_usage((profile, (dataset, modelname, save_dict, save_dir, 2)))
			fp.write("AverageMem,{}MB\n".format(sum(mem_by_model[modelname])/len(mem_by_model[modelname])))
			fp.write("MaxMem,{}MB\n".format(max(mem_by_model[modelname])))


if __name__ == '__main__':
	main()