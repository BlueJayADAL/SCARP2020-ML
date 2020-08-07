# SCARP2020-ML
The repository for SCARP 2020 on Network Traffic Analytics.

The project is codenamed **ANTA** for Accelerated Network Traffic Analytics.

This is a joint project worked on by the following individuals from the Universities specified below:
* Elizabethtown College (Elizabethtown, PA)
	* Dr. Li - Advisor of Project
	* Matthew Grohotolski - Programmer
	* Connor DiLeo - Programmer
* University of Massachusetts Lowell (Lowell, MA)
	* Onur Barut - Programmer

## Walkthrough Guide
1. Currently there is one main runnable file (nsyss_benchmark.py). A sample command can be found below with respective arguments for running.

`python nsyss_benchmark.py --dataset NetML --model LR daal_LR --kfolds 10`

- -d, --dataset [DATASET]: Specifies which dataset to use (supported: NetML, CICIDS)
- -m, --model [MODELTYPE]: Specifies which model to use (supported: LR, daal_LR, kNN, daal_kNN, RF, daal_DF, SVM ,daal_SVM, MLP, 1D_CNN, vino_1D_CNN, 2D_CNN, vino_2D_CNN, LSTM, vino_LSTM, CNN+LSTM, vino_CNN+LSTM)
- -k, --kfolds[Number]: Specifies how many times to run each model

The above command supports a variety of model types which are all listed below.

Standard Supported Models:
* Linear Regression (LR)
* Random Forest (RF)
* Support Vector Machine (svm)
* K Nearest Neighbors (kNN)

DAAL Supported Models:
* Linear Regression (daal_LR)
* Decision Forest (daal_DF)
* Support Vector Machine (daal_SVM)
* K Nearest Neighbors (daal_kNN)

Standard AI Supported Models:
* Artificial Neural Network (ANN)
* Long Short Term Memory Neural Network (LSTM)
* Convolutional + Long Short Term Memory Neural Network (CNN+LSTM)
* Convolutional 1D Neural Network (1D_CNN)
* Convolutional 2D Neural Network (2D_CNN)

OpenVINO AI Supported Models:
* VINO Artificial Neural Network (vino_ANN)
* VINO Long Short Term Memory Neural Network (vino_LSTM)
* VINO Convolutional + Long Short Term Memory Neural Network (vino_CNN+LSTM)
* VINO Convolutional 1D Neural Network (vino\_1D_CNN)
* VINO Convolutional 2D Neural Network (vino\_2D_CNN)

Once a model is ran through the command above, it will begin training and testing.
The profiler also keeps tabs on how much memory is utilized along with training and testing times.

Finally, once a model is finished it is saved in the results folder in either NetML or CICIDS2017 folder and is categorized by the starttime when the model began running.
A list of final performance results can be found in the final model folder with average performance statistics.

Some benchmark commands are also provided in order to test various aspects of the program. See more example commands below.

With NetML dataset:

* Run all ML / DAAL models: 
`python nsyss_benchmark.py --dataset NetML --model LR daal_LR RF daal_DF SVM daal_SVM kNN daal_kNN --kfolds 10`

* Run all DL / VINO models: 
`python nsyss_benchmark.py --dataset NetML --model ANN vino_ANN LSTM vino_LSTM CNN+LSTM vino_CNN+LSTM 1D_CNN vino_1D_CNN 2D_CNN vino_2D_CNN --kfolds 10`

With CICIDS2017 dataset:

* Run all ML / DAAL models: 
`python nsyss_benchmark.py --dataset CICIDS2017 --model LR daal_LR RF daal_DF SVM daal_SVM kNN daal_kNN --kfolds 10`

* Run all DL / VINO models: 
`python nsyss_benchmark.py --dataset CICIDS2017 --model ANN vino_ANN LSTM vino_LSTM CNN+LSTM vino_CNN+LSTM 1D_CNN vino_1D_CNN 2D_CNN vino_2D_CNN --kfolds 10`

## Additional Resources
### daal4py Documentation Home
* #### https://intelpython.github.io/daal4py/index.html
### Intel DAAL Developer Guide
* #### https://software.intel.com/en-us/download/intel-data-analytics-acceleration-library-developer-guide
### OpenVINO Documentation Home
* #### https://docs.openvinotoolkit.org/latest/index.html
### OpenVINO Developer Guide
* #### https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
