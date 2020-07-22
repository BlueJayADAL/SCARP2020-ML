# SCARP2020-ML
The repository for SCARP 2020 on Network Traffic Analytics

## Walkthrough Guide
1. Currently there is one main runnable file (ModelRunner.py). A sample command can be found below with respective arguments for running.

`python ModelRunner.py --dataset NetML --model lr --load false --save true --runs 1`

- -d, --dataset [DATASET]: Specifies which dataset to use (supported: NetML, CICIDS)
- -m, --model [MODELTYPE]: Specifies which model to use (supported: lr, df, svm, ann)
- -l, --load [T/F]: Specifies whether or not to load from disk (supported: true and false)
- -s, --save [T/F]: Specifies whether or not to save a model if loading is false (supported: true and false)
- -r, --runs[Number]: Specifies how many times to run the model

The above command supports a variety of model types which are all listed below.

Standard Supported Models:
* Linear Regression (lr)
* Decision Forest (df)
* Support Vector Machine (svm)
* K Nearest Neighbors (knn)

DAAL Supported Models:
* Linear Regression (daallr)
* Decision Forest (daaldf)
* Support Vector Machine (daalsvm)
* K Nearest Neighbors (daalknn)

Standard AI Supported Models:
* Artificial Neural Network (ann)
* Recurrent Neural Network (rnn)
* Convolutional 1D Neural Network (cnn1d)
* Convolutional 2D Neural Network (cnn1d)

OpenVINO AI Supported Models:
* VINO Artificial Neural Network (vinoann)
* VINO Recurrent Neural Network (vinornn)
* VINO Convolutional 1D Neural Network (vinocnn1d)
* VINO Convolutional 2D Neural Network (vinocnn1d)

Once a model is ran through the command above, it will begin training and testing.
Furthermore, if a model is saved it may be loaded to output testing results again.

## Additional Resources
### daal4py Documentation Home
* #### https://intelpython.github.io/daal4py/index.html
### Intel DAAL Developer Guide
* #### https://software.intel.com/en-us/download/intel-data-analytics-acceleration-library-developer-guide
### OpenVINO Documentation Home
* #### https://docs.openvinotoolkit.org/latest/index.html
### OpenVINO Developer Guide
* #### https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
