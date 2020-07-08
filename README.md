# SCARP2020-ML
The repository for SCARP 2020 on Network Traffic Analytics

## Walkthrough Guide
1. Currently there is one main runnable file (daalModels.py). A sample command can be found below with respective arguments for running.
    
`python daalModels.py --dataset NetML --model lr --load False --runs 1`

- -d, --dataset [DATASET]: Specifies which dataset to use (supported: NetML, CICIDS)
- -m, --model [MODELTYPE]: Specifies which model to use (supported: lr, df, svm, ann)
- -l, --load [T or F]: Specifies whether or not to load from disk (supported: true and false)
- -r, --runs[Number]: Specifies how many times to run the model

DAAL Supported Models:
* Linear Regression (lr)
* Decision Forest (df)
* Support Vector Machine (svm)

Once a model is ran through the command above, it will begin training and testing. This will be changed shortly to allow for specifying either train or test.

## Additional Resources
### daal4py Documentation Home
#### - https://intelpython.github.io/daal4py/index.html
### Intel DAAL Developer Guide
#### - https://software.intel.com/en-us/download/intel-data-analytics-acceleration-library-developer-guide
