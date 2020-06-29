## INTRO
The following is the repository for SCARP 2020 project.  Each Model file has their own main method, Therefore need to  need to be ran separately.

## Walkthrough Guide
It is recommended to do the use a virtual envirement.

1. Read and download the necessary files from the repository which can be found [here](https://evalai.cloudcv.org/web/challenges/challenge-page/526/overview)
2. In this repository there is a model folder. In this folder are all the programs needed in order to run the datasets. Put these .py files are in the netml-competition repository that you downloaded.
3. Once in virtual envirement you can access and runthese files. EXAMPLE: python RF_baseline.py --dataset ./data/NetML --anno top --submit test-std 

There are three annotations you can choose from, top, mid and fine
The "./data/NetML" section is the location of where your dataset is located

This command will create a results folder. In this results folder there will be a picture of a confusion matrix with F1, AP scores and submission_test.json file
