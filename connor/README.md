## INTRO
The following is the repository for SCARP 2020 project. 

## Walkthrough Guide
It is recommended to do the use a virtual envirement.

1. Read and download the necessary files from the repository which can be found [here](https://github.com/ACANETS/NetML-Competition2020)
2. Once the repository is downloaded please put Main.py in the main folder.
3. Once in virtual envirement you can access and run Main.py files. EXAMPLE: python Main.py --dataset ./data/NetML --anno top --submit test-std --model RF --runs 2

*There are three annotations you can choose from, top, mid and fine
*The three models available are RF, SVM, MLP, and LR 
*--runs is the amount of times you want the program to run the model
*The "./data/NetML" section is the location of where your dataset is located

*This command will create a results folder. In this results folder there will be a picture of a confusion matrix with F1, AP scores and submission_test.json file
