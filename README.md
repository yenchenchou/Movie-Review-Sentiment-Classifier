# Movie Review Sentiment Classifier
## Introduction
This project demonstrates a interactive web app for users to paste movie related reviews and the web app will classify whether the review is a positive or negative one by a deployed recurrent neural network. The sentiment classifier is deployed through AWS SageMaker using LSTM in Pytorch. 

## Prerequisites
Here are setups you need to run the code. Please see the [README](https://github.com/udacity/sagemaker-deployment/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook and downloading the project files (as well as the other notebooks).
1. Create notebook instance on AWS SageMaker and choose 'ml.p2.xlarge' as training instance. Remember to make sure your have 'ml.p2.xlarge' instance available since you may have 0 limit of this type of instance.
2. Upload the file and make sure you have the following packages. All packages below should be built-in the SageMaker instance
```Python
import os
import pickle
import glob
import re

import boto3
import nltk
import numpy as np
import pandas as pd
import sagemaker
import torch
import torch.utils.data
import torch.optim as optim
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchModel
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# the below resources are from the script in the project folder
from train.model import LSTMClassifier
```
## Running the project
Once you upload the files in the instance, run 'SageMaker Project.ipynb' and everything shall go well. But this step only enable you to see the performance of the classifier. You need to do some setups on AWS. See [Deployment](##Deployment)

## Deployment
The deployment detailed information is in the 'SageMaker Project.ipynb' notebook. Here are the major tasks for completing the deployment. Once finishing building and saving the model:
1. Create an IAM Role for the Lambda function
2. Setting up a Lambda function
3. Setting up API Gateway

## Author
YenChenChou - [LinkedIn](https://www.linkedin.com/in/yc-chou/)

