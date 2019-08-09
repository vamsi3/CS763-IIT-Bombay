# ====================================================================================================================
# 												testModel.py
# ====================================================================================================================

## AUTHOR: Vamsi Krishna Reddy Satti

# This script loads the model saved in ‘(model name)’ folder and runs it on the test data.
# Additionally, this script saves the predictions as 1D tensor named ‘testPrediction.bin’ in present working directory. 

# Usage example:
# python testModel.py -modelName bestModel -data ../cs763-assign3/test.bin

import argparse
import pickle
import os
import torch
import torchfile
from src import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU support if CUDA supported hardware is present


# Parse the arguments
parser = argparse.ArgumentParser(description="This script predicts on test data given a trained model")
parser.add_argument("-modelName", dest="model_name", required=True, type=str, help="name of the model")
parser.add_argument("-data", dest="test_data_path", required=True, type=str, help="location of the test data")
args = parser.parse_args()


# Load the model and preprocessing parameters
model_file_path = f"./{args.model_name}/model.bin"
with open(model_file_path, 'rb') as f:
	state = pickle.load(f)
config, net, preprocess = state['config'], state['model'], state['preprocess']

for key in preprocess:
	preprocess[key] = preprocess[key].to(device)

net.to(device)
net.eval() # set model to eval mode


# Read the test data and get the predictions from model
X_test = torch.tensor(torchfile.load(args.test_data_path, force_8bytes_long=True), dtype=torch.float).reshape(-1, 108*108).to(device)

# Preprocess the data
X_test -= preprocess['mean']
X_test /= preprocess['std']
X_test = X_test.reshape(-1, 1, 108, 108)


i=0
while i*config['batch_size'] < X_test.shape[0]:
	out = net(X_test[i*config['batch_size']: (i+1)*config['batch_size']])
	test_prediction = out if i == 0 else torch.cat((test_prediction, out), dim=0)
	i += 1

# Adjust datatypes and save the predictions on test data
test_prediction = test_prediction.cpu().numpy()
with open("./testPrediction.bin", 'w+b') as file:
	torch.save(test_prediction, file)

