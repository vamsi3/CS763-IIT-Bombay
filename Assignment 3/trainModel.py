# ====================================================================================================================
# 												trainModel.py
# ====================================================================================================================

## AUTHOR: Vamsi Krishna Reddy Satti

# This script trains a model having the performance of the bestModel on the data provided.
# The training time does not exceed 3 hours.

# Usage example:
# python trainModel.py -modelName bestModel -data ../cs763-assign3/data.bin -target ../cs763-assign3/labels.bin

import argparse
import pickle
import os
import torch
import torchfile
from src import criterion, layers, model, optim, utils


# Model training configuration of hyperparameters
config = {
	'lr':					6e-5,
	'epochs':				200,
	'batch_size':			256,
}



torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU support if CUDA supported hardware is present


# Parse the arguments
parser = argparse.ArgumentParser(description="This script trains my best model")
parser.add_argument("-modelName", dest="model_name", required=True, type=str, help="name of the model")
parser.add_argument("-data", dest="train_data_path", required=True, type=str, help="location of the training data")
parser.add_argument("-target", dest="train_labels_path", required=True, type=str, help="location of the training labels")
args = parser.parse_args()


# I'm using Windows, so since default of long in Windows is 4 bytes, need to force long as 8 bytes.
X_train = torch.tensor(torchfile.load(args.train_data_path, force_8bytes_long=True), dtype=torch.float64).reshape(-1, 108*108).to(device)
y_train = torch.tensor(torchfile.load(args.train_labels_path, force_8bytes_long=True), dtype=torch.float64).reshape(-1).long().to(device)


# Model definition
net = model.Model()
net.add_layer(layers.Conv2d((108, 108), 1, 16, kernel_size=18, stride=2))
net.add_layer(layers.ReLU())
net.add_layer(layers.MaxPool2d(2))
net.add_layer(layers.BatchNorm2d(16))
net.add_layer(layers.Conv2d((23, 23), 16, 32, kernel_size=5, stride=2))
net.add_layer(layers.ReLU())
net.add_layer(layers.MaxPool2d(2))
net.add_layer(layers.BatchNorm2d(32))
net.add_layer(layers.Flatten())
net.add_layer(layers.Linear(5 * 5 * 32, 256))
net.add_layer(layers.ReLU())
net.add_layer(layers.BatchNorm1d(256))
net.add_layer(layers.Linear(256, 128))
net.add_layer(layers.ReLU())
net.add_layer(layers.BatchNorm1d(64))
net.add_layer(layers.Linear(128, 64))
net.add_layer(layers.ReLU())
net.add_layer(layers.BatchNorm1d(64))
net.add_layer(layers.Linear(64, 6))
net.to(device)
net.train() # set model to train mode


# Preprocess the data
preprocess = {}
preprocess['mean'] = X_train.mean(0, keepdim=True)
preprocess['std'] = X_train.std(0, keepdim=True)
X_train -= preprocess['mean']
X_train /= preprocess['std']
X_train = X_train.reshape(-1, 1, 108, 108)


# Initialize the DataLoader
dataloader_train = utils.DataLoader((X_train, y_train), batch_size=config['batch_size'], shuffle=True)


# Loss function used is Cross Entropy Loss
# Optimizer used is Stochastic Gradient Descent with Nesterov momentum
loss_fn = criterion.CrossEntropyLoss()
optimizer = optim.Adam(net, config['lr'])
scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.96)


# Training of the model happens here...
for epoch in range(config['epochs']):
	print(f"[Epoch] {epoch} starts...")
	scheduler.step()
	for X, y in dataloader_train:
		output = net(X)
		loss = loss_fn(output, y)
		net.zero_grad()
		grad = loss_fn.backward(output, y)
		net.backward(output, grad)
		optimizer.step()


# Save the model to a file
for key in preprocess:
	preprocess[key] = preprocess[key].cpu()
state = {
	'config': config,
	'model': net.cpu(),
	'preprocess': preprocess
}

model_file_path = f"./{args.model_name}/model.bin"
os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
with open(model_file_path, 'w+b') as file:
	pickle.dump(state, file)
