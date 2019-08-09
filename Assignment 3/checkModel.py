# ====================================================================================================================
# 												checkModel.py
# ====================================================================================================================

## AUTHOR: Vamsi Krishna Reddy Satti

# Too tired of writing this script. Way too inconsistent design on this script I felt :(.
# No mood to write description of this script.

# Usage example:
# python checkModel.py -config ../cs763-assign3/CS763DeepLearningHW/modelConfig_1.txt -i ../cs763-assign3/CS763DeepLearningHW/input_sample_1.bin -og ../cs763-assign3/CS763DeepLearningHW/gradOutput_sample_1.bin -o ./temp/output.bin -ow ./temp/gradW.bin -ob ./temp/gradB.bin -ig ./temp/gradInput.bin

import argparse
import os
import torch
import torchfile
from src import layers, model


torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU support if CUDA supported hardware is present


# Parse the arguments
parser = argparse.ArgumentParser(description="This script checks the Model")
parser.add_argument("-config", dest="model_config_path", required=True, type=str, help="[IN] path to modelConﬁg.txt")
parser.add_argument("-i", dest="input_path", required=True, type=str, help="[IN] path to input.bin")
parser.add_argument("-og", dest="grad_output_path", required=True, type=str, help="[IN] path to gradOutput.bin")
parser.add_argument("-o", dest="output_path", required=True, type=str, help="[OUT] path to output.bin")
parser.add_argument("-ow", dest="grad_weights_path", required=True, type=str, help="[OUT] path to gradWeight.bin")
parser.add_argument("-ob", dest="grad_bias_path", required=True, type=str, help="[OUT] path to gradB.bin")
parser.add_argument("-ig", dest="grad_input_path", required=True, type=str, help="[OUT] path to gradInput.bin")
args = parser.parse_args()


# Define the model
net = model.Model()
with open(args.model_config_path, 'r') as file:
	num_linear_layers = int(file.readline())
	while num_linear_layers > 0:
		tokens = file.readline().strip().split()
		if (tokens[0] == 'linear'):
			input_size, output_size = int(tokens[1]), int(tokens[2])
			net.add_layer(layers.Linear(input_size, output_size))
			num_linear_layers -= 1
		elif (tokens[0] == 'relu'):
			net.add_layer(layers.ReLU())
	weights_path = file.readline()[:-1]
	bias_path = file.readline()[:-1]


# Read the weights, bias of all linear layers and assign to model
weights = torchfile.load(weights_path, force_8bytes_long=True)
bias = torchfile.load(bias_path, force_8bytes_long=True)
num_linear_layers = 0
for layer in net.layers:
	if type(layer).__name__ == 'Linear':
		layer.params['weights'] = torch.tensor(weights[num_linear_layers], dtype=torch.float64).to(device).t()
		layer.params['bias'] = torch.tensor(bias[num_linear_layers], dtype=torch.float64).to(device).reshape(1, -1)
		num_linear_layers += 1


# Read input, grad_output and do a forward and backward pass on the model
input = torch.tensor(torchfile.load(args.input_path, force_8bytes_long=True), dtype=torch.float64).to(device)
input = input.reshape(input.shape[0], -1)
grad_output = torch.tensor(torchfile.load(args.grad_output_path, force_8bytes_long=True), dtype=torch.float64).to(device)

net.to(device)
net.train()
output = net(input)
net.zero_grad()
grad_input = net.backward(output, grad_output)


# Create grad variables as needed for saving into .bin files
grad_bias, grad_weights = [], []
for layer in net.layers:
	if type(layer).__name__ == 'Linear':
		grad_weights.append(layer.grad['weights'].t().cpu().numpy())
		grad_bias.append(layer.grad['bias'].reshape(-1).cpu().numpy())


# Adjust datatypes and save the required variables
output = output.cpu().numpy()
grad_input = grad_input.cpu().numpy()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, 'w+b') as file:
	torch.save(output, file)

os.makedirs(os.path.dirname(args.grad_weights_path), exist_ok=True)
with open(args.grad_weights_path, 'w+b') as file:
	torch.save(grad_weights, file)

os.makedirs(os.path.dirname(args.grad_bias_path), exist_ok=True)
with open(args.grad_bias_path, 'w+b') as file:
	torch.save(grad_bias, file)

os.makedirs(os.path.dirname(args.grad_input_path), exist_ok=True)
with open(args.grad_input_path, 'w+b') as file:
	torch.save(grad_input, file)

