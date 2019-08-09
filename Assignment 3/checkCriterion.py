# ====================================================================================================================
# 												checkCriterion.py
# ====================================================================================================================

## AUTHOR: Vamsi Krishna Reddy Satti

# This script computes the average loss for given input and target and prints in console.
# Additionally, this script saves ‘gradInput.bin’ which contains gradient of loss w.r.t input.

# Usage example:
# python checkCriterion.py -i "../cs763-assign3/CS763DeepLearningHW/input_criterion_sample_1.bin" -t "../cs763-assign3/CS763DeepLearningHW/target_sample_1.bin" -ig "./temp/gradCriterionInput_sample_1.bin"

import argparse
import os
import torch
import torchfile
from src import criterion


torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU support if CUDA supported hardware is present


# Parse the arguments
parser = argparse.ArgumentParser(description="This script checks the Criterion")
parser.add_argument("-i", dest="input_path", required=True, type=str, help="[IN] path to input.bin")
parser.add_argument("-t", dest="target_path", required=True, type=str, help="[IN] path to target.bin")
parser.add_argument("-ig", dest="grad_input_path", required=True, type=str, help="[OUT] path to gradInput.bin")
args = parser.parse_args()


# Read the files to get input, target tensors
input = torch.tensor(torchfile.load(args.input_path, force_8bytes_long=True), dtype=torch.float64).to(device)
target = torch.tensor(torchfile.load(args.target_path, force_8bytes_long=True), dtype=torch.float64).reshape(-1).long().to(device) - 1 # Note the -1 here


# Get to work!
loss_fn = criterion.CrossEntropyLoss()
loss = loss_fn(input, target).item()
grad_input = loss_fn.backward(input, target)

print(loss)
grad_input = grad_input.cpu().numpy()
os.makedirs(os.path.dirname(args.grad_input_path), exist_ok=True)
with open(args.grad_input_path, 'w+b') as file:
	torch.save(grad_input, file)

