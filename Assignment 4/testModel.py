# ====================================================================================================================
# 												testModel.py
# ====================================================================================================================

## AUTHOR: Vamsi Krishna Reddy Satti

# This script loads the model saved in ‘(model name)’ folder and runs it on the test data.
# Additionally, this script saves the predictions as 1D tensor named ‘testPrediction.bin’ in present working directory. 

# Usage example:
# python testModel.py -modelName bestModel -data ../assign42019/test.bin

import argparse
import torch
from src import nn


torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU support if CUDA supported hardware is present


# Parse the arguments
parser = argparse.ArgumentParser(description="This script predicts on test data given a trained model")
parser.add_argument("-modelName", dest="model_name", required=True, type=str, help="name of the model")
parser.add_argument("-data", dest="test_data_path", required=True, type=str, help="location of the test data")
args = parser.parse_args()

# Read the test data
with open(args.test_data_path, 'r') as f:
    data = list(map(lambda line: list(map(int, line.split())), f.readlines()))
test_num_seq = len(data)
seq_len = torch.tensor(list(map(len, data)), dtype=torch.long, device=device)
test_max_seq_len = seq_len.max().item()
padded_data = list(map(lambda line: line + [0] * (test_max_seq_len - len(line)), data))
test_seq = torch.tensor(padded_data, dtype=torch.long, device=device)
test_max_feature = test_seq.max() # == 340
onehot_test_seq = torch.zeros((test_num_seq, test_max_seq_len, test_max_feature), dtype=torch.float, device=device).scatter_(2, test_seq.unsqueeze(2), 1)

# Load the model
rnn = nn.RNN(input_size=340, hidden_size=1, batch_first=True)
rnn.load_state_dict(torch.load(f"./{args.model_name}/model.bin"))
rnn.to(device)
rnn.eval()

# Get the predictions from model
rnn.eval()
output, ht = rnn(onehot_test_seq)
pred = output[torch.arange(seq_len.numel(), dtype=torch.long), seq_len-1, 0]
pred[pred >= 0] = 1
pred[pred < 0] = 0
pred = pred.cpu().long().reshape(-1)
torch.save(pred, 'testPrediction.bin')
