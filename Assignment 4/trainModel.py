# ====================================================================================================================
# 												trainModel.py
# ====================================================================================================================

## AUTHOR: Vamsi Krishna Reddy Satti

# This script trains a model having the performance of the bestModel on the data provided.
# The training time does not exceed 3 hours.

# Usage example:
# python trainModel.py -modelName bestModel -data ../assign42019/train_data.txt -target ../assign42019/train_labels.txt

import argparse
import os
import torch
from src.utils.data import DataLoader
from src import nn, optim

# Model training configuration of hyperparameters
config = {
	'lr':					0.01,
	'epochs':			    30,
	'batch_size':			64,
}

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU support if CUDA supported hardware is present

# Parse the arguments
parser = argparse.ArgumentParser(description="This script trains my best model")
parser.add_argument("-modelName", dest="model_name", required=True, type=str, help="name of the model")
parser.add_argument("-data", dest="train_data_path", required=True, type=str, help="location of the training data")
parser.add_argument("-target", dest="train_labels_path", required=True, type=str, help="location of the training labels")
args = parser.parse_args()


with open(args.train_data_path, 'r') as f:
    data = list(map(lambda line: list(map(int, line.split())), f.readlines()))
num_seq = len(data)
seq_len = torch.tensor(list(map(len, data)), dtype=torch.long, device=device)
max_seq_len = seq_len.max().item()
padded_data = list(map(lambda line: line + [0] * (max_seq_len - len(line)), data))
train_seq = torch.tensor(padded_data, dtype=torch.long, device=device)
max_feature = train_seq.max()

with open(args.train_labels_path, 'r') as f:
    y = torch.tensor(list(map(lambda line: int(line), f.readlines())), dtype=torch.float, device=device)

# Model definition
rnn = nn.RNN(input_size=max_feature, hidden_size=1, batch_first=True)
rnn.to(device)
rnn.train() # set model to train mode


# Initialize the DataLoader
train_dataloader = DataLoader((train_seq, seq_len, y), batch_size=config['batch_size'], shuffle=True)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(rnn.parameters(), lr=config['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


# Training of the model happens here...
for epoch in range(config['epochs']):
    scheduler.step()
    acc = 0.0
    for x, l, y in train_dataloader:
        x = torch.zeros((x.size(0), max_seq_len, max_feature), dtype=torch.float, device=device).scatter_(2, x.unsqueeze(2), 1)
        rnn.zero_grad()
        output, ht = rnn(x)
        
        pred = output[torch.arange(l.numel(), dtype=torch.long), l-1, 0]
        
        loss = loss_fn(pred, y)
        
        pred2 = pred.clone()
        pred2[pred2 >= 0] = 1
        pred2[pred2 < 0] = 0
        acc += (pred2 == y).sum()
        
        grad = loss_fn.backward(pred, y)
        
        grad_expanded = torch.zeros_like(output)
        grad_expanded[torch.arange(l.numel(), dtype=torch.long), l-1, 0] = grad.reshape(-1)
        
        rnn.backward(grad_expanded)
        optimizer.step()
    print(f"Epoch: {epoch}   |   Train Accuracy: {acc.item() * 100 / num_seq}")

# Save the model to a file
model_file_path = f'./{args.model_name}/model.bin'
os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
torch.save(rnn.state_dict(), model_file_path)
