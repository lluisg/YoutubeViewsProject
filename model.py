import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hid_dim_lstm, hid_dim_fc, output_dim, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hid_dim_lstm, batch_first = True)
        self.fc1 = nn.Linear(hid_dim_lstm, hid_dim_fc)
        self.fc2 = nn.Linear(hid_dim_fc, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src):

        self.lstm.flatten_parameters()
        x, (hidden, cell) = self.lstm(src)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.squeeze()[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
