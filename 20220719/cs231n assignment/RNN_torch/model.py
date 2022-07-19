import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn



class RNN(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x(batch, time_step, input_size)
        # h_state(n_layers, batch, hidden_size)
        # r_out(batch, time_step, output_size = hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):  # size是tensor的形状是一个数组，size(1)就是里面的第二个值域，
            # 就是time_step的值的个数 即第二个维度的大小
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
