import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from RNN_torch.model import RNN

# Hyper parameters
BATCH_SIZE = 64
EPOCH = 1
TIME_STEP = 28  # 考虑多少个时间点的数据
INPUT_SIZE = 1  # 每个时间点给RNN多少个数据点
LR = 0.01

rnn = RNN(INPUT_SIZE)
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(50):
    start, end = step * np.pi, (step + 1) * np.pi
    # use sin pre cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape(batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data  # !!! this step is important

    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # clear gradient for next train
    loss.backward()  # back propagation, compute gradient
    optimizer.step()

    # plot
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.5)

plt.ioff()
plt.show()
