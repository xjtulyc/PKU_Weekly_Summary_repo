import torch
import torch.optim as optim
from torch import nn
from tqdm import trange

from dataset.cifar_dataset import trainloader, testloader
from nn2layer.model import Net

# load the pretrained model to the cnn of ground pics(above)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp = Net().double().to(device)

Epoch = 100

loss_fn = nn.MSELoss()
opt = optim.Adam(mlp.parameters(), lr=1e-5)

# Train
for epoch in trange(Epoch):
    best_ac = 100
    for step, (image, label) in enumerate(trainloader):
        image = image.double().to(device)
        label = label.double().to(device)
        predict = mlp(image[0].view(-1))
        loss = loss_fn(predict, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Epoch: {}; Step: {}; Loss: {}'.format(epoch, step, loss.item()))

# Test
acc = 0
step = None
for step, (image, label) in enumerate(testloader):
    image = image.double().to(device)
    label = label.double().to(device)
    predict = mlp(image[0].view(-1))
    if predict == label:
        acc += 1
    print('Label: {}; Predict: {}'.format(label, predict))
acc = acc / step
print('ACC: {}'.format(acc))
