import os

import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

import model.feature_extractor as model
import utils.ShiftCrop as ShiftCrop
from dataset.dataset import innoDataset

polar_model = model.vgg16net()
ground_model = model.vgg16net()
if os.path.exists('inno/vgg16-397923af.pth'):
    pretrained_dict = torch.load('inno/vgg16-397923af.pth')

    model_dict = polar_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    polar_model.load_state_dict(model_dict)
    # load the pretrained model to the cnn of polar pics(above)
    model_dict = ground_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    ground_model.load_state_dict(model_dict)

# load the pretrained model to the cnn of ground pics(above)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ground_model = ground_model.double()
polar_model = polar_model.double()
ground_model = ground_model.to(device)
polar_model = polar_model.to(device)

# inno_data = innoDataset()
# polardata = innoDataset(is_ground=False)
# grounddata = innoDataset(is_ground=True)
data = innoDataset()
batch_size = 10
Epoch = 200
# polar_dataloader = DataLoader(polardata, batch_size=batch_size, shuffle=False)
# ground_dataloader = DataLoader(grounddata, batch_size=batch_size, shuffle=False)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

loss_fn = nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None,
                               reduction='mean')
optim1 = optim.Adam(ground_model.parameters(), lr=1e-5)
optim2 = optim.Adam(polar_model.parameters(), lr=1e-5)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ground_model = ground_model.to(device)
polar_model = polar_model.to(device)
loss_data = []
for epoch in trange(Epoch):
    best_ac = 100
    for step, (ground_image, air_image, air_image_other, idx) in enumerate(dataloader):
        ground_image, air_image, air_image_other = \
            ground_image.to(device), air_image.to(device), air_image_other.to(device)
        ground_feature = ground_model(ground_image[0])
        polar_feature = polar_model(air_image[0])
        polar_feature_old = polar_feature
        polar_feature_other = polar_model(air_image_other[0])
        polar_feature = ShiftCrop.Corr(ground_feature, polar_feature)
        # print(ground_feature.shape)
        # print(polar_feature.shape)
        # print(polar_feature_other.shape)
        r"""
        Triplet_Loss(Anchor, Positive, Negative)
        Anchor = ground_feature
        Positive = polar_feature
        Negative = polar_feature (other)
        """
        r'''
        **Choose one**
        Negative Example
        '''
        # loss = loss_fn(ground_feature, polar_feature, polar_feature_old)
        loss = loss_fn(ground_feature, polar_feature, polar_feature_other)
        if loss.item() < best_ac:
            torch.save(polar_model.state_dict(), 'polar.pth')
            torch.save(ground_model.state_dict(), 'ground.pth')
            torch.save(polar_feature, 'polar_feature.pth')
        optim1.zero_grad()
        optim2.zero_grad()
        loss.backward()
        optim1.step()
        optim2.step()
        loss_data.append([epoch, step, loss.item()])
        # print("epoch: " + str(epoch) + '    ' + 'training_loss:  ' + str(loss.item()))
        print("Epoch: {}; Step: {}; Loss: {}".format(int(epoch), int(step), loss.item()))

df = pd.DataFrame({'Epoch': loss_data[:, 0],
                   'Step': loss_data[:, 1],
                   'Loss': loss_data[:, 2]})
df.to_csv('result.csv')
