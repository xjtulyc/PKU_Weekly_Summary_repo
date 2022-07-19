import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import model.feature_extractor as model
from dataset.dataset import innoDataset

ground_model = model.vgg16net()
ground_dict = torch.load('ground.pth')
ground_model.load_state_dict(ground_dict)
# print(ground_dict)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ground_model = ground_model.double()
ground_model = ground_model.to(device)

transform = transforms.Compose([
    transforms.Resize([1024, 256]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
data = innoDataset()

dataloader = DataLoader(data, batch_size=1, shuffle=False)
polar_feature = torch.load('polar_feature.pth').to(device)
# print(polar_feature)
loss_fn = nn.MSELoss(reduction='mean')
with torch.no_grad():
    for step, (ground_image, _, _, label) in enumerate(dataloader):
        ground_image = ground_image.to(device)
        label = label.to(device)
        # print(ground_image[0])
        result = ground_model(ground_image[0])
        loss = None
        for idx, j in enumerate(polar_feature):
            j = j.view([1, 4, 64, 16]).to(device)
            if loss == None:
                # print(type(j))
                loss = loss_fn(result, j)
                pre = idx
            elif loss_fn(result, j) < loss:
                loss = loss_fn(result, j)
                pre = idx
        if pre == label:
            print('true')
        else:
            print('false')
