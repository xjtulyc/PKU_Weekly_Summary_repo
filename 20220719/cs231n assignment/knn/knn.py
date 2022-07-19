import numpy as np
import torch
from tqdm import tqdm

from dataset.cifar_dataset import trainloader

num_train = 10000
num_test = 5000

loss_fn = torch.nn.MSELoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def top_k(distance, k):
    distance = distance.to('cpu').numpy()
    distance = distance[np.argsort(distance[:, 1]), :]
    return_list = distance[:k, 0]
    return int(return_list[np.random.randint(k)])


# knn分类函数，输入训练样本，训练样本标签，测试样本(为ndarray)，k值（k默认为3）；返回预测标签
def KNN(testdata, k=3):
    image, label = None, None
    distance = torch.Tensor([[-1, torch.inf]]).to(device)
    for step, (image, label) in enumerate(tqdm(trainloader)):
        image, label, testdata = image.to(device), label.to(device), testdata.to(device)
        loss = loss_fn(image, testdata)
        distance = torch.cat((distance, torch.Tensor([[label, loss.item()]]).to(device)), dim=0)
        pass
    return top_k(distance, k)
