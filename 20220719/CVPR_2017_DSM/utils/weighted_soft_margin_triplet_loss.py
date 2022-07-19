# import torch
# from torch import nn
#
# triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2, reduction='mean')
# anchor = torch.Tensor([[1, 0, 1], [1, 1, 1]])
# positive = torch.Tensor([[4, 1, 5], [2, 2, 2]])
# negative = torch.Tensor([[3, 1, 2], [1, 1, 2]])
#
# loss = triplet_loss(anchor, positive, negative)
# print(loss)
r"""
具体怎么算出来的呢？

对于第一组向量：a = [1,0,1], p=[4,1,5], n =[3,1,2]

dap = math.sqrt((1-4)^2 + (0-1)^2 + (1-5)^2 ) = math.sqrt(9 +1 + 16) = 5.099

dan = math.sqrt((1-3)^2 + (0-1)^2 + (1-2)^2 ) = math.sqrt(4 + 1 +1) = 2.4495

max (dap - dan + 0.3 , 0 ) = 2.9495

对于第二组向量：a = [1,1,1], p=[2,2,2], n =[1,1,2]

dap = math.sqrt((1-2)^2 + (1-2)^2 + (1-2)^2) = math.sqrt(3) = 1.732

dan = math.sqrt((1-1)^2 + (1-1)^2 + (1-2)^2) = 1

max(dap - dan + 0.3, 0) = 1.032

最后求平均：（2.9495 + 1.032）/2 = 1.9908

"""
