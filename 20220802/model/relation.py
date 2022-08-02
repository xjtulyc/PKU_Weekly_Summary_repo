import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.faster_rcnn_utils import TwoMLPHead


def ReLU(input):
    return nn.ReLU(input)

def Relative_Geometry_Feature(f_G_m, f_G_n):
    return torch.Tensor([])

def relation_attention(query, key, value, f_R_mn, mask=None, dropout=None):
    d_k = query.size(-1)
    w_mn_A = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    w_mn_G = ReLU(f_R_mn)
    scores = torch.matmul(w_mn_G, torch.exp(w_mn_A)).view(-1)  # w_mn
    scores /= torch.sum(scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        pass
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        pass
    return torch.matmul(p_attn, value.view(-1)), p_attn


class Relation(nn.Module):
    def __init__(self, f_A_shape, roi_out_size):
        super(Relation, self).__init__()
        self.f_A_shape = f_A_shape  # 1024
        self.roi_out_size = roi_out_size  # [7,7]
        self.dropout = nn.Dropout()
        self.mask = None
        # K
        self.K_network = self.K_network_init()
        # Q
        self.Q_network = self.Q_network_init()
        # V
        self.V_network = self.V_network_init()
        # embedding
        self.embedding_network = self.embedding_network_init()
        # parameter initialization
        self.para_init()
        pass

    def forward(self, f_A, f_G_n, f_G_m):
        f_A, f_G_n, f_G_m = f_A.view(-1), f_G_n.view(-1), f_G_m.view(-1)
        f_R = None
        V = self.V_network(f_A)
        Q = self.Q_network(f_A)
        K = self.K_network(f_A)
        f_R_mn = self.embedding_network(f_G_n, f_G_m)
        f_R, p_attn = relation_attention(query=Q,
                                         key=K,
                                         value=V,
                                         f_R_mn=f_R_mn,
                                         mask=self.mask,
                                         dropout=self.dropout)
        return f_R

    def V_network_init(self):
        '''
        MLP
        :param f_A:
        :return:
        '''
        return TwoMLPHead(self.f_A_shape, self.f_A_shape)

    def Q_network_init(self):
        '''
        MLP
        :param f_A:
        :return:
        '''
        return TwoMLPHead(self.f_A_shape, self.f_A_shape)

    def K_network_init(self):
        '''
        MLP
        :param f_A:
        :return:
        '''
        return TwoMLPHead(self.f_A_shape, self.f_A_shape)

    def embedding_network_init(self):
        return TwoMLPHead(4, self.f_A_shape)

    def para_init(self):
        pass
