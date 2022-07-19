import torch


def Corr(ground_feature, polar_feature):
    # troch.max()[1] return the index of the max one
    shift_width = int(ground_feature.shape[2])  # torch.size([1,4,64,16])
    corr_distance = torch.zeros(shift_width)
    best_start = None
    for start in range(shift_width):
        corr_distance[start] = torch.dot(ground_feature.view(-1),
                                         torch.cat((polar_feature[:, :, start:, :],
                                                    polar_feature[:, :, :start, :]), 2).view(-1))
        pass
    best_start = int(torch.max(corr_distance, 0).indices)
    return torch.cat((polar_feature[:, :, best_start:, :],
                      polar_feature[:, :, :best_start, :]), 2)
