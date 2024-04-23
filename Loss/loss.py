# from .hausdorff.hausdorff import hausdorff_distance
import numpy as np
import cv2
import torch
import torch.nn as nn


# def HDloss(img1, img2):
#     h, w = img1.shape
#     point1 = []
#     point2 = []
#     for i in range(h):
#         for j in range(w):
#             if (img1[i, j] > 0):
#                 point1.append((i, j))
#             if (img2[i, j] > 0):
#                 point2.append((i, j))
#     point1 = np.array(point1)
#     point2 = np.array(point2)
#     return hausdorff_distance(point1, point2, distance='euclidean')


def ChamferDistanceLoss(img, target, distanceType=cv2.DIST_L2, maskSize=5, device='cpu', normed=False):
    if (normed):
        target = 1 - target
        target = target // np.max(target)
        dist = cv2.distanceTransform(src=target, distanceType=distanceType, maskSize=maskSize)
        dist = cv2.normalize(dist, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return np.sum(img * dist)
    target = 255 - target
    target = target // np.max(target) * 255
    dist = cv2.distanceTransform(src=target, distanceType=distanceType, maskSize=maskSize)
    dist = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return np.sum(img * dist)


class VesselLoss:
    def __init__(self, weight={'CD': 0.1, 'RH': 0.0001}, normed=False):
        # try1 {'CD': 0.001, 'RH': 0.0001}
        # no_vessel_loss : {'CD': 0, 'RH': 0}
        self.noremed = normed
        self.weight = weight
        self.Hingloss = torch.nn.HingeEmbeddingLoss(margin=0, reduction='sum')

    def __call__(self, vessel: torch.Tensor(), label: torch.Tensor()):
        vessel = (vessel + 1) / 2
        label = label + 1
        lossReHing = self.Hingloss(vessel - label, torch.Tensor([-1]).cuda())

        label = (label.squeeze(0).squeeze(0).detach().cpu().numpy() + 0.3).astype('uint8')

        # print(np.sum(label))
        # input('--------------')

        dist = cv2.distanceTransform(src=label, distanceType=cv2.DIST_L2, maskSize=5)
        dist = cv2.normalize(dist, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        dist = torch.Tensor(dist).cuda()
        lossCD = torch.sum(vessel * dist)
        # print(self.weight['CD']*lossCD)
        # print(self.weight['RH']*lossReHing)
        # input('--------------------------')
        # return {'RH': self.weight['RH'] * lossReHing, 'CD': self.weight['CD'] * lossCD}
        return self.weight['CD'] * lossCD, self.weight['RH'] * lossReHing


def MAEloss(img1, img2):
    return np.mean(np.abs(img1 - img2))


def MSEloss(img1, img2):
    return np.mean(np.square(img1 - img2))


def RMSEloss(img1, img2):
    return np.sqrt(np.mean(np.square(img1 - img2)))


from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        # print('/////////////////////////')
        # print(input.shape, target.shape)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))

        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # print(input.size())
    # print(target.size())
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def TVLoss_L1(x):
    diff_x = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    diff_y = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return diff_x + diff_y


def Depth_Aware_Loss(grid, use_sum = False):
    # horizontal
    w_edge = grid[:, 0:-1, :, :] - grid[:, 1:, :, :]
    w_loss = torch.sum(w_edge[:, 0:-1, :, :] * w_edge[:, 1:, :, :], -1) / torch.sqrt(
        torch.sum(w_edge[:, 0:-1, :, :] * w_edge[:, 0:-1, :, :], -1) * torch.sum(
            w_edge[:, 1:, :, :] * w_edge[:, 1:, :, :], -1))

    # vertical
    h_edge = grid[:, :, 0:-1, :] - grid[:, :, 1:, :]
    h_loss = torch.sum(h_edge[:, :, 0:-1, :] * h_edge[:, :, 1:, :], -1) / torch.sqrt(
        torch.sum(h_edge[:, :, 0:-1, :] * h_edge[:, :, 0:-1, :], -1) * torch.sum(
            h_edge[:, :, 1:, :] * h_edge[:, :, 1:, :], -1))

    if use_sum:
        loss = torch.sum(1 - w_loss) + torch.sum(1 - h_loss)
    else:
        loss = torch.mean(1 - w_loss) + torch.mean(1 - h_loss)
    return loss


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, features, g_ref):
        loss = 0
        # print(features.shape)
        # print(features.size())
        # input('------------')
        for feat, g_re in zip(features, g_ref):
            print(feat.shape)
            input('---------')
            g = gram_matrix(feat)
            g_r = gram_matrix(g_re)
            loss += self.mse(g, g_r)
        return loss
