# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


import torch
import torch.nn.functional as F

def ssim_score(tensor1, tensor2):
    """
    SSIM的数值范围在 -1 到 1 之间，当两个图像完全相同时为 1, 两个图像完全不同时为 -1。值越接近 1, 表示两个图像越相似；值越接近 -1, 表示两个图像越不相似。
    """
    # 计算亮度的均值
    mu1 = tensor1.mean(dim=(1, 2, 3), keepdim=True)
    mu2 = tensor2.mean(dim=(1, 2, 3), keepdim=True)

    # 计算亮度的标准差
    sigma1 = torch.sqrt(((tensor1 - mu1) ** 2).mean(dim=(1, 2, 3), keepdim=True))
    sigma2 = torch.sqrt(((tensor2 - mu2) ** 2).mean(dim=(1, 2, 3), keepdim=True))

    # 计算亮度的协方差
    sigma12 = ((tensor1 - mu1) * (tensor2 - mu2)).mean(dim=(1, 2, 3), keepdim=True)

    # SSIM参数设置
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # 计算SSIM指数
    ssim = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
    # 取每个样本的SSIM平均值
    ssim_mean = ssim.mean()
    return ssim_mean.item()


def ssim_loss(pred,
              encode_features,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    pred_mask = F.sigmoid(pred)
    loss_all = 0.0
    for fA, fB in zip(encode_features[0],encode_features[1]):
        foreA = fA*pred_mask
        foreB = fB*pred_mask
        backA = fA*(1-pred_mask)
        backB = fB*(1-pred_mask)
        # 计算前景和背景的SSIM
        ssim_fore = ssim_score(foreA, foreB)
        ssim_back = ssim_score(backA, backB)

        # 对a的目标进行优化，使其更接近-1
        loss_a = (ssim_fore + 1)**2
        # 对b的目标进行优化，使其更接近1
        loss_b = (ssim_back - 1)**2

        # 将两个loss进行加权组合
        loss = class_weight[0]*loss_a + class_weight[1]*loss_b
        loss_all+=loss
    return loss_all


@MODELS.register_module()
class CDLoss(nn.Module):                              
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_cd',
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')
        self.cls_criterion = ssim_loss
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                encode_features,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            encode_features,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
