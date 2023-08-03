# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead
from ..losses import accuracy


@MODELS.register_module()
class FPNHeadCDLoss(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output

    def loss_by_feat(self, seg_logits,
                     batch_data_samples, encode_features) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        encode_featuresA, encode_featuresB = encode_features[0], encode_features[1]
        encode_featuresA = [resize(
                input=item,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for item in encode_featuresA]
        encode_featuresB = [resize(
                input=item,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for item in encode_featuresB]

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name != "loss_cd":
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        [encode_featuresA, encode_featuresB],
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            else:
                if loss_decode.loss_name != "loss_cd":
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        [encode_featuresA, encode_featuresB],
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def loss(self, inputs, batch_data_samples,
             train_cfg, encode_features):
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples, encode_features)
        return losses

    def predict(self, inputs, batch_img_metas,
                test_cfg):
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, decode_features = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)