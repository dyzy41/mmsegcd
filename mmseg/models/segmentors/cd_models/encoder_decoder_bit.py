# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..base import BaseSegmentor
from mmseg.models.decode_heads.decode_head_base import BaseDecodeHeadForCD

from .bit import *


@MODELS.register_module()
class BIT(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.decode_head = BaseDecodeHeadForCD()
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        # TODO: reduce hard-coded parameters
        dim = 32
        mlp_dim = 2*dim
        chn = dim
        in_ch=3
        out_ch=2
        backbone='resnet18'
        n_stages=4
        use_tokenizer=True
        token_len=4
        pool_mode='max'
        pool_size=2
        enc_with_pos=True
        enc_depth=1
        enc_head_dim=64
        dec_with_softmax=True
        dec_depth=8
        dec_head_dim=8

        self.backbone = Backbone(in_ch, chn, arch=backbone, n_stages=n_stages)

        self.use_tokenizer = use_tokenizer
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.conv_att = Conv1x1(32, token_len, bias=False)
            self.token_len = token_len

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = nn.Parameter(torch.randn(1,self.token_len*2,chn))

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_head_dim = enc_head_dim
        self.dec_head_dim = dec_head_dim
        
        self.encoder = TransformerEncoder(
            dim=dim, 
            depth=enc_depth, 
            n_heads=8,
            head_dim=enc_head_dim,
            mlp_dim=mlp_dim,
            dropout_rate=0.
        )
        self.decoder = TransformerDecoder(
            dim=dim, 
            depth=dec_depth,
            n_heads=8, 
            head_dim=dec_head_dim, 
            mlp_dim=mlp_dim, 
            dropout_rate=0.,
            apply_softmax=dec_with_softmax
        )

        self.classifier = DoubleConv(chn, out_ch)

    def bit_get_semantic_tokens(self, x):
        b, c = x.shape[:2]
        att_map = self.conv_att(x)
        att_map = att_map.reshape((b,self.token_len,1,-1))
        att_map = F.softmax(att_map, dim=-1)
        x = x.reshape((b,1,c,-1))
        tokens = (x*att_map).sum(-1)
        return tokens

    def bit_get_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.permute((0,2,3,1)).flatten(1,2)
        return tokens

    def bit_encode(self, x):
        if self.enc_with_pos:
            x += self.enc_pos_embedding
        x = self.encoder(x)
        return x

    def bit_decode(self, x, m):
        b, c, h, w = x.shape
        x = x.permute((0,2,3,1)).flatten(1,2)
        x = self.decoder(x, m)
        x = x.transpose(1,2).reshape((b,c,h,w))
        return x

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        # if self.with_neck:
        #     x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)
        
        seg_logits = self.decode_predict([xA, xB], batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
    
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        losses = dict()

        loss_decode = self._decode_head_forward_train([xA, xB], data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)
        x = [xA[i]+xB[i] for i in range(len(xA))]
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    def decode_loss(self, inputs: List[Tensor],
                                   data_samples: SampleList, cfg) -> dict:
        # inputs is the bitemporal feature list, process them into seg_logits, the size is random and the channel is the num_class.
        seg_logits = self.decode_forward(inputs)
        
        loss_decode = self.decode_head.loss_by_feat(seg_logits, data_samples)

        return loss_decode
    
    def decode_predict(self, inputs, batch_img_metas,
                test_cfg) -> Tensor:
        seg_logits = self.decode_forward(inputs)

        return self.decode_head.predict_by_feat(seg_logits, batch_img_metas)
    
    def decode_forward(self, inputs):
        x1, x2 = inputs[0], inputs[1]
                # Tokenization
        if self.use_tokenizer:
            token1 = self.bit_get_semantic_tokens(x1)
            token2 = self.bit_get_semantic_tokens(x2)
        else:
            token1 = self.bit_get_reshaped_tokens(x1)
            token2 = self.bit_get_reshaped_tokens(x2)

        # Transformer encoder forward
        token = torch.cat([token1, token2], dim=1)
        token = self.bit_encode(token)
        token1, token2 = torch.chunk(token, 2, dim=1)

        # Transformer decoder forward
        y1 = self.bit_decode(x1, token1)
        y2 = self.bit_decode(x2, token2)

        # Feature differencing
        y = torch.abs(y1 - y2)
        # Classifier forward
        pred = self.classifier(y)
        return pred
