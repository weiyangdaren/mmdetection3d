# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.model import stack_batch
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptConfigType
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.data_preprocessors.utils import multiview_img_stack_batch
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape, dynamic_scatter_3d
# from .utils import multiview_img_stack_batch
# from .voxelize import VoxelizationByGridShape, dynamic_scatter_3d


@MODELS.register_module()
class OmniDet3DDataPreprocessor(Det3DDataPreprocessor):

    def __init__(self,
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 batch_first: bool = True,
                 max_voxels: Optional[int] = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False,
                 batch_augments: Optional[List[dict]] = None,
                 img_keys: Optional[List[str]] = None,) -> None:

        super().__init__(
            voxel=voxel,
            voxel_type=voxel_type,
            voxel_layer=voxel_layer,
            batch_first=batch_first,
            max_voxels=max_voxels,
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments,)
        self.img_keys = img_keys if img_keys is not None else [
            'cam_rgb', 'cam_nusc', 'cam_dvs', 'cam_fisheye']
        self.point_keys = ['points', 'lidar_points', 'semantic_lidar_points']

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """

        # get the pad_shape of each image
        img_pad_shape = {}
        for img_key in self.img_keys:
            if img_key not in data['inputs']:
                continue
            batch_pad_shape = self._get_pad_shape(data['inputs'][img_key])
            img_pad_shape[img_key] = batch_pad_shape

        # collate image data and transfer to the target device
        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        for key in self.point_keys:
            if key not in inputs:
                continue
            batch_inputs[key] = inputs[key]
            if self.voxel:
                voxel_dict = self.voxelize(inputs[key], data_samples)
                batch_inputs[f'{key}_voxels'] = voxel_dict

        for img_key in self.img_keys:
            # if 'imgs' in inputs:
            imgs = inputs[img_key]

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  img_pad_shape[img_key]):
                    data_sample.metainfo[img_key].update({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs[img_key] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        # channel transform
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        _batch_img = _batch_img.float()
        # Normalization.
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    'If the mean has 3 values, the input tensor '
                    'should in shape of (3, H, W), but got the '
                    f'tensor with shape {_batch_img.shape}')
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:
        """Copy data to the target device and perform normalization, padding
        and bgr2rgb conversion and stack based on ``BaseDataPreprocessor``.

        Collates the data sampled from dataloader into a list of dict and list
        of labels, and then copies tensor to the target device.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore

        for key in self.img_keys:
            if key in data['inputs']:
                _batch_imgs = data['inputs'][key]
                # Process data with `pseudo_collate`.
                if is_seq_of(_batch_imgs, torch.Tensor):
                    batch_imgs = []
                    img_dim = _batch_imgs[0].dim()
                    for _batch_img in _batch_imgs:
                        if img_dim == 3:
                            _batch_img = self.preprocess_img(_batch_img)
                        elif img_dim == 4:
                            _batch_img = [
                                self.preprocess_img(_img) for _img in _batch_img
                            ]
                            _batch_img = torch.stack(_batch_img, dim=0)
                        batch_imgs.append(_batch_img)

                    if img_dim == 3:
                        batch_imgs = stack_batch(
                            batch_imgs, self.pad_size_divisor, self.pad_value)

                    elif img_dim == 4:
                        batch_imgs = multiview_img_stack_batch(
                            batch_imgs, self.pad_size_divisor, self.pad_value)

                # Process data with `default_collate`.
                elif isinstance(_batch_imgs, torch.Tensor):
                    assert _batch_imgs.dim() == 4, (
                        'The input of `ImgDataPreprocessor` should be a NCHW '
                        'tensor or a list of tensor, but got a tensor with '
                        f'shape: {_batch_imgs.shape}')
                    if self._channel_conversion:
                        _batch_imgs = _batch_imgs[:, [2, 1, 0], ...]
                    # Convert to float after channel conversion to ensure
                    # efficiency
                    _batch_imgs = _batch_imgs.float()
                    if self._enable_normalize:
                        _batch_imgs = (_batch_imgs - self.mean) / self.std
                    h, w = _batch_imgs.shape[2:]
                    target_h = math.ceil(
                        h / self.pad_size_divisor) * self.pad_size_divisor
                    target_w = math.ceil(
                        w / self.pad_size_divisor) * self.pad_size_divisor
                    pad_h = target_h - h
                    pad_w = target_w - w
                    batch_imgs = F.pad(_batch_imgs, (0, pad_w, 0, pad_h),
                                       'constant', self.pad_value)

                else:
                    raise TypeError(
                        'Output of `cast_data` should be a list of dict '
                        'or a tuple with inputs and data_samples, but got '
                        f'{type(data)}: {data}')

                data['inputs'][key] = batch_imgs
        data.setdefault('data_samples', None)
        return data

    def _get_pad_shape(self, data: dict) -> List[Tuple[int, int]]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        # rewrite `_get_pad_shape` for obtaining image inputs.
        _batch_inputs = data
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got '
                            f'{type(data)}: {data}')
        return batch_pad_shape
