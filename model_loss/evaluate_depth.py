# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from .model_loss import *
from model_layer import *

cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
STEREO_SCALE_FACTOR = 5.4
"""
Following up https://github.com/nianticlabs/monodepth2/tree/master/networks
"""
def compute_depth_error(ground_truth, prediction):
    """
    추정한 뎁스와 GT의 차이를 계산
    Args: 
        ground_truth: [B, 1, H, W]
        prediction: [B, 1, H, W]
    """
    threshold = torch.maximum((ground_truth / prediction), (prediction / ground_truth))
    
    a1        = threshold[(threshold < 1.25     )].mean()
    a2        = threshold[(threshold < 1.25 ** 2)].mean()
    a3        = threshold[(threshold < 1.25 ** 3)].mean()

    rmse      = (ground_truth - prediction) ** 2
    rmse      = torch.sqrt(rmse.mean())
    
    rmse_log  = (torch.log(ground_truth) - torch.log(prediction)) ** 2
    rmse_log  = torch.sqrt(rmse_log.mean())

    abs_rel   = torch.mean(torch.abs(ground_truth - prediction) / ground_truth)
    sqrt_rel  = torch.mean(((ground_truth - prediction) ** 2) / ground_truth)

    return a1, a2, a3, rmse, rmse_log, abs_rel, sqrt_rel


def compute_depth_metric(inputs, outputs):
    """
    입력 딕셔너리의 그라운드 트루스 뎁스와 출력 딕셔너리의 예측 뎁스 간,
    일치하는 부분만 반영해서 계산하는 함수
    Compute depth mterics, to allow monitoring during training
    This is not particularly accurate as it averages over the entire batch,
    So, It is only used to give an indication of validation performance
    """

    """
    1. 출력 딕셔너리에 있는 예측 뎁스를 GT 사이즈에 맞게 interpolate함
    2. 입력 딕셔너리에 있는 GT 뎁스를 불러오고 0 이상이면 True인 마스크를 만듬
    3. mask 크기만큼 crop_mask를 만들고 여기서 중앙 - 하단부 영역을 1로 만듬 (하단부가 공통으로 유효한 뎁스 영역이기 때문에)
    4. mask ([T, T, T, T]) 에 crop_mask ([F, T, T, F]) 를 곱하면 mask ([F, T, T, F]) 가 됨
    5. predict_depth와 ground_depth에 mask를 씌워서 해당 영역의 True 값만 골라냄
    6. predict
    """
    predict_depth = outputs[("depth", 0, 0)]
    predict_depth = torch.clamp(
        F.interpolate(predict_depth, [375, 1245], mode = "bilinear", align_corners = False), 1e-3, 80)
    predict_depth = predict_depth.detach()
    
    # Ground Truth
    ground_depth  = inputs[("depth", 0)]
    mask          = ground_depth > 0

    crop_mask     = torch.zeros_like(mask)
    crop_mask[:, :, 175:315, 44:980] = 1
    mask          = mask * crop_mask

    ground_depth  = ground_depth[mask]
    predict_depth = predict_depth[mask]
    predict_depth *= torch.median(ground_depth) / torch.median(predict_depth)
    predict_depth = torch.clamp(predict_depth, min = 1e-3, max = 80)

    detph_error   = compute_depth_error(ground_truth = ground_depth, prediction = predict_depth)
    return detph_error


def pose_process_disparity(l_disp, r_disp):
    """
    1. 어떤 함수인지 잘 모르겠음
    """
    _, h, w = l_disp.shape
    m_disp  = 0.5 * (l_disp + r_disp)
    l, _    = np.meshgrid(np.linsapce(0, 1, w), np.linspace(0, 1, h))
    l_mask  = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask  = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
