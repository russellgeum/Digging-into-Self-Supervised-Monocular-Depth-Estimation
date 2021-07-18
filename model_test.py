import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from pprint import pprint 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_parser import *
from model_utility import *

from model_loss import *
from model_layer import *
from model_test import *
from model_dataloader import *


def disparity2depth(disp, min_depth, max_depth):
    """
    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth    = 1 / scaled_disp
    return scaled_disp, depth


def compute_depth_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_depth_losses(inputs, outputs):
    """
    Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    depth_pred = outputs
    depth_pred = torch.clamp(F.interpolate(
        depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    depth_gt = inputs[("depth", 0)]
    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min = 1e-3, max = 80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)
    return depth_errors


def inference_for_eval(loader, encoder_weight, decoder_wiehgt, device):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 100
    HEIGHT    = 375
    WIDTH     = 1242
    
    metric      = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    metric_dict = {}
    metric_dict.update({key : [] for key in metric})

    ### 
    model_state_dict = {
        "encoder": encoder_weight,
        "decoder": decoder_wiehgt}
    encoder_weight = torch.load(model_state_dict["encoder"])
    decoder_weight = torch.load(model_state_dict["decoder"])

    model = {}
    model["encoder"] = ResnetEncoder(18, None).to(device)
    filtered_encoder = {
        k: v for k, v in encoder_weight.items() if k in model["encoder"].state_dict()};
    model["encoder"].load_state_dict(filtered_encoder)
    model["encoder"].eval()

    model["decoder"] = DepthDecoder(model["encoder"].num_ch_enc, range(4)).to(device)
    model["decoder"].load_state_dict(decoder_weight)
    model["decoder"].eval()

    for _, inputs in tqdm(enumerate(loader)):
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        with torch.no_grad():
            outputs   = model["decoder"](model["encoder"](inputs[("color", 0, 0)]))
            dispairty = outputs[("disp", 0)]
            dispairty = F.interpolate(
                dispairty, (HEIGHT, WIDTH), mode = "bilinear", align_corners = False)

            _, depth  = disparity2depth(dispairty, MIN_DEPTH, MAX_DEPTH)
            errors    = compute_depth_losses(inputs, depth)

        for index, name in enumerate(metric):
            metric_dict[name].append(errors[index].item())
            
    for key in metric:
        metric_dict[key] = np.mean(metric_dict[key])
        print("  {} {:0.3f}".format(key, metric_dict[key]), end = " ")
    return outputs



if __name__ == "__main__":
    device   = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    testpath = {
        "kitti_benchmark":       "./splits/kitti_test/kitti_benchmark_test_files.txt",
        "kitti_eigen_benchmark": "./splits/kitti_test/kitti_eigen_benchmark_test_files.txt",
        "kitti_eigen_test":      "./splits/kitti_test/kitti_eigen_test_files.txt"}

    test_filename = readlines(testpath["kitti_eigen_benchmark"])
    test_dataset  = KITTIMonoDataset("./dataset/kitti", test_filename, False, [0], ".jpg", 1)
    test_loader   = DataLoader(test_dataset, batch_size = 16, shuffle = False, drop_last = True)
    print(len(test_filename), test_loader.__len__())

    encoder_path = {
        "monodepth2_640x192": "./model_save/monodepth2/mono_640x192/encoder.pth",
        "separate_benchmark": "./model_save/reproduce/encoder.pt"}
    decoder_path = {
        "monodepth2_640x192": "./model_save/monodepth2/mono_640x192/depth.pth",
        "separate_benchmark": "./model_save/reproduce/decoder.pt"}

    # Monodepth2
    standard = inference_for_eval(
        test_loader, encoder_path["monodepth2_640x192"], decoder_path["monodepth2_640x192"], device)

    # Monodepth2-custom
    custom1  = inference_for_eval(
        test_loader, encoder_path["separate_benchmark"], decoder_path["separate_benchmark"], device)