import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model_utility import *
from model_loader import *
from model_layer import *
from model_loss import *



class compute(object):
    def __init__(self, opt, device):
        """
        class compute
            def forward_network
            def pose_estimation
            def depth2warping
            def loss_summation
        """
        self.opt    = opt
        self.device = device
        if opt.pose_frames == "all":
            self.num_pose_frames = len(opt.frame_ids)
        else:
            self.num_pose_frames = 2


    def forward_depth(self, inputs, outputs, setting):
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        if self.opt.pose_type == "posecnn":
            output["features"] = setting.model["encoder"](inputs[("color_aug", 0, 0)])
            outputs.update(setting.model["decoder"](output["features"]))
        
        elif self.opt.pose_type == "shared":
            all_frames   = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.opt.frame_ids])
            all_features = self.model["encoder"](all_frames)
            all_features = [torch.split(feature, self.opt.batch) for feature in all_features]

            for index, frame_id in enumerate(self.opt.frame_ids):
                outputs.update({index: [feature[index] for feature in all_features]})
            outputs.update(setting.model["decoder"](features[0]))

        elif self.opt.pose_type == "separate": # separate 타입이면 posecnn처럼 이미지 한 장을 포워드
            outputs["features"] = setting.model["encoder"](inputs[("color_aug", 0, 0)])
            outputs.update(setting.model["decoder"](outputs["features"]))
        
        return inputs, outputs


    def forward_pose(self, inputs, outputs, setting):
        """
        포즈를 계산하는 함수
        프레임 아이디가 [0, -2, -1, 1, 2] 이면 [-2, -1, 1, 2]를 순회해서
        [-2, 0], [-1, 0], [0, 1], [0, 2] 형태로 묶어, backward pose, forward pose를 추정
        framd_id < 0 -> 0이 뒤에 오게 묶고, frame_id < 0이므로 invert matrix = True
        frame_id > 0 -> 0이 앞에 오게 묶고, frame_id > 0이므로 invert matrix = False
        """
        if self.num_pose_frames == 2:
            if self.opt.pose_type == "posecnn":
                all_frames = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.opt.frame_ids}

                for frame_id in self.opt.frame_ids[1: ]:
                    if frame_id < 0:
                        pose_inputs = torch.cat([all_frames[frame_id], all_frames[0]], dim = 1)
                    else:
                        pose_inputs = torch.cat([all_frames[0], all_frames[frame_id]], dim = 1)
                    
                    axisangle, translation        = setting.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))

            elif self.opt.pose_type == "shared":
                all_features = {frame_id: outputs[frame_id] for frame_id in self.opt.frame_ids}

                for frame_id in self.opt.frame_ids[1: ]:
                    if frame_id < 0:
                        pose_inputs = [all_features[frame_id], all_features[0]]
                    else:
                        pose_inputs = [all_features[0], all_features[frame_id]]

                    axisangle, translation        = setting.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))

            elif self.opt.pose_type == "separate":
                all_frames = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.opt.frame_ids}

                for frame_id in self.opt.frame_ids[1: ]:
                    if frame_id < 0:
                        pose_inputs = torch.cat([all_frames[frame_id], all_frames[0]], dim = 1)
                    else:
                        pose_inputs = torch.cat([all_frames[0], all_frames[frame_id]], dim = 1)

                    pose_inputs                   = [setting.model["pose_encoder"](pose_inputs)]
                    axisangle, translation        = setting.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))
                    
        else:
            if self.opt.pose_type == "posecnn":
                all_frames             = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.opt.frame_ids], dim =1)
                axisangle, translation = setting.model["pose_decoder"](all_frames)
            elif self.opt.pose_type == "shared":
                all_features           = [features[frame_id] for frame_id in self.opt.frame_ids]
                axisangle, translation = setting.model["pose_decoder"](all_features)
            elif self.opt.pose_type == "separate":
                all_frames             = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.opt.frame_ids], dim = 1)
                pose_inputs            = [setting.model["pose_encoder"](all_frames)]
                axisangle, translation = setting.model["pose_decoder"](pose_inputs)
            
            for index, frame_id in enumerate(self.opt.frame_ids[1: ]):
                outputs[("R", frame_id, 0)]   = axisangle
                outputs[("T", frame_id, 0)]   = translation
                outputs[("c2c", frame_id, 0)] = param2matrix(
                    axisangle = axisangle[:, index], translation = translation[:, index])
        return inputs, outputs  # outputs에 "R", "T", "cam2cam" 키를 가지고 나옴


    def image2warping(self, inputs, outputs, setting): # 뎁스 계산
        for scale in self.opt.scales:
            disparity = outputs[("disp", scale)]
            disparity = interpolate(
                disparity, self.opt.height, self.opt.width, "bilinear", False)
            _, depth  = disparity2depth(
                disparity, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth # saving depth for depth eval

            for frame_id in self.opt.frame_ids[1: ]:
                # shared는 monodepth2의 arxiv ver1, separate가 monodepth2의 arxiv ver2
                if self.opt.pose_type in ["shared", "separate"]:
                    transformation = outputs[("c2c", frame_id, 0)]

                # https://arxiv.org/abs/1712.00175
                elif self.opt.pose_type == "posecnn":
                    axisangle, translation = outputs[("R", frame_id, 0)], outputs[("T", frame_id, 0)]
                    mean_inv_depth         = (1 / depth).mean(3, True).mean(2, True)
                    transformation         = param2matrix(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], (frame_id < 0))

                camera_coords = setting.model["inv_projection"](depth, inputs[("inv_K", 0)])
                frame_coords  = setting.model["for_projection"](camera_coords, inputs[("K", 0)], transformation)
                outputs[("warp_color", frame_id, scale)] = grid_sample(
                    inputs[("color", frame_id, 0)], frame_coords, "border", True)
        return inputs, outputs # outputs에 "depth", warp_color" 키 값들을 가지고 리턴


    def compute_loss(self, inputs, outputs, setting):
        total_loss = 0
        for scale in self.opt.scales: # 0 1 2 3
            scale_loss = 0
            disp   = outputs[("disp", scale)]    # 다양한 사이즈의 스케일 (4개)
            color  = inputs[("color", 0, scale)] # 다양한 사이즈의 스케일 (4개)
            target = inputs[("color", 0, 0)]     # scale = 0의 타겟 이미지
            
            # target과 warping 이미지에 대해서 reprojection_loss 계산
            reprojection_loss = []
            for frame_id in self.opt.frame_ids[1: ]:
                prediction = outputs[("warp_color", frame_id, scale)]
                reprojection_loss.append(setting.loss["reprojection"](prediction, target))
            reprojection_loss = torch.cat(reprojection_loss, 1)
            
            # Contribution 1. automasking loss
            # target과 color 이미지에 대해서 reprojection loss 계산
            # -> 변하지 않는 부분은 로스가 작을 것
            if self.opt.use_automasking:
                identity_loss = []
                for frame_id in self.opt.frame_ids[1: ]:
                    prediction = inputs[("color", frame_id, 0)]
                    identity_loss.append(setting.loss["reprojection"](prediction, target))
                identity_loss = torch.cat(identity_loss, 1)
            

            if self.opt.use_automasking:
                identity_loss += 0.00001 * torch.randn(identity_loss.shape).to(self.device)
                combined_loss = torch.cat((identity_loss, reprojection_loss), dim = 1)
            else:
                combined_loss = reprojection_loss

            # Contribution 1. minimum loss
            if combined_loss.shape[1] == 1:
                to_optimise = combined_loss
            else:
                to_optimise, idxs = torch.min(combined_loss, dim = 1)

            # disp와 color로 smooth loss 계산
            smooth_loss = setting.loss["edge_aware"](disp = disp, color = color)
            
            # Loss summation
            scale_loss = scale_loss + to_optimise.mean()
            scale_loss = scale_loss + self.opt.disp_smoothness * smooth_loss / (2 ** scale)
            total_loss = total_loss + scale_loss
        
        total_loss      = total_loss / len(self.opt.scales)
        outputs["loss"] = total_loss
        return outputs