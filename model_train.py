from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_layer import *
from model_loss import *
from model_dataloader import *
from model_parser import *
from model_logger import *
from model_utility import *

"""
2021 07 07 디버깅
model_loss 폴더의 SSIM loss에서
원래 코드
SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
실수 코드
SSIMd = (intensity_y ** 2 + intensity_y ** 2 + self.C1) * (sigma_x ** 2 + sigma_y ** 2 + self.C2)
--> 성능 상승 조금 있음


2021 07 07 디버깅
def compute_pose에서 self.frame_ids: 로 되어, 0까지 모두 포함하여 계산하는 오류
if frameid < 0을 만족하지 않으므로 [0, 0] concat이 되고 포즈를 계산하는 것이 무의미 -> 당연히 포즈가 안 정확
--> 성능 상승 조금 있음

2021 07 07 디버깅
파이토치 1.3까지 F.grid_sample는 align_corners가 기본 False임
파이토치 1.4부터 align_corners가 기본 True로 변경
--> 성능에 크게 상승을 줄까? -> False로 주는게 맞음

2021 07 09 디버깅
이미지 리사이즈 시, interpolation을 Image.ANAIRAS를 주거나 이거와 동일한 cv2.INTER_AREA를 줄 것
"""

# Templet
option_templet1 = ">>>   Epoch {0:19d}   Batch size {1:5d}    Learning rate {2:0.5f}"
option_templet2 = ">>>   Scheduler step {0:10d}   Smoothness weight   {1:4.3f}"
option_templet3 = ">>>   Image Size {0:10d} {1:3d}   min-max Depth   {2:8.2f} {3:0.2f}"
option_templet4 = ">>>   Train iteration      {0:3d}   Valid iteration     {1:3d}"
option_templet5 = ">>>   Train batch iteration     {0:4d}   Valid batch iteration   {1:4d}"


# pytorch randomnetss
def pytorch_randomness(random_seed = 777):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


class trainer(object):
    def __init__(self, options):
        pytorch_randomness()
        self.device           = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(">>>   Using of CUDA  :  ", self.device)
        self.opt              = options
        self.epoch            = options.epoch
        self.batch_size       = options.batch
        self.learning_rate    = options.learning_rate
        self.scheduler_step   = options.scheduler_step
        self.min_depth        = options.min_depth # 모델에 입력할 이미지의 해상도와 min depth, max depth 정의
        self.max_depth        = options.max_depth

        # 아래로는 이미지 사이즈와 뎁스 옵션에 대한 정의 (for KITTI)
        self.frame_ids        = options.frame_ids    # [-2, -1, 0, 1, 2]
        self.num_frames       = len(self.frame_ids)  # len([-2, -1, 0, 1, 2])
        if options.pose_frames == "all":
            self.num_pose_frames = self.num_frames
        else:
            self.num_pose_frames = 2
        
        self.dataset = self.opt.datapath.split("/")[2]
        print(">>>   Using dataset  :  ", self.dataset)
        if self.dataset == "kitti":
            self.original_scale   = (375, 1242)
            self.default_scale    = [(320, 1024), (192, 640), (96, 320), (48, 160)]
        elif self.dataset == "cityscapes":
            self.original_scale   = (1024, 2048)
            self.default_scale    = [(512, 1024), (256, 512), (128, 256), (64, 128)]

        self.scale            = options.scale # 0, 1, 2, 3 중 하나, default_scale에서 어떤 스케일을 쓸지 선택
        self.scales           = options.scales       # 스케일의 범위 [0, 1, 2, 3] (현재 지정한 self.scale로부터 시작)
        self.num_scales       = len(options.scales)  # [0, 1, 2, 3]의 길이는 4
        self.resolution       = [(self.default_scale[self.scale][0]//(2**i), 
                                  self.default_scale[self.scale][1]//(2**i))
                                  for i in self.scales]

        self.height           = self.resolution[0][0] # 학습에 사용할 4개의 스케일 중 가장 맨 앞 스케일이 모델에 입력
        self.width            = self.resolution[0][1]
        self.num_layers       = options.num_layers    # 레즈넷 버전 18 or 36 or 50, 디폴트 18
        self.weight_init      = options.weight_init   # "pretrained"
        self.pose_type        = options.pose_type     # 포즈 네트워크 타입 "posecnn", "separate_resnet", shared_resnet"

        # 1. train, valid 데이터 파일 경로 확보
        self.datapath    = self.opt.datapath
        self.filepath    = self.opt.splits + "/" + self.opt.datatype + "/{}_files.txt"
        train_filename   = readlines(self.filepath.format("train"))
        valid_filename   = readlines(self.filepath.format("val"))

        self.train_dataloader = self.definition_dataloader(
                                    train_filename, is_training = True, shuffle = True, mode = "train")
        self.valid_dataloader = self.definition_dataloader(
                                    valid_filename, is_training = False, shuffle = False, mode = "val")
        self.train_length, self.valid_length = len(self.train_dataloader), len(self.valid_dataloader)
        self.train_iteration = self.train_length * self.epoch
        self.valid_iteration = self.valid_length * self.epoch

    
        # 2. 모델과 모델 파라미터, BackPropagation과 Project3D를 로드
        self.model      = {}
        self.parameters = []
        self.definition_model()
        self.backward_project = Depth2PointCloud(self.batch_size, self.height, self.width)
        self.forward_project  = PointCloud2Pixel(self.batch_size, self.height, self.width)
        self.backward_project.to(self.device)
        self.forward_project.to(self.device)

        # 3. 모델 로스 (SSIM) 옵티마이저, 스케쥴러 로드
        self.ssim = SSIM()
        self.ssim.to(self.device)
        self.reprojection_loss  = ReprojectionLoss()
        self.smooth_loss        = SmoothLoss()
        self.reprojection_loss.to(self.device)
        self.smooth_loss.to(self.device)

        self.model_optimizer = torch.optim.Adam(
                                    self.parameters, self.learning_rate)
        self.model_scheduler = torch.optim.lr_scheduler.StepLR(
                                    self.model_optimizer, self.scheduler_step,  0.1)
        self.depth_metrics   = ["loss", "a1", "a2", "a3", "rmse", "log_rmse", "abs_rel", "sq_rel"]

        print("Pose Network type  :", self.opt.pose_type)
        print("Dataset type       :", self.opt.datatype)
        print(option_templet1.format(self.epoch, self.batch_size, self.learning_rate))
        print(option_templet2.format(self.scheduler_step, self.opt.disparity_smoothness))
        print(option_templet3.format(self.height, self.width, self.min_depth, self.max_depth))
        print(option_templet4.format(self.train_length, self.valid_length))
        print(option_templet5.format(self.train_iteration, self.valid_iteration))

    
    def definition_dataloader(self, filename, is_training, shuffle, mode): # 키티 데이터를 불러오는 함수
        if self.dataset == "kitti":
            dataset = KITTIMonoDataset(
                self.datapath, filename, is_training, self.frame_ids,
                ext = ".jpg",  scale = self.opt.scale)
        elif self.dataset == "cityscapes":
            dataset = CityscapesMonoDataset(
                self.datapath, filename, is_training, self.frame_ids, 
                mode = mode, ext = ".jpg", scale = self.opt.scale)

        dataloader = DataLoader(
            dataset, self.batch_size, shuffle, num_workers = self.opt.num_workers, drop_last = True)
        return dataloader


    def definition_model(self): # 뎁스 인코더, 디코더, 포즈 네트워크 로드하는 함수
        self.model["encoder"] = ResnetEncoder(
                                    num_layers = self.num_layers, pretrained = self.weight_init)
        self.model["decoder"] = DepthDecoder(
                                    num_ch_enc = self.model["encoder"].num_ch_enc, scales = self.scales)
        
        if self.pose_type == "posecnn":
            self.model["pose_decoder"] = PoseCNN(num_input_frames = self.num_pose_frames)
        elif self.pose_type == "shared":
            self.model["pose_decoder"] = PoseDecoder(self.model["encoder"].num_ch_enc, self.num_pose_frames)
        elif self.pose_type == "separate":
            self.model["pose_encoder"] = ResnetEncoder(self.num_layers, self.weight_init, self.num_pose_frames)
            self.model["pose_decoder"] = PoseDecoder(self.model["pose_encoder"].num_ch_enc, 
                                                    num_input_features = 1, num_frames_to_predict_for = 2)
        for key in self.model:
            self.model[key].to(self.device)
            self.parameters += list(self.model[key].parameters())


    def model_train(self): # 훈련 루프
        """
        1. epoch_train_log, epoch_valid_log 에포크마다의 로스와 메트릭을 저장함
        2. batch_train_log, batch_valid_log는 한 번 에포크가 지나면 초기화 되어야 함
        3. batch_log에 있는 값들을 평균하면 1 에포크의 데이터 샘플에 대한 평균 -> epoch_log에 저장
        """
        epoch_train_log = {key: [] for key in self.depth_metrics} # 에포크마다 로그를 기록해서 누적
        epoch_valid_log = {key: [] for key in self.depth_metrics}

        for epoch in range(self.epoch):
            batch_train_log = {key: [] for key in self.depth_metrics} # 배치 데이터의 로스와 메트릭을 누적
            batch_valid_log = {key: [] for key in self.depth_metrics}

            self.set_train()
            for batch_idx, train_inputs in tqdm(enumerate(self.train_dataloader)):
                train_outputs, train_loss = self.batch_process(train_inputs)

                self.model_optimizer.zero_grad()
                train_loss["loss"].backward()
                self.model_optimizer.step()
                
                batch_train_log = self.compute_metric(train_inputs, train_outputs, train_loss, batch_train_log)

                # if batch_idx % 250 == 0:
                #     for index, key in enumerate(self.depth_metrics):
                #         print("{}: {:0.3f}   ".format(key, batch_train_log[key][-1]), end = ' ')
            
            self.set_valid()
            for batch_idx, valid_inputs in tqdm(enumerate(self.valid_dataloader)):
                with torch.no_grad():
                    valid_outputs, valid_loss = self.batch_process(valid_inputs)

                    batch_valid_log = self.compute_metric(valid_inputs, valid_outputs, valid_loss, batch_valid_log)
            
            self.model_scheduler.step()
            for key in self.depth_metrics:
                epoch_train_log[key].append(np.mean(batch_train_log[key]))
                epoch_valid_log[key].append(np.mean(batch_valid_log[key]))

            model_print(epoch, self.depth_metrics, batch_train_log, batch_valid_log)
            self.model_save(epoch, epoch_train_log, epoch_valid_log)
        print(">>> End Game")


    # 모델 저장 함수
    def model_save(self, epoch, train_log, valid_log):
        """
        2021 07 07
        로스 함수 부분의 수식이 잘못된 것 수정
        프레임 아이디 순회하는 반복문 오류 수정
        """
        save_directory = os.path.join("./model_save", self.opt.save_name)
        train_log_directory = os.path.join(save_directory, "train_log")
        valid_log_directory = os.path.join(save_directory, "valid_log")

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
            os.makedirs(train_log_directory)
            os.makedirs(valid_log_directory)
        
        if (epoch+1) % 10 == 0: # epoch가 특정 조건을 만족시키는 조건문
            torch.save(self.model["encoder"].state_dict(), 
                save_directory + "/" + "encoder{}.pt".format(epoch+1))
            torch.save(self.model["decoder"].state_dict(), 
                save_directory + "/" + "decoder{}.pt".format(epoch+1))
            torch.save(self.model["pose_decoder"].state_dict(), 
                save_directory + "/" + "pose_decoder{}.pt".format(epoch+1))
            if self.pose_type == "separate":
                torch.save(self.model["pose_encoder"].state_dict(),
                save_directory + "/" + "pose_encoder{}.pt".format(epoch+1))
        
        if (epoch+1) == self.epoch:
            for key in train_log: # 모델의 로그 기록 저장
                train_log_path = os.path.join(train_log_directory, "{}.npy".format(key))
                valid_log_path = os.path.join(valid_log_directory, "{}.npy".format(key))
                np.save(train_log_path, train_log[key])
                np.save(valid_log_path, valid_log[key])


    def batch_process(self, inputs): # 배치 데이터마다 처리
        for key in inputs:
            inputs[key ] = inputs[key].to(self.device)

        if self.pose_type == "posecnn": # posecnn 타입의 경우는 이미지 한 장을 뎁스 네트워크에 포워드
            features = self.model["encoder"](inputs[("color_aug", 0, 0)])
            outputs  = self.model["decoder"](features)
        
        elif self.pose_type == "shared": # shared 이면 모든 프레임을 concat해서 포워드
            all_frames   = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids])
            all_features = self.model["encoder"](all_frames)
            all_features = [torch.split(feature, self.batch_size) for feature in all_features]

            features = {}
            for index, frame_id in enumerate(self.frame_ids):
                features[frame_id] = [feature[index] for feature in all_features]
            outputs = self.model["decoder"](features[0])

        elif self.pose_type == "separate": # separate 타입이면 posecnn처럼 이미지 한 장을 포워드
            features = self.model["encoder"](inputs[("color_aug", 0, 0)])
            outputs  = self.model["decoder"](features)

        """
        1. color_aug를 으로 추출한 outputs에 disparity뿐만 아니라 R, T, cam2cam 키를 추가
        2. train_inputs, train_outputs를 이용해서 [R|T]를 계산하고 camera_coords -> frame_coords으로 변환
        3. outputs에 스케일마다 인터폴레이트한 뎁스 저장, 
           스케일마다 프레임 코디네이트 저장,
           스케일마다 그리드 샘플링한 이미지 저장 (소스 이미지를 타겟 좌표 관점에서 바라본 것)
        """
        inputs, outputs = self.compute_pose(inputs, outputs, features)
        inputs, outputs = self.compute_depth(inputs, outputs)
        loss            = self.compute_loss(inputs, outputs)
        return outputs, loss


    def compute_pose(self, inputs, outputs, features): # 포즈 계산
        """
        포즈를 계산하는 함수
        프레임 아이디가 [0, -1, 1] 이면 [0, 1, 2]를 순회해서
        [-1, 0], [0, 1] 형태로 묶어, backward pose, forward pose를 추정
        framd_id < 0 -> 0이 뒤에 오게 묶고, frame_id < 0이므로 invert matrix = True
        frame_id > 0 -> 0이 앞에 오게 묶고, frame_id > 0이므로 invert matrix = False
        """
        if self.num_pose_frames == 2:
            if self.pose_type == "posecnn":
                all_frames = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids}

                for frame_id in self.frame_ids[1:]:
                    if frame_id < 0:
                        pose_inputs = torch.cat([all_frames[frame_id], all_frames[0]], dim = 1)
                    else:
                        pose_inputs = torch.cat([all_frames[0], all_frames[frame_id]], dim = 1)
                    
                    axisangle, translation = self.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))

            elif self.pose_type == "shared":
                all_features = {frame_id: features[frame_id] for frame_id in self.frame_ids}

                for frame_id in self.frame_ids[1:]:
                    if frame_id < 0:
                        pose_inputs = [all_features[frame_id], all_features[0]]
                    else:
                        pose_inputs = [all_features[0], all_features[frame_id]]

                    axisangle, translation = self.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))

            elif self.pose_type == "separate":
                all_frames = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids}

                for frame_id in self.frame_ids[1:]:
                    if frame_id < 0:
                        pose_inputs = torch.cat([all_frames[frame_id], all_frames[0]], dim = 1)
                    else:
                        pose_inputs = torch.cat([all_frames[0], all_frames[frame_id]], dim = 1)

                    pose_inputs = [self.model["pose_encoder"](pose_inputs)]
                    axisangle, translation = self.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))
        
        else:
            if self.pose_type == "posecnn":
                all_frames = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids], dim =1)
                axisangle, translation = self.model["pose_decoder"](all_frames)

            elif self.pose_type == "shared":
                all_features = [features[frame_id] for frame_id in self.frame_ids]
                axisangle, translation = self.model["pose_decoder"](all_features)

            elif self.pose_type == "separate":
                all_frames  = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids], dim = 1)
                pose_inputs = [self.model["pose_encoder"](all_frames)]
                axisangle, translation = self.model["pose_decoder"](pose_inputs)
            
            for index, frame_id in enumerate(self.frame_ids[1: ]):
                outputs[("R", frame_id, 0)]   = axisangle
                outputs[("T", frame_id, 0)]   = translation
                outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0])
        return inputs, outputs # outputs에 "R", "T", "cam2cam" 키를 가지고 나옴
        

    def compute_depth(self, inputs, outputs): # 뎁스 계산
        """
        1. 추정한 pose와 disparity로 depth 및 warping image를 계산 (source_scale = 0 -> (320, 1024)를 의미)
        2. 4개 스케일의 디스패리티 맵을 source_scale (원래 이미지 크기) 로 interpolate
        3. 원본 크기의 disparity를 얻은 후, 동일 크기의 뎁스로 변환하여 로스에 사용
        4. 키 프레임을 제외한 warping image와 target -> photometric loss
        5. 4개의 이미지 스케일과 디스패리티 스케일 -> smooth loss
        """
        source_scale = 0
        for scale in self.scales:
            # frame_id == 0인 프레임에 해당하는 여러 스케일의 disp로 원본 사이즈 뎁스를 추정
            # 어떤 scale의 disp를 depth로 변환하고 corresponding scale 키로 저장
            disparity = outputs[("disp", scale)]
            disparity = F.interpolate(
                disparity, [self.height, self.width], mode = "bilinear", align_corners = False)
            _, depth  = disparity2depth(disparity, self.min_depth, self.max_depth)
            outputs[("depth", 0, scale)] = depth # saving depth for depth eval

            for index, frame_id in enumerate(self.frame_ids[1: ]): # 0을 제외하고 [-2, -1, 1, 2] or [-1 ,1] etc ...
                """
                Ps ~ KTDK^-1Pt
                0. R, T를 이용하여 transformation 행렬 계산
                1. frame_id = 0에서 source_scale의 뎁스 정보를 이용하여 카메라 좌표 획득
                   K, transformation으로 다른 frame_id ([-2, -1, 1, 2]) 뷰들로 변환
                2. id = 0의 프레임에서 id = frame_id의 프레임으로 옮긴 좌표를 저장
                3. 그 좌표와 해당 frame_id 이미지로 grid_sample을 통한 warping,
                   warping 이미지를 output[("warped_color" ~~ )]에 저장
                """
                transformation = outputs[("c2c", frame_id, 0)]
                if self.pose_type == "posecnn":
                    axisangle      = outputs[("R", frame_id, 0)]
                    translation    = outputs[("T", frame_id, 0)]
                    inv_depth      = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    transformation = param2matrix(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                camera_coords  = self.backward_project(depth, inputs[("inv_K", source_scale)])
                frame_coords   = self.forward_project(camera_coords, inputs[("K", source_scale)], transformation)
                outputs[("warped_color", frame_id, scale)] = F.grid_sample(
                                                                inputs[("color", frame_id, source_scale)], 
                                                                frame_coords, 
                                                                padding_mode = "border", 
                                                                align_corners = False)
        return inputs, outputs # outputs에 "depth", warped_color" 키를 가지고 나옴


    def compute_loss(self, inputs, outputs): # 로스 계산
        """
        타겟 이미지, 와핑 이미지를 가지고 로스를 계산하는 함수
        1. 스케일 리스트를 돌면서 scale: 0, 1, 2, 3에 대한 뎁스 계산
        2. 해당 스케일의 디스패리티와 타겟 이미지, scale = 0의 원본 타겟 이미지 지정
        3. 프레임 아이디 리스트를 돌면서 reprojection_loss 계산 (와핑이 잘 되면 로스 0, 와핑이 잘 안되면 로스 큼)

        4. (매우 중요) self.opt.use_automasking을 켜서 identity_reprojection_loss 계산
            reprojection_loss warping한 이미지와 target 사이의 로스를 계산
            identity_reprojection_loss warping하지 않은 원본과 타겟을 계산 

            reprojection_loss < identity_reprojection_loss
            포즈를 잘 알아서 와핑시킨 이미지가 타겟 이미지와 더욱 잘 일치하다는 것의 의미?
            정적인 물체가 배경처럼 간주됨, 그러니까 카메라와 같은 속도, 방향으로 움직이는 물체 -> 로스에서 무시

            reprojection_loss > identity_reprojection_loss
            포즈를 잘 알아서 와핑시켰는데 오히려 와핑 시키지 않은 이미지와의 로스보다도 크다. 어떤 의미?
            포즈에 비해 동적인 물체 모션이 별도로 있다. 이것은 로스에 반영해야겠다.

        5. 이 둘을 하나로 합치고, 나머지 smooth loss를 계산해서 더 함
        """
        loss_dict    = {}
        total_loss   = 0
        source_scale = 0
        for scale in self.scales: # 0 1 2 3
            scale_loss     = 0
            disparity      = outputs[("disp", scale)]            # 다양한 사이즈의 스케일 (4개)
            target_scale   = inputs[("color", 0, scale)]         # 다양한 사이즈의 스케일 (4개)
            target_default = inputs[("color", 0, source_scale)] # scale = 0의 타겟 이미지
            
            # target과 warping 이미지에 대해서 reprojection_loss 계산
            reprojection_loss = []
            for frame_id in self.frame_ids[1: ]:
                prediction = outputs[("warped_color", frame_id, scale)]         # prediction: [B C 320 1024]
                reprojection_loss.append(
                    self.reprojection_loss(prediction, target_default))
            reprojection_loss = torch.cat(reprojection_loss, 1)

            if self.opt.use_ave_reprojection:
                reprojection_loss = reprojection_loss.mean(1, keepdim = True)
            else:
                reprojection_loss = reprojection_loss
            
            # if self.opt.use_automasking(True) == if not self.disable_automasking(False)
            if self.opt.use_automasking:
                identity_reprojection_loss = []
                for frame_id in self.frame_ids[1: ]:
                    source_default  = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_loss.append(
                        self.reprojection_loss(source_default, target_default))
                identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)

                if self.opt.use_ave_reprojection:
                    identity_reprojection_loss = identity_reprojection_loss.mean(1, keepdim = True)
                else:
                    identity_reprojection_loss = identity_reprojection_loss
        
            
            if self.opt.use_automasking:
                identity_reprojection_loss += 0.00001 * torch.randn(identity_reprojection_loss.shape).to(self.device)
                concat_loss = torch.cat((identity_reprojection_loss, reprojection_loss), dim = 1)
            else:
                concat_loss = reprojection_loss

            if concat_loss.shape[1] == 1:
                selected_reproejction = concat_loss
            else:
                selected_reproejction, _ = torch.min(concat_loss, dim = 1)

            # 각 스케일의 이미지와 disparity로 smooth_loss를 계산
            smooth_loss = self.smooth_loss(disp = disparity, color = target_scale) # disparity와 color로 smooth loss 계산

            scale_loss += selected_reproejction.mean() # Loss summation
            scale_loss += self.opt.disparity_smoothness * smooth_loss / (2**scale)
            total_loss += scale_loss
        
        total_loss        /= self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict


    def compute_metric(self, inputs, outputs, loss, log): # 메트릭 계산
        log["loss"].append(loss["loss"].detach().cpu().numpy())

        depth_errors = compute_depth_metric(inputs, outputs)
        for index, metric in enumerate(self.depth_metrics[1:]):
            log[metric].append(depth_errors[index].cpu().numpy())
        return log # log key = ["loss", "a1", "a2", "a3", "rmse", "log_rmse", "abs_rel", "sq_rel"]

    
    def set_train(self): # 훈련 모드
        for m in self.model.values():
            m.train()

    def set_valid(self): # 검증 모드
        for m in self.model.values():
            m.eval()


if __name__ == "__main__":
    args    = main()
    TRAINER = trainer(options = args)
    TRAINER.model_train()