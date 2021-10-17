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



class setting(object):
    def __init__(self, opt, device):
        self.opt    = opt
        self.device = device
        if opt.pose_frames == "all":
            self.num_pose_frames = len(opt.frame_ids)
        else:
            self.num_pose_frames = 2

        # 1. Set Dataloader
        self.filepath  = opt.splits + "/" + opt.datatype + "/{}_files.txt"
        train_filename = readlines(self.filepath.format("train"))
        valid_filename = readlines(self.filepath.format("val"))
        # train_filename = train_filename[: 120]
        # valid_filename = valid_filename[: 120]

        self.train_dataloader = self.set_loader(train_filename, True, True)
        self.valid_dataloader = self.set_loader(valid_filename, False, False)

        # 2. Set Model
        self.model = {}
        self.parameters = []
        self.set_model()

        # 3. Set Loss
        self.loss = {}
        self.set_loss()

        # 4. Set Optimizer
        self.optim = {}
        self.set_optim()


    # 키티 데이터를 불러오는 함수
    def set_loader(self, filename, is_training, shuffle):
        if self.opt.dataset == "kitti_mono":
            dataset = KITTIMonoDataset_v2(
                self.opt.datapath, filename, is_training, self.opt.frame_ids, 
                self.opt.height, self.opt.width, ".jpg", 4)
        elif self.opt.dataset == "kitti_stereo":
            dataset = KITTIMonoStereoDataset(
                self.opt.datapath, filename, is_training, self.opt.frame_ids,
                self.opt.height, self.opt.width, ".jpg", 4)
                
        dataloader = DataLoader(
            dataset, self.opt.batch, shuffle, num_workers = self.opt.num_workers, drop_last = True)
        if is_training:
            print(">>> Setting {} loader".format("Train"))
        else:
            print(">>> Setting {} loader".format("Valid"))
        return dataloader


    # 뎁스 인코더, 디코더, 포즈 네트워크, 마스크 네트워크 로드하는 함수
    def set_model(self):
        self.model["encoder"] = ResnetEncoder(
            num_layers = self.opt.num_layers, pretrained = self.opt.weight_init)
        self.model["decoder"] = DepthDecoder(
            num_ch_enc = self.model["encoder"].num_ch_enc, scales = self.opt.scales)
        
        if self.opt.pose_type == "posecnn":
            self.model["pose_decoder"] = PoseCNN(num_input_frames = self.num_pose_frames)

        elif self.opt.pose_type == "shared":
            self.model["pose_decoder"] = PoseDecoder(self.model["encoder"].num_ch_enc, self.num_pose_frames)
            
        elif self.opt.pose_type == "separate":
            self.model["pose_encoder"] = ResnetEncoder(
                self.opt.num_layers, self.opt.weight_init, self.num_pose_frames)
            self.model["pose_decoder"] = PoseDecoder(
                self.model["pose_encoder"].num_ch_enc, num_input_features = 1, num_frames_to_predict_for = 2)

        self.inv_projection = {}
        self.for_projection = {}
        self.inv_projection[0] = Depth2PointCloud(self.opt.batch, self.opt.height, self.opt.width).to(self.device)
        self.for_projection[0] = PointCloud2Pixel(self.opt.batch, self.opt.height, self.opt.width).to(self.device)
        
        for key in self.model:
            self.model[key] = self.model[key].to(self.device)
            self.parameters += list(self.model[key].parameters())
        print(">>> Setting model")


    def set_loss(self):
        self.loss["reprojection"] = ReprojectionLoss()
        self.loss["edge_aware"]   = SmoothLoss()
        self.loss.update({key: self.loss[key].to(self.device) for key in self.loss}) 
        print(">>> Setting loss")

    
    def set_optim(self):
        self.optim["optimizer"] = torch.optim.Adam(self.parameters, self.opt.learning_rate)
        self.optim["scheduler"] = torch.optim.lr_scheduler.StepLR(self.optim["optimizer"], self.opt.scheduler_step)
        print(">>> Setting optim")

    
    def set_train(self):
        for value in self.model.values():
            value.train()


    def set_valid(self):
        for value in self.model.values():
            value.eval()