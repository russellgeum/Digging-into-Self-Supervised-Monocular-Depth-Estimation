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
        self.dataset   = opt.datapath.split("/")[2]
        self.filepath  = opt.splits + "/" + opt.datatype + "/{}_files.txt"
        train_filename = readlines(self.filepath.format("train"))
        valid_filename = readlines(self.filepath.format("val"))
        train_filename = train_filename + valid_filename

        self.train_dataloader = self.load_dataloader(train_filename, True, True)
        self.valid_dataloader = self.load_dataloader(valid_filename, False, False)
        print(" ")
        print(">>> Setting dataloader")

        # 2. Set Model
        self.model = {}
        self.parameters = []
        self.load_network()
        print(">>> Setting model")

        # 3. Set Model and Optimizer
        self.loss = {}
        self.load_loss()

        self.optim = {}
        self.load_optimizer()
        print(">>> Setting loss & optimizer")


    def load_dataloader(self, filename, is_training, shuffle): # 키티 데이터를 불러오는 함수
        if self.dataset == "kitti":
            dataset = KITTIMonoDataset(
                self.opt.datapath, filename, is_training, self.opt.frame_ids, ".jpg", self.opt.height, self.opt.width, 4)
            # dataset = KITTIRAWDataset(self.opt.datapath, filename,
            #     self.opt.height, self.opt.width, self.opt.frame_ids, 4, is_training, ".jpg")
        dataloader = DataLoader(
            dataset, self.opt.batch, shuffle, num_workers = self.opt.num_workers, drop_last = True)
        return dataloader


    # 뎁스 인코더, 디코더, 포즈 네트워크, 마스크 네트워크 로드하는 함수
    def load_network(self):
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

        for key in self.model:
            self.parameters += list(self.model[key].parameters())

        self.model["inv_projection"] = Depth2PointCloud(self.opt.batch, self.opt.height, self.opt.width)
        self.model["for_projection"] = PointCloud2Pixel(self.opt.batch, self.opt.height, self.opt.width)

        for key in self.model:
            self.model[key] = self.model[key].to(self.device)


    def load_loss(self):
        self.loss["reprojection"] = ReprojectionLoss()
        self.loss["edge_aware"]   = SmoothLoss()
        self.loss.update({key: self.loss[key].to(self.device) for key in self.loss}) 

    
    def load_optimizer(self):
        self.optim["optimizer"] = torch.optim.Adam(self.parameters, self.opt.learning_rate)
        self.optim["scheduler"] = torch.optim.lr_scheduler.StepLR(self.optim["optimizer"], self.opt.scheduler_step)

    
    def set_train(self):
        for value in self.model.values():
            value.train()


    def set_valid(self):
        for value in self.model.values():
            value.eval()