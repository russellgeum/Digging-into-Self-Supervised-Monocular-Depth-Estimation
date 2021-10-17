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



class control(object):
    def __init__(self, opt, device):
        """
        control
            def print
            def save
            def load
            def metric
        """
        self.opt    = opt
        self.device = device
        self.metric_name = ["loss", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


    def metric(self, inputs, outputs, metric_dict): # 배치 데이터마다 메트릭 계산
        metric_dict["loss"].append(outputs["loss"].detach().cpu().numpy())

        depth_errors = compute_depth_metric(inputs, outputs, "torch")
        for index, metric in enumerate(self.metric_name[1:]):
            metric_dict[metric].append(depth_errors[index].cpu().numpy())
        return metric_dict


    def print(self, epoch, train_log, valid_log):
        print("EPOCH   {0}".format(epoch+1))
        print("Train Log", end = " ")
        for key in self.metric_name:
            print("  {} {:0.3f}".format(key, np.mean(train_log[key])), end = " ")
        print(" ")
        print("Valid Log", end = " " )
        for key in self.metric_name:
            print("  {} {:0.3f}".format(key, np.mean(valid_log[key])), end = " ")
        print(" ")


    def save(self, epoch, train_log, valid_log, setting):
        save_directory = os.path.join("./model_save", self.opt.save)
        loss_directory = os.path.join(save_directory, "loss")

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        if not os.path.isdir(loss_directory):
            os.makedirs(loss_directory)
        
        if (epoch+1) % 2 == 0: # epoch가 특정 조건을 만족시키는 조건문, 뎁스 인코더, 디코더 모델 저장
            for key in setting.model:
                torch.save(setting.model[key].state_dict(),
                    os.path.join(save_directory, key + str(epoch+1) + ".pt"))
        
        if (epoch+1) == self.opt.epoch:
            for key in setting.model:
                torch.save(setting.model[key].state_dict(),
                    os.path.join(save_directory, key + str(epoch+1) + ".pt"))

            for key in self.metric_name: # 모델의 로그 기록 저장
                np.save(os.path.join(loss_directory, key + ".npy"), train_log[key])
                np.save(os.path.join(loss_directory, key + ".npy"), valid_log[key])