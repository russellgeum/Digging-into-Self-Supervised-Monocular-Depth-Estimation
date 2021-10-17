from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model_utility import *
from model_option import *
from model_loader import *
from model_layer import *
from model_loss import *
from model_tool import *



class trainer(object):
    def __init__(self, opt):
        # pytorch_randomness()
        self.opt     = opt
        self.device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(">>> Using device name: ", self.device)

        self.setting = setting(opt, self.device)
        self.compute = compute(opt, self.device)
        self.control = control(opt, self.device)
        print(" ")

        print("Dataset           {}".format(opt.dataset))
        print("Epoch             {}".format(opt.epoch))
        print("Batch             {}".format(opt.batch))
        print("Learning rate     {}".format(opt.learning_rate))
        print("Scheduler step    {}".format(opt.scheduler_step))
        print("Disp smoothness   {}".format(opt.disp_smoothness))

        print("Input size        ({}, {})".format(opt.height, opt.width))
        print("min-max depth     ({}, {})".format(opt.min_depth, opt.max_depth))

        print("num of iteration")
        print("   Train          {}".format(self.setting.train_dataloader.__len__()))
        print("   Valid          {}".format(self.setting.valid_dataloader.__len__()))
        print("   Total train    {}".format(self.opt.epoch * self.setting.train_dataloader.__len__()))
        print("   Total valid    {}".format(self.opt.epoch * self.setting.train_dataloader.__len__()))
        print(" ")

    
    def train(self):
        print(">>> >>> >>> Training Start")
        epoch_train = {key: [] for key in self.control.metric_name}
        epoch_valid = {key: [] for key in self.control.metric_name}

        for epoch in range(self.opt.epoch):
            batch_train = {key: [] for key in self.control.metric_name}
            batch_valid = {key: [] for key in self.control.metric_name}
    
            self.setting.set_train()
            for _, train_inputs in tqdm(enumerate(self.setting.train_dataloader)):
                train_outputs = self.batch_process(train_inputs)
                
                self.setting.optim["optimizer"].zero_grad()
                train_outputs["loss"].backward()
                print("Loss {}".format(train_outputs["loss"].item()))
                # print(self.setting.model["decoder"].convs[("dispconv", 0)].conv.weight.grad[0, :, :, :])
                self.setting.optim["optimizer"].step()

                batch_train   = self.control.metric(train_inputs, train_outputs, batch_train)

            self.setting.set_valid()
            for _, valid_inputs in tqdm(enumerate(self.setting.valid_dataloader)):
                with torch.no_grad():
                    valid_outputs = self.batch_process(valid_inputs)
                    batch_valid   = self.control.metric(valid_inputs, valid_outputs, batch_valid)

            self.setting.optim["scheduler"].step()
            for key in self.control.metric_name:
                epoch_train[key].append(np.mean(batch_train[key]))
                epoch_valid[key].append(np.mean(batch_valid[key]))

            self.control.print(epoch, batch_train, batch_valid)
            self.control.save(epoch, epoch_train, epoch_valid, self.setting)

    
    def batch_process(self, inputs):
        outputs = {}
        inputs, outputs = self.compute.forward_depth(inputs, outputs, self.setting)
        inputs, outputs = self.compute.forward_pose(inputs, outputs, self.setting)
        inputs, outputs = self.compute.image2warping(inputs, outputs, self.setting)
        outputs = self.compute.compute_loss(inputs, outputs, self.setting)
        return outputs



if __name__ == "__main__":
    trainer(options()).train()