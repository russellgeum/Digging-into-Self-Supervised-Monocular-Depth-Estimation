import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import dataloader

import albumentations
from albumentations import Resize
if albumentations.__version__ == "0.5.2":
    from albumentations.pytorch.transforms import ToTensor
else:
    from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import ColorJitter
from skimage.transform import resize
from model_utility import *



class KITTIStereoDataset(Dataset):
    def __init__(self, datapath, filename, is_training, height = 192, width = 640, ext = "jpg", scale = 4):
        super(KITTIStereoDataset, self).__init__()
        """
        KITTIMonoDataset for torchvision
        """
        if height % 32 != 0 or width % 32 != 0:
            raise "(H, W)는 32의 나눗셉 나머지가 0일 것, KITTI 권장 사이즈는 (320, 1024) or (192, 640)"
        self.datapath    = datapath
        self.filename    = filename
        self.is_training = is_training
        self.height      = height
        self.width       = width
        self.ext         = ext
        self.scale       = scale

        self.interp      = Image.ANTIALIAS
        self.others_map  = {"l": "r", "r": "l"}
        self.side_map    = {"2": 2, "3": 3, "l": 2, "r": 3}
        # intrinsic camera (https://github.com/nianticlabs/monodepth2)
        self.K           = np.array([[0.58,    0,    0.5,    0],
                                     [0,    1.92,    0.5,    0],
                                     [0,       0,      1,    0],
                                     [0,       0,      0,    1]], dtype = np.float32) 

        self.numpy2tensor = transforms.ToTensor()
        self.transforms   = transforms.ColorJitter.get_params(
            (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
        
        self.resize = {}
        for scale in range(self.scale):
            self.resize[scale] = transforms.Resize(
                (self.height // (2**scale), self.width // (2**scale)), interpolation = self.interp)


    def load_image(self, folder, frame_index, side, do_flip):
        image_name = "{:010d}{}".format(frame_index, self.ext)
        image_path = os.path.join(self.datapath, folder, "image_0{}/data".format(self.side_map[side]), image_name)
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                image = img.convert('RGB')
                
                if do_flip == True:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                return image


    def load_point(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.datapath, folder.split("/")[0])
        velo_filename = os.path.join(
            self.datapath, folder, "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth = point2depth(calib_path, velo_filename, self.side_map[side])
        depth = resize(depth, (375, 1242), order = 0, preserve_range = True, mode = "constant")

        if do_flip == True:
            depth = np.fliplr(depth)

        depth = np.expand_dims(depth, axis = 0)
        depth = torch.from_numpy(depth.astype(np.float32))
        return depth


    def resize_intrinsic(self, input_data):
        for scale in range(self.scale):
            K_copy       = self.K.copy()
            K_copy[0, :] = K_copy[0, :] * self.width // (2 ** scale)
            K_copy[1, :] = K_copy[1, :] * self.height // (2 ** scale)
            inv_K        = np.linalg.pinv(K_copy)

            input_data[("K", scale)]     = torch.from_numpy(K_copy)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
        return input_data


    def stereo_translation(self, input_data, side, do_flip):
        stereo_translation = np.eye(4, dtype=np.float32)
        baseline_sign      = -1 if do_flip else 1
        side_sign          = -1 if side == "l" else 1

        stereo_translation[0, 3] = side_sign * baseline_sign * 0.1
        input_data["stereo"] = torch.from_numpy(stereo_translation)
        return input_data

    
    def __getitem__(self, index):
        do_color    = self.is_training and random.random() > 0.5
        do_flip     = self.is_training and random.random() > 0.5

        batch_line  = self.filename[index].split()
        folder_name = batch_line[0]
        key_frame   = int(batch_line[1])
        side        = batch_line[2]

        input_data  = {}
        if do_color == True:
            image = self.load_image(folder_name, key_frame, side, do_flip)
            other = self.load_image(folder_name, key_frame, self.others_map[side], do_flip)
            for scale in range(self.scale):
                resize_image = self.resize[scale](image)
                resize_other = self.resize[scale](other)

                input_data.update(
                    {("color", 0, scale): self.numpy2tensor(resize_image)})
                input_data.update(
                    {("color", "s", scale): self.numpy2tensor(resize_other)})
                input_data.update(
                    {("color_aug", 0, scale): self.numpy2tensor(self.transforms(resize_image))})
                input_data.update(
                    {("color_aug", "s", scale): self.numpy2tensor(self.transforms(resize_other))})

        else:
            idnetity = (lambda x: x)
            image = self.load_image(folder_name, key_frame, side, do_flip)
            other = self.load_image(folder_name, key_frame, self.others_map[side], do_flip)
            for scale in range(self.scale):
                resize_image = self.resize[scale](image)
                resize_other = self.resize[scale](other)

                input_data.update(
                    {("color", 0, scale): self.numpy2tensor(resize_image)})
                input_data.update(
                    {("color", "s", scale): self.numpy2tensor(resize_other)})
                input_data.update(
                    {("color_aug", 0, scale): self.numpy2tensor(idnetity(resize_image))})
                input_data.update(
                    {("color_aug", "s", scale): self.numpy2tensor(idnetity(resize_other))})
        
        input_data.update(
            {("depth", 0, 0): self.load_point(folder_name, key_frame, side, do_flip)})
        input_data.update(
            {("depth", "s", 0): self.load_point(folder_name, key_frame, self.others_map[side], do_flip)})
        input_data = self.resize_intrinsic(input_data)
        input_data = self.stereo_translation(input_data, side, do_flip)
        return input_data

    def __len__(self):
        return len(self.filename)



class KITTIMonoStereoDataset(Dataset):
    def __init__(self, datapath, filename, is_training, frame_ids, height = 192, width = 640, ext = "jpg", scale = 4):
        super(KITTIMonoStereoDataset, self).__init__()
        """
        KITTIMonoDataset for torchvision
        """
        if height % 32 != 0 or width % 32 != 0:
            raise "(H, W)는 32의 나눗셉 나머지가 0일 것, KITTI 권장 사이즈는 (320, 1024) or (192, 640)"
        if "s" not in frame_ids:
            raise "'s'는 frame_ids에 포함되어야 함"

        self.datapath    = datapath
        self.filename    = filename
        self.is_training = is_training
        self.frame_ids   = frame_ids
        self.height      = height
        self.width       = width
        self.ext         = ext
        self.scale       = scale

        self.interp      = Image.ANTIALIAS
        self.others_map  = {"l": "r", "r": "l"}
        self.side_map    = {"2": 2, "3": 3, "l": 2, "r": 3}
        # intrinsic camera (https://github.com/nianticlabs/monodepth2)
        self.K           = np.array([[0.58,    0,    0.5,    0],
                                     [0,    1.92,    0.5,    0],
                                     [0,       0,      1,    0],
                                     [0,       0,      0,    1]], dtype = np.float32) 

        self.numpy2tensor = transforms.ToTensor()
        self.transforms   = transforms.ColorJitter.get_params(
            (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
        
        self.resize = {}
        for scale in range(self.scale):
            self.resize[scale] = transforms.Resize(
                (self.height // (2**scale), self.width // (2**scale)), interpolation = self.interp)


    def load_image(self, folder, frame_index, side, do_flip):
        image_name = "{:010d}{}".format(frame_index, self.ext)
        image_path = os.path.join(self.datapath, folder, "image_0{}/data".format(self.side_map[side]), image_name)
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                image = img.convert('RGB')
                
                if do_flip == True:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                return image


    def load_point(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.datapath, folder.split("/")[0])
        velo_filename = os.path.join(
            self.datapath, folder, "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth = point2depth(calib_path, velo_filename, self.side_map[side])
        depth = resize(depth, (375, 1242), order = 0, preserve_range = True, mode = "constant")

        if do_flip == True:
            depth = np.fliplr(depth)

        depth = np.expand_dims(depth, axis = 0)
        depth = torch.from_numpy(depth.astype(np.float32))
        return depth


    def resize_intrinsic(self, input_data):
        for scale in range(self.scale):
            K_copy       = self.K.copy()
            K_copy[0, :] = K_copy[0, :] * self.width // (2 ** scale)
            K_copy[1, :] = K_copy[1, :] * self.height // (2 ** scale)
            inv_K        = np.linalg.pinv(K_copy)

            input_data[("K", scale)]     = torch.from_numpy(K_copy)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
        return input_data


    def stereo_translation(self, input_data, side, do_flip):
        stereo_translation = np.eye(4, dtype = np.float32)
        baseline_sign      = -1 if do_flip else 1
        side_sign          = -1 if side == "l" else 1

        stereo_translation[0, 3] = side_sign * baseline_sign * 0.1
        input_data["stereo"] = torch.from_numpy(stereo_translation)
        return input_data


    def __getitem__(self, index):
        do_color    = self.is_training and random.random() > 0.5
        do_flip     = self.is_training and random.random() > 0.5

        batch_line  = self.filename[index].split()
        folder_name = batch_line[0]
        key_frame   = int(batch_line[1])
        side        = batch_line[2]

        input_data  = {}
        if do_color == True:
            augments = transforms.ColorJitter.get_params(
                (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
            for frame_id in self.frame_ids:
                if frame_id != "s":
                    image = self.load_image(folder_name, key_frame + frame_id, side, do_flip)
                elif frame_id == "s":
                    image = self.load_image(folder_name, key_frame, self.others_map[side], do_flip)
                
                for scale in range(self.scale):
                    resize = self.resize[scale](image)
                    input_data.update(
                        {("color", frame_id, scale): self.numpy2tensor(resize)})
                    input_data.update(
                        {("color_aug", frame_id, scale): self.numpy2tensor(augments(resize))})
        else:
            identity = (lambda x: x)
            for frame_id in self.frame_ids:
                if frame_id != "s":
                    image = self.load_image(folder_name, key_frame + frame_id, side, do_flip)
                elif frame_id == "s":
                    image = self.load_image(folder_name, key_frame, self.others_map[side], do_flip)

                for scale in range(self.scale):
                    resize = self.resize[scale](image)
                    input_data.update(
                        {("color", frame_id, scale): self.numpy2tensor(resize)})
                    input_data.update(
                        {("color_aug", frame_id, scale): self.numpy2tensor(identity(resize))})

        input_data.update({("depth", 0): self.load_point(folder_name, key_frame, side, do_flip)})
        input_data = self.resize_intrinsic(input_data)
        input_data = self.stereo_translation(input_data, side, do_flip)
        return input_data


    def __len__(self):
        return len(self.filename)