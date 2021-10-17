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



class KITTIMonoDataset(Dataset):
    def __init__(self, datapath, filename, is_training, frame_ids, height, width, ext = "jpg", scale = 4):
        super(KITTIMonoDataset, self).__init__()
        """
        Args:
            datapath:    "./dataset/kitti"
            filename:    splits file of KITTI
            is_training: True or False
            frame_ids:   relative position list of key frame
            ext:         ".jpg" or ".png"
            height:      height of image
            width:       width of image
            scale:       pyramid scale of image
        
        interpolation 1은 쓰지 말 것, 성능이 나오지 않음, 0 아니면 3으로 실험
        (albumentation Resize interpolation option)
        0 : cv2.INTER_NEAREST, 
        1 : cv2.INTER_LINEAR, 
        2 : cv2.INTER_CUBIC, 
        3 : cv2.INTER_AREA, 
        4 : cv2.INTER_LANCZOS4. Default: cv2.INTER_AREA.
        """
        if height % 32 != 0 or width % 32 != 0:
            raise "(H, W)는 32의 나눗셉 나머지가 0일 것, KITTI 권장 사이즈는 (320, 1024) or (192, 640)"
            
        self.datapath     = datapath
        self.filename     = filename
        self.is_training  = is_training
        self.frame_ids    = frame_ids
        self.height       = height
        self.width        = width
        self.ext          = ext
        self.scale        = scale
        
        self.inter        = cv2.INTER_AREA # Image.ANTIALIAS와 동등한가?
        self.side_map     = {"2": 2, "3": 3, "l": 2, "r": 3}
        # intrinsic camera (https://github.com/nianticlabs/monodepth2)
        self.K            = np.array([[0.58,    0,    0.5,    0],
                                      [0,    1.92,    0.5,    0],
                                      [0,       0,      1,    0],
                                      [0,       0,      0,    1]], dtype = np.float32) 

        self.scales        = list(range(scale))
        self.origin_scale  = (375, 1242)
        self.resize_scale  = (self.height, self.width) # 권장 스케일 (320, 1024), (192, 640)
        self.scale_list    = [(self.height//(2**i), self.width//(2**i)) for i in self.scales]

        self.resize        = {}
        for scale, (height, width) in enumerate(self.scale_list):
            self.resize[scale] = Resize(height = int(height), width  = int(width), interpolation = self.inter)
        # depth_resize는 interpolation = 0으로 설정
        self.depth_resize   = Resize(height = self.origin_scale[0], width = self.origin_scale[1], interpolation = 0)

        self.augment_key    = "image"
        self.brightness     = (0.8, 1.2)
        self.contrast       = (0.8, 1.2)
        self.saturation     = (0.8, 1.2)
        self.hue            = (-0.1, 0.1)
        self.HorizontalFlip = HorizontalFlip(p = 1.0)
        self.ColorJitter    = ColorJitter(
            brightness = self.brightness, contrast = self.contrast, saturation = self.saturation, hue = self.hue, p = 1.0)
        if albumentations.__version__ == "0.5.2":
            self.image2tensor = ToTensor()
        else:
            self.image2tensor = ToTensorV2()
            
        print(">>>  KITTI scaling table")
        print(">>>  Interpolation     :  {0}".format(self.inter))
        print(">>>  Is training???    :  {0}".format(self.is_training))


    def flip_image(self, numpy_image):
        numpy_image = self.HorizontalFlip(image = numpy_image)
        return numpy_image[self.augment_key]

    def resize_image(self, scale, numpy_image):
        numpy_image = self.resize[scale](image = numpy_image)
        return numpy_image[self.augment_key]

    def recolor_image(self, numpy_image):
        numpy_image = self.ColorJitter(image = numpy_image)
        return numpy_image[self.augment_key]

    def numpy2tensor(self, numpy_image):
        if albumentations.__version__ == "0.5.2":
            tensor_image = self.image2tensor(image = numpy_image)
        else:
            tensor_image = self.image2tensor(image = numpy_image)
            tensor_image[self.augment_key] = tensor_image[self.augment_key] / 255.0
        return tensor_image[self.augment_key]


    def get_image_path(self, folder_name, frame_index, side):
        image_name = f"{frame_index:010d}" + self.ext # key_frame is int ex) 1 or 10 or 3 ... ..
        image_path = os.path.join(
            self.datapath, folder_name , "image_0{}/data/{}".format(self.side_map[side], image_name))
        return image_path

    def get_point_path(self, folder_name, key_frame):
        calib_path = os.path.join(self.datapath, folder_name.split("/")[0])
        point_name = "velodyne_points/data/{:010d}.bin".format(key_frame)
        point_path = os.path.join(self.datapath, folder_name, point_name)
        return calib_path, point_path

    def load_image(self, image_path, do_flip): # 이미지를 로드, 나중에 PIL로 고치기
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                image_instance = img.convert('RGB')
                numpy_image    = np.array(image_instance)

                if do_flip == True:
                    numpy_image = self.flip_image(numpy_image)
                return numpy_image

    def load_point(self, calib_path, point_path, side, do_flip):
        """
        키 프레임의 포인트 클라우드를 불러오고, 원본 스케일로 리사이즈하여 input_data 뎁스 키에 저장
        Args:
            input_data: 데이터를 담을 딕셔너리
            index:      배치로 들어갈 데이터의 인덱스
            do_flip:    뒤집을꺼면 미리 뒤집자 (is_training 여부에서 미리 결정)
        이슈)
        문제: 계속해서 결과물의 시각적 느낌과는 다르게, 뎁스 메트릭이 정확하게 측정 안되는 문제 발생
        원인: 포인트클라우드에서 추출한 뎁스 맵을 전처리할 때 잘못 전처리 된 듯함 
        --> 향후 어떤 차이가 있는지 잘 파악할 것
        해결 1
        skimage.transform.resize 함수로 해결 (monodepth2의 방식을 그대로 따름)
        order = 0, presevr_range = True, mode = "constant" 인자가 무엇인지 이해할 것
        
        해결 2
        Albumentations.Resize 메서드를 그대로 사용하되, interpolation = 0으로 둘 것
        
        해결 3
        포인트 클라우드 좌표가 몇 번 카메라로 매핑될지 결정해야함, 기본 값은 cam = 2이였는데
        3번 카메라를 쓰면 3번 카메라 좌표로 변환되어야 하지만 메트릭이 정확해지지 않음
        """
        depth = point2depth(calib_path = calib_path, point_path = point_path, cam = self.side_map[side])
        depth = resize(depth, self.origin_scale, order = 0, preserve_range = True, mode = "constant")
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1])).astype(np.float32)
        # depth = self.depth_resize(image = depth)
        # depth = np.reshape(depth["image"], (1, depth["image"].shape[0], depth["image"].shape[1]))
        
        if do_flip == True:
            depth = self.flip_image(depth)
        return depth

    def preprocessing_image(self, input_data, folder_name, key_frame, side, do_flip):
        """
        key_frame는 키프레임 (시퀀스의 중앙에 있을수도, 맨 뒤에 있을수도 있음)
        frame_ids가 중요한데 key_frame (키 프레임) 기준으로 상대적인 위치를 나타냄
        ex)
        key_frame = 123, frame_ids = [-1, 0, 1]
        for index in frame_ids:
            outputs = load_image(index + key_frame)
        """
        for frame_id in self.frame_ids:
            image_path  = self.get_image_path(folder_name, frame_id + key_frame, side)
            image_array = self.load_image(image_path, do_flip)
            input_data.update({("color", frame_id, scale): self.resize_image(scale, image_array) for scale in self.scales})
        return input_data

    def preprocessing_point(self, input_data, folder_name, key_frame, side, do_flip):
        """
        1. 캘리브레이션, 포인트 클라우드 파일 로드
        2. 포인트 클라우드 -> 뎁스 이미지로 변환한 데이터를 로드
        3. input_data 딕셔너리에 키 프레임의 뎁스 이미지 데이터 저장
        """
        calib_path, point_path   = self.get_point_path(folder_name, key_frame)
        depth                    = self.load_point(calib_path, point_path, side, do_flip)
        input_data[("depth", 0)] = torch.from_numpy(depth)
        return input_data

    def preprocessing_intrinsic(self, input_data):
        """
        1. 원본 intrinsic을 사용할 경우, "스케일링 크기 / 원본 크기" 비율을 곱해서 intrinsic을 줄여줌
        2. monodepth2의 intrinsic을 사용할 경우, 스케일링 크기만 곱해서 intrinsic을 늘려줌
        """
        for scale in self.scales:
            K_copy       = self.K.copy()
            K_copy[0, :] = K_copy[0, :] * self.scale_list[scale][1]
            K_copy[1, :] = K_copy[1, :] * self.scale_list[scale][0]
            inv_K        = np.linalg.pinv(K_copy)

            input_data[("K", scale)]     = torch.from_numpy(K_copy)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
        return input_data


    def __getitem__(self, index):
        """
        returns 
            ("color", <frame_id>, <scale>)             for raw color images,
            ("color_aug", <frame_id>, <scale>)         for aug color images,
            ("depth", 0) and ("K", 0) and ("inv_K", 0) for depth, intrinsic of key frame
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        do_flip     = self.is_training and random.random() > 0.5
        do_auge     = self.is_training and random.random() > 0.5
        
        batch_line  = self.filename[index].split()
        folder_name = batch_line[0]
        key_frame   = int(batch_line[1])
        side        = batch_line[2]
    
        # input_data 딕셔너리를 지정하고, folder_name, key_frame, side 여부 입력
        # 이미지 로드하고, 넘파이 타입에서 이미지 전처리 (flip -> resize -> recolor)
        input_data = {}
        input_data = self.preprocessing_image(input_data, folder_name, key_frame, side, do_flip)      
        if do_auge:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    self.recolor_image(input_data[("color", frame_id, scale)]) for scale in self.scales})
        else:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    input_data[("color", frame_id, scale)] for scale in self.scales})
        # input_data의 모든 이미지를 텐서 타입으로 변환
        input_data.update({key: self.numpy2tensor(input_data[key]) for key in input_data})

        # 1. 원본 이미지를 스케일링한 비율만큼 K, inv_K도 동일하게 스케일링
        # 2. Point Cloud의 뎁스 데이터를 로드
        input_data = self.preprocessing_point(input_data, folder_name, key_frame, side, do_flip)
        input_data = self.preprocessing_intrinsic(input_data)
        return input_data

    def __len__(self):
        return len(self.filename)



class KITTIMonoDataset_v2(Dataset):
    def __init__(self, datapath, filename, is_training, frame_ids, height = 192, width = 640, ext = "jpg", scale = 4):
        super(KITTIMonoDataset_v2, self).__init__()
        """
        KITTIMonoDataset for torchvision
        """
        if height % 32 != 0 or width % 32 != 0:
            raise "(H, W)는 32의 나눗셉 나머지가 0일 것, KITTI 권장 사이즈는 (320, 1024) or (192, 640)"
        self.datapath    = datapath
        self.filename    = filename
        self.is_training = is_training
        self.frame_ids   = frame_ids
        self.height      = height
        self.width       = width
        self.ext         = ext
        self.scale       = scale

        self.interp      = Image.ANTIALIAS
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
            K_copy[1, :] = K_copy[1, :] * self.width // (2 ** scale)
            inv_K        = np.linalg.pinv(K_copy)

            input_data[("K", scale)]     = torch.from_numpy(K_copy)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
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
            for frame_id in self.frame_ids:
                original_image = self.load_image(folder_name, key_frame + frame_id, side, do_flip)

                for scale in range(self.scale):
                    resize_image = self.resize[scale](original_image)
                    input_data.update(
                        {("color", frame_id, scale): self.numpy2tensor(resize_image)})
                    input_data.update(
                        {("color_aug", frame_id, scale): self.numpy2tensor(self.transforms(resize_image))})

        else: # do_color == False
            identity = (lambda x: x)
            for frame_id in self.frame_ids:
                original_image = self.load_image(folder_name, key_frame + frame_id, side, do_flip)

                for scale in range(self.scale):
                    resize_image = self.resize[scale](original_image)
                    input_data.update(
                        {("color", frame_id, scale): self.numpy2tensor(resize_image)})
                    input_data.update(
                        {("color_aug", frame_id, scale): self.numpy2tensor(identity(resize_image))})

        input_data.update(
            {("depth", 0): self.load_point(folder_name, key_frame, side, do_flip)})
        input_data = self.resize_intrinsic(input_data)
        return input_data


    def __len__(self):
        return len(self.filename)