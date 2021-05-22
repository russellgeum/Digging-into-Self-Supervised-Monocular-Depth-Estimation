from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import cv2
# from PIL import Image

import torch
from torch.utils.data import Dataset

from albumentations import Resize
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import ColorJitter

from model_utility import *



class KITTIDataset(Dataset):
    def __init__(self, dictionary, is_training = False, scale = 1):
        """
        Args
            dictionary: dictionary
            is_training: bool
            scale: int

        dictionary = {"image" : 
                      "point" : 
                      "v2c"   : 
                      "c2c"   :}
        1) point 키가 있다면 "point", "v2c", "c2c" 키들의 값을 모두 사용
        2) point 키가 없다면 "image" 키의 값들만 사용
        """
        self.dictionary     = dictionary
        self.image_key      = "image"
        self.point_key      = "point"
        self.velo2cam_key   = "v2c"
        self.cam2cam_key    = "c2c"
        self.image_sequence = dictionary[self.image_key]
        for dict_key in dictionary:
            if "point" == dict_key:
                self.point_sequence = dictionary[self.point_key]
                self.v2c_sequence   = dictionary[self.velo2cam_key]
                self.c2c_seqeunce   = dictionary[self.cam2cam_key]
                break

        """
        self.image_sequence[0] = (이미지1 경로, 이미지2, 경로, 이미지3 경로)
        self.image_Seqeunce[1] = (이미지2 경로, 이미지3 경로, 이미지4 경로)
        시퀀스 수에 따른 프레임 아이디와 프레임 인덱스 매칭을 메뉴얼리하게 설정
        시퀀스 2이면 인덱스 1번 프레임이 키 프레임
        시퀀스 3이면 인덱스 1번 프레임이 키 프레임
        시퀀스 5이면 인덱스 2번 프레임이 키 프레임
        시퀀스 7이면 인덱스 3번 프레임이 키 프레임
        시퀀스 2n+1이면 인덱스 n번 프레임이 키 프레임
        """
        self.num_sequence   = len(self.image_sequence[0]) # 2 3 5 (7)
        if self.num_sequence == 2:
            self.frame_idx = [1, 0]
            self.frame_ids = [0, -1]
            self.depth_idx = 1
            # self.frame_idx = [1, 0]
            # self.frame_ids = [0, -1]
            # self.depth_idx = 1
        elif self.num_sequence == 3:
            self.frame_idx = [1, 0, 2]
            self.frame_ids = [0, -1, 1]
            self.depth_idx = 1
            # self.frame_idx = [2, 0, 1]
            # self.frame_ids = [0, -2, -1]
            # self.depth_idx = 2
        elif self.num_sequence == 5:
            self.frame_idx = [2, 0, 1, 3, 4]
            self.frame_ids = [0, -2, -1, 1, 2]
            self.depth_idx = 2
            # self.frame_idx = [4, 0, 1, 2, 3]
            # self.frame_ids = [0, -4, -3, -2, -1]
            # self.depth_idx = 4

        self.K             = np.array([[0.58, 0, 0.5, 0],
                                       [0, 1.92, 0.5, 0],
                                       [0,    0,   1, 0],
                                       [0,    0,   0, 1]], dtype = np.float32) # intrinsic camera (https://github.com/nianticlabs/monodepth2)

       # albumentations 라이브러리를 사용하기 위한 옵션
        self.is_training   = is_training
        self.augment_key   = "image"
        
        self.scale          = scale
        self.original_scale = (375, 1245)
        self.default_scale  = (320, 1024)
        self.resize         = {}
        self.num_scales     = list(range(4)) #  [0, 1, 2, 3]
        self.scale_size     = [
                            (int(self.default_scale[0]/2**(self.scale+i)), 
                             int(self.default_scale[1]/2**(self.scale+i))) \
                             for i in range(len(range(4)))]
        """
        데이터로더 프로세스 플로우
        1. 좌우로 뒤집을지 말지 결정하는 boolean do_flip과 함께 이미지를 로드 (원본 이미지)
           키티 데이터 원본 크기는 (375, 1245)
        2. 원본 스케일 이미지를 0부터 3까지 해당하는 사이즈로 리사이즈
           ("color", <frame_ids>, <scale> != -1) 키 형태로 저장
        3. is_training 모드이면 do_auge를 주고, 각 스케일 이미지마다 augmentation을 적용
           ("color_aug", <frame_ids>, <scale> != -1) 키 형태로 저장
        4. GT로 사용하는 Point2Depth 이미지는 원본 스케일로 저장
           그리고 원본 스케일에 맞게 세팅된 K는
           scale = 0이면 (2**0)로 나눔
           scale = 1이면 (2**1)로 나눔
           scale = 2이면 (2**2)로 나눔
        5. 마지막으로 input_data의 모든 키 값들을 numpy2tensor 변환
        """
        for scale, (height, width) in zip(self.num_scales, self.scale_size):
            self.resize[scale] = Resize(height = int(height), width  = int(width), interpolation = 1)

        self.brightness     = (0.8, 1.2)
        self.contrast       = (0.8, 1.2)
        self.saturation     = (0.8, 1.2)
        self.hue            = (-0.1, 0.1)
        self.HorizontalFlip = HorizontalFlip(p = 1.0)
        self.ColorJitter    = ColorJitter(
                                brightness = self.brightness,
                                contrast = self.contrast,
                                saturation = self.saturation,
                                hue = self.hue,
                                p = 1.0)
        self.image2tensor   = ToTensor()
        self.depthresize    = Resize(height = self.original_scale[0], width = self.original_scale[1], interpolation = 1)



    def load_image(self, file_path, do_flip): # 이미지를 로드, 나중에 PIL로 고치기
        opencv_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        numpy_image  = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        if do_flip == True:
            numpy_image = self.flip_image(numpy_image)
        return numpy_image

    def resize_image(self, scale_factor, numpy_image):
        numpy_image = self.resize[scale_factor](image = numpy_image)
        return numpy_image[self.augment_key]

    def flip_image(self, numpy_image):
        numpy_image = self.HorizontalFlip(image = numpy_image)
        return numpy_image[self.augment_key]

    def recolor_image(self, numpy_image):
        numpy_image = self.ColorJitter(image = numpy_image)
        return numpy_image[self.augment_key]

    def numpy2tensor(self, numpy_image):
        tensor_image = self.image2tensor(image = numpy_image)
        return tensor_image[self.augment_key]



    def preprocessing_image(self, input_data, index, do_flip):
        """
        시퀀스 이미지를 불러오고 0, 1, 2, 3 스케일로 리사이즈 후, 
        ("name", <frame_id>, <scale>) 형식으로 딕셔너리 저장

        Args:
            input_data: 데이터를 담을 딕셔너리
            index:      배치로 들어갈 데이터의 인덱스
            do_flip:    뒤집을꺼면 미리 뒤집자 (is_training에서 발동)
        
        예시)
        in zip ([0, -2, -1, 1, 2], [2, 0, 1, 3, 4])로, 프레임 아이디와 프레임 인덱스를 동시에 순회한다.
        해당 프레임 인덱스의 이미지를 로드하여 매핑되는 프레임 아이디의 원본 스케일 키로 저장한다.
        ex) 프레임 인덱스 [0, 1, 2, 3, 4] 중 2번 프레임이 0번 아이디로 키프레임에 해당
            프레임 인덱스 0번, 1번은 (2, 0) 기준으로 (1, -1), (0, -2)에 해당
            프레임 인덱스 3번, 4번은 (2, 0) 기준으로 (3, 1), (4, 2)에 해당
        """
        for frame_id, seq_index in zip(self.frame_ids, self.frame_idx):
            image = self.load_image(self.image_sequence[index][seq_index], do_flip)
            input_data[("color", frame_id, 0)] = self.resize_image(0, image)
            input_data[("color", frame_id, 1)] = self.resize_image(1, image)
            input_data[("color", frame_id, 2)] = self.resize_image(2, image)
            input_data[("color", frame_id, 3)] = self.resize_image(3, image)
        return input_data


    def preprocessing_depth(self, input_data, index, do_flip):
        """
        키 프레임의 포인트 클라우드를 불러오고, 
        원본 스케일로 리사이즈하여 input_data 뎁스 키에 저장
        Args:
            input_data: 데이터를 담을 딕셔너리
            index:      배치로 들어갈 데이터의 인덱스
            do_flip:    뒤집을꺼면 미리 뒤집자 (is_training 여부에서 미리 결정)
        
        과정)
        self.v2c_sequence[batch_index]               ./dataset/calib_path/2011_09_26/calib_velo_to_cam.txt
        self.c2c_seqeunce[batch_index]               ./dataset/calib_path/2011_09_26/calib_cam_to_cam.txt
        self.point_seqeunce[batch_index][seq_index]  ./dataset/raw_data/train/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin
        """
        depth = Point2Depth(
                    self.v2c_sequence[index], self.c2c_seqeunce[index], 
                    self.point_sequence[index][self.depth_idx])
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
        if do_flip == True:
            depth = self.flip_image(depth)

        depth = self.depthresize(image = depth)
        input_data[("depth", 0)] = depth["image"]
        return input_data


    def scaling_K(self, input_data):
        """
        만약 (320, 1024) 스케일부터 사용한다면, self.scale = 0이고 self.default_scale에서 0으로 나눔
        만약 (160, 512) 스케일부터 사용한다면 self.scale = 1이고 self.default_scale에서 2로 나눔
        """
        for scale in self.num_scales:
            K       = self.K.copy()
            K[0, :] *= self.default_scale[0] // (2 ** (scale+self.scale)) 
            K[1, :] *= self.default_scale[1] // (2 ** (scale+self.scale))
            inv_K   = np.linalg.pinv(K)
            input_data[("K", scale)]     = torch.from_numpy(K)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
        return input_data


    def __len__(self):
        return len(self.image_sequence)
    
    
    def __getitem__(self, index):
        # self.is_training이 True이고 0.5 chance로 random.random()이 더 크면 auge, flip을 수행
        do_flip        = self.is_training and random.random() > 0.5
        do_auge        = self.is_training and random.random() > 0.5

        # 데이터를 담아서 리턴할 배치 딕셔너리
        input_data     = {}
        input_data     = self.preprocessing_image(input_data, index, do_flip)
        input_data     = self.preprocessing_depth(input_data, index, do_flip)

        # 원본 이미지를 스케일링한 것과 동일하게 K, inv_K 계산해서 동일하게 스케일링
        input_data     = self.scaling_K(input_data)

        """
        self.frame_ids: [0, -1] or [0, -1, 1] or [0, -2, -1, 1, 2]
        프레임 아이디 (즉, 시퀀스마다) 마다
        서로 다른 스케일을 동시에 recolor augments를 하여 "colo_aug" 키 값으로 저장
        """
        if do_auge:
            for frame_id in self.frame_ids:
                input_data[("color_aug", frame_id, 0)] = self.recolor_image(input_data[("color", frame_id, 0)])
                input_data[("color_aug", frame_id, 1)] = self.recolor_image(input_data[("color", frame_id, 1)])
                input_data[("color_aug", frame_id, 2)] = self.recolor_image(input_data[("color", frame_id, 2)])
                input_data[("color_aug", frame_id, 3)] = self.recolor_image(input_data[("color", frame_id, 3)])
        else:
            for frame_id in self.frame_ids:
                input_data[("color_aug", frame_id, 0)] = input_data[("color", frame_id, 0)]
                input_data[("color_aug", frame_id, 1)] = input_data[("color", frame_id, 1)]
                input_data[("color_aug", frame_id, 2)] = input_data[("color", frame_id, 2)]
                input_data[("color_aug", frame_id, 3)] = input_data[("color", frame_id, 3)]


        """
        마지막 단계, 
        input_data에 들어있는 모든 딕셔너리의 값 중 텐서 타입이 아닌 것은 토치로 전환
        """
        for key in input_data:
            if ("color" in key) or ("color_aug" in key) or ("depth" in key):
                input_data[key] = self.numpy2tensor(input_data[key])
        return input_data