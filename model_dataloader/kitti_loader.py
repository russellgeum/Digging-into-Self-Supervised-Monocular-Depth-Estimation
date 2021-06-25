import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import skimage.transform
from albumentations import Resize
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import ColorJitter

from model_utility import *



class KITTIDataset(Dataset):
    def __init__(self, 
                dictionary: list, frame_idx: list, frame_ids: list, key_index: int, 
                is_training = False, scale = 1):
        """
        Generating monocular sequence of KITTI Raw Dataset
        Args
            dictionary: 데이터 딕셔너리
            is_training: bool
            frame_ids: 프레임 아이디
            keyframe_idx: 키 프레임 인덱스
            is_training: training 데이터 유/무
            scale: 데이터 스케일

        dictionary = {"image" : 
                      "point" : 
                      "v2c"   : 
                      "c2c"   :}
        1) point 키가 있다면 "point", "v2c", "c2c" 키들의 값을 모두 사용
        2) point 키가 없다면 "image" 키의 값들만 사용
        """
        if scale > 3:
            raise "scale must be 0 or 1 or 2 or 3"
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

        # self.image_sequence[index] = (이미지1 경로, 이미지2 경로, 이미지3 경로)
        self.num_sequence = len(self.image_sequence[0]) # 프레임의 장수
        self.frame_idx    = frame_idx
        self.frame_ids    = frame_ids
        self.key_index    = key_index

        # intrinsic camera (https://github.com/nianticlabs/monodepth2)
        self.K             = np.array([[0.58, 0, 0.5, 0],
                                       [0, 1.92, 0.5, 0],
                                       [0,    0,   1, 0],
                                       [0,    0,   0, 1]], dtype = np.float32)
        # albumentations 라이브러리를 사용하기 위한 옵션
        self.is_training    = is_training
        self.resize         = {}
        self.num_scales     = list(range(4)) # [0, 1, 2, 3]
        self.original_scale = (375, 1242)
        self.default_scale  = [(320, 1024), (192, 640), (96, 320), (48, 160)] # corresponding as scale: 0, 1, 2, 3
        self.scale_list = [(self.default_scale[scale][0]//(2**i), 
                            self.default_scale[scale][1]//(2**i)) for i in self.num_scales]
        print("-- KITTI scaling table")
        print("Scale factor is {0}".format(scale))
        print("Default 0 scale     :  {0} {1}".format(self.default_scale[0][0], self.default_scale[0][1]))
        print("Default 1 scale     :  {0} {1}".format(int(self.default_scale[1][0]), int(self.default_scale[1][1])))
        print("Default 2 scale     :  {0} {1}".format(int(self.default_scale[2][0]), int(self.default_scale[1][1])))
        print("Default 3 scale     :  {0} {1}".format(int(self.default_scale[3][0]), int(self.default_scale[1][1])))
        print("Resolution List (from scale {0}) : {1}".format(scale, self.scale_list))

        """
        데이터로더 프로세스 플로우
        1. 좌우로 뒤집을지 말지 결정하는 boolean do_flip과 함께 이미지를 로드 (원본 이미지)
           키티 데이터 원본 크기는 (375, 1245)
        2. 원본 스케일 이미지를 원하는 스케일로 바꾸고, 그 스케일부터 2배율로 줄어드는 리스케일 [0, 1, 2, 3]
           ("color", <frame_ids>, <scale>) 키 형태로 저장
        3. is_training 모드이면 do_auge를 주고, 각 스케일 이미지마다 augmentation을 적용
           ("color_aug", <frame_ids>, <scale>) 키 형태로 저장
        4. GT로 사용하는 Point2Depth 이미지는 원본 스케일로 저장
           그리고 원본 스케일에 맞게 세팅된 K는 monodepth2의 intrinsic parameter를 따름
        5. 마지막으로 input_data의 모든 키 값들을 numpy2tensor 변환
        """
        for scale, (height, width) in zip(self.num_scales, self.scale_list):
            self.resize[scale] = Resize(height = int(height), width  = int(width), interpolation = 1)
        self.depth_resize   = Resize(height = self.original_scale[0], width = self.original_scale[1], interpolation = 0)
        self.augment_key    = "image"
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


    def load_image(self, file_path, do_flip): # 이미지를 로드, 나중에 PIL로 고치기
        # numpy_image = cv2.imread(file_path)
        image = Image.open(file_path)
        numpy_image = np.array(image)
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
        프레임 5장은 인덱스 순서가 [0, 1, 2, 3, 4]이다. 이것을 아이디 [-2, -1, 0, 1, 2]로 매핑한다.
        해당 프레임 인덱스의 이미지를 로드하여 매핑하는 프레임 아이디로 저장한다.
        """
        for frame_id, frame_ind in zip(self.frame_ids, self.frame_idx): # 배치 데이터마다 프레임 인덱스에 해당하는 이미지를 4개의 스케일로 변환
            image = self.load_image(self.image_sequence[index][frame_ind], do_flip)
            input_data.update({("color", frame_id, scale): self.resize_image(scale, image) for scale in self.num_scales})
        return input_data


    def preprocessing_depth(self, input_data, index, do_flip):
        """
        Args:
            input_data: 데이터를 담을 딕셔너리
            index:      배치로 들어갈 데이터의 인덱스
            do_flip:    뒤집을꺼면 미리 뒤집자 (is_training 여부에서 미리 결정)

        키 프레임의 포인트 클라우드를 불러오고, 원본 스케일로 리사이즈하여 input_data 뎁스 키에 저장

        이슈)
        문제: 계속해서 결과물의 시각적 느낌과는 다르게, 뎁스 메트릭이 정확하게 측정 안되는 문제 발생
        원인: 포인트클라우드에서 추출한 뎁스 맵을 전처리할 때 잘못 전처리 된 듯함 --> 향후 어떤 차이가 있는지 잘 파악할 것
        해결: skimage.transform.resize 함수로 해결 order = 0, presevr_range = True, mode = "constant" 인자가 무엇인지 이해할 것
              또 대체할 수 있는 함수가 무엇인지 찾아볼 것
        또 다른 해결
        Albumentations.Resize 메서드를 그대로 사용하되, interpolation = 0으로 둘 것
        """
        depth = Point2Depth(self.v2c_sequence[index], self.c2c_seqeunce[index],self.point_sequence[index][self.key_index])
        # depth = self.depth_resize(image = depth)
        # depth = np.reshape(depth["image"], (depth["image"].shape[0], depth["image"].shape[1], 1))

        depth = skimage.transform.resize(depth, (1242, 375)[::-1], order=0, preserve_range=True, mode='constant')
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
        if do_flip == True:
            depth = self.flip_image(depth)
        input_data[("depth", 0)] = depth
        return input_data


    def preprocessing_intrinsic(self, input_data, index):
        """
        1. 원본 intrinsic을 사용할 경우, "스케일링 크기 / 원본 크기" 비율을 곱해서 intrinsic을 줄여줌
        2. monodepth2의 intrinsic을 사용할 경우, 스케일링 크기만 곱해서 intrinsic을 늘려줌
        """
        K_copy       = self.K.copy()
        K_copy[0, :] = K_copy[0, :] * self.scale_list[0][1]
        K_copy[1, :] = K_copy[1, :] * self.scale_list[0][0]
        inv_K        = np.linalg.pinv(K_copy)
        input_data[("K", 0)]     = torch.from_numpy(K_copy)
        input_data[("inv_K", 0)] = torch.from_numpy(inv_K)
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
        do_flip        = self.is_training and random.random() > 0.5
        do_auge        = self.is_training and random.random() > 0.5

        input_data     = {} # 데이터를 담아서 리턴할 배치 딕셔너리
        input_data     = self.preprocessing_image(input_data, index, do_flip)
        input_data     = self.preprocessing_depth(input_data, index, do_flip)

        if do_auge: # 프레임 아이디마다 서로 다른 스케일을 한 번에 recolor augments를 하여 "colo_aug" 키로 저장
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    self.recolor_image(input_data[("color", frame_id, scale)]) for scale in self.num_scales})

        else:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    input_data[("color", frame_id, scale)] for scale in self.num_scales})

        # input_data에 들어있는 모든 키의 값을 토치 텐서 타입으로 젼환
        input_data = {key: self.numpy2tensor(input_data[key]) for key in input_data}

        # 원본 이미지를 스케일링한 것과 동일하게 K, inv_K 계산해서 동일하게 스케일링
        input_data  = self.preprocessing_intrinsic(input_data, index)
        return input_data


    def __len__(self):
        return len(self.image_sequence)