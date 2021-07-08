import os
import cv2
import random
import numpy as np
from PIL import Image
from natsort import natsorted

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import skimage.transform
from albumentations import Resize
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import ColorJitter
from model_utility import *



class GetKITTI(object):
    def __init__(self, datapath: str, mode: list, cut: list):
        """
        KITTI 데이터 폴더의 구조
        ㄴKITTI
            ㄴ2011_09_26/2011_09_26_drive_0001_sync
                    ㄴimage_00/data
                    ㄴimage_01/data
                    ㄴimage_02/data
                        ㄴ0000000000.jpg
                        ㄴ0000000001.jpg
                    ㄴimage_03/data
                        ㄴ0000000000.jpg
                        ㄴ0000000001.jpg
                    ㄴvelodyne_points/data
                        ㄴ0000000000.bin
                        ㄴ0000000001.bin
            ㄴ2011_09_26/2011_09_26_drive_0002_sync
            ㄴ2011_09_28
            ㄴ2011_09_29
            ㄴ2011_09_30
            ㄴ2011_10_03
        
        벨로다인 포인트 파일을 기준으로 키프레임 선택 (키프레임은 GT가 존재해야함)
        따라서 벨로다인 파일의 확장자를 제거한 프레임 인덱스로 키프레임 형식으로 지정
        해당 프레임 인덱스 이름에 yyyy_mm_dd 형태와 side_map을 섞어서 부여
        (레거시 kitti_splits 규약을 따름)

        ex)
        2011_10_03/2011_10_03_drive_0034_sync 0000000190 r
        2011_10_03/2011_10_03_drive_0034_sync 0000000307 l
        2011_09_26/2011_09_26_drive_0117_sync 0000000134 r
        2011_09_30/2011_09_30_drive_0028_sync 0000000534 l

        Args:
            datapath: "./dataset/kitti"
            model: "train" or "val" 데이터 폴더 리스트
            side: "l" or "r"
        """
        self.datapath  = datapath
        self.mode      = mode
        self.cut       = cut
        self.side_map  = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.l_path    = "image_02/data"
        self.r_path    = "image_03/data"
        self.velo_path = "velodyne_points/data"


    def search(self):
        all_filename = []
        for yyyy_mm_dd in self.mode: # 각 날짜 별 폴더마다 순회
            velodyne_path = os.path.join(self.datapath, yyyy_mm_dd, self.velo_path)
            # left_path  = os.path.join(self.datapath, yyyy_mm_dd, self.l_path)
            # right_path = os.path.join(self.datapath, yyyy_mm_dd, self.r_path)

            # 왼쪽 카메라 파일 (image_02) 와 오른쪽 카메라 파일 (image_03)을 순회
            left_filename  = []
            right_filename = []
            for filename in os.listdir(velodyne_path): # 벨로다인 파일 이름을 순회
                # 0000000011.bin, ... ... 0000000035.bin 파일을 0000000011, 0000000035으로 쪼갬
                frame_index = filename.split(".bin")[0]

                # 쪼갠 frame_index은 image_02, image_03 폴더에 이미지 파일로 매칭됨 (스테레오 이미지)
                left_filename.append("".join([yyyy_mm_dd, " ", frame_index, " ", "l"]))
                right_filename.append("".join([yyyy_mm_dd, " ", frame_index, " ", "r"]))
            
            # 상대적인 프레임 아이디를 위해 양 끝 N개는 잘라줌
            modified_left_filename  = self.side_cut(left_filename, self.cut)
            modified_right_filename = self.side_cut(right_filename, self.cut)
            all_filename += modified_left_filename
            all_filename += modified_right_filename

        print("전체 길이  :  {}".format(len(all_filename)))
        random.shuffle(all_filename)
        return all_filename


    def side_cut(self, filename, cut):
        """
        이 클래스가 존재하는 이유
        프레임 인덱스가 키 프레임 기준으로 좌, 우 몇 까지 consecutive frame을 쓸 지 모름
        만약 키 프레임 인덱스가 0이면 왼쪽 consecutive frame은 음수가 되기 때문에 사용할 수 없음
        그래서 consecutive frame 범위를 두기 위해 맨 끝 프레임은 잘라내는 목적
        """        
        modified_filename = []
        modified_filename = filename[cut[0]: len(filename)-cut[1]]
        return modified_filename



class KITTIMonoDataset(Dataset):
    def __init__(self, datapath, filename, is_training, frame_ids, ext, scale = 1):
        super(KITTIMonoDataset, self).__init__()
        """
        Args:
            datapath: "./dataset/kitti"
            filename: splits file of KITTI
            is_training: True or False
            frame_ids: relative position list of key frame
            mode: "train" or "val" or "test"
            ext: ".jpg" or ".png"
            scale: 1
        
        interpolation 1은 쓰지 말 것, 성능이 나오지 않음, 0 아니면 3으로 실험
        albumentation Resize interpolation option
        0 : cv2.INTER_NEAREST, 
        1 : cv2.INTER_LINEAR, 
        2 : cv2.INTER_CUBIC, 
        3 : cv2.INTER_AREA, 
        4 : cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
        """
        self.datapath     = datapath
        self.filename     = filename
        self.is_training  = is_training
        self.frame_ids    = frame_ids
        self.ext          = ext
        self.scale        = scale
        self.inter        = cv2.INTER_AREA # Image.ANTIALIAS와 동등한가?
        self.side_map     = {"2": 2, "3": 3, "l": 2, "r": 3}
        # intrinsic camera (https://github.com/nianticlabs/monodepth2)
        self.K            = np.array([[0.58,    0,    0.5,    0],
                                      [0,    1.92,    0.5,    0],
                                      [0,       0,      1,    0],
                                      [0,       0,      0,    1]], dtype = np.float32) 

        self.resize        = {}
        self.scales        = list(range(4))
        self.origin_scale  = (375, 1242)
        self.default_scale = [(320, 1024), (192, 640), (96, 320), (48, 160)] # corresponding as scale: 0, 1, 2, 3
        self.scale_list    = [(self.default_scale[self.scale][0]//(2**i), 
                               self.default_scale[self.scale][1]//(2**i)) for i in self.scales]

        """
        데이터로더 프로세스 플로우
        1. 좌우로 뒤집을지 말지 결정하는 do_flip(bool)과 함께 이미지를 로드 (원본 이미지)
           키티 데이터 원본 크기는 (375, 1245)
        2. 원본 스케일 이미지를 원하는 스케일로 바꾸고, 그 스케일부터 2배율로 줄어드는 리스케일 [0, 1, 2, 3]
           ("color", <frame_id>, <scale>) 키 형태로 저장
        3. is_training 모드이면 do_auge를 주고, 각 이미지마다 augmentation을 적용
           ("color_aug", <frame_id>, <scale>) 키 형태로 저장
        4. GT로 사용하는 Point2Depth 이미지는 원본 스케일로 저장
           최소 스케일에 맞게 세팅된 K는 monodepth2의 intrinsic parameter를 따름 
           -> 변형한 이미지 스케일을 곱해서 복원
        5. 마지막으로 input_data의 모든 키 값들을 numpy2tensor 변환
        """
        for scale, (height, width) in enumerate(self.scale_list):
            self.resize[scale] = Resize(
                height = int(height), width  = int(width), interpolation = self.inter)
        self.depth_resize   = Resize(
            height = self.origin_scale[0], width = self.origin_scale[1], interpolation = 0)

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
        print("-- KITTI scaling table")
        print("interpolation     :  {0}".format(self.inter))
        print("Scale factor      :  {0}".format(self.scale))
        print("Default 0 scale   :  {0} {1}".format(self.default_scale[0][0], self.default_scale[0][1]))
        print("Default 1 scale   :  {0} {1}".format(int(self.default_scale[1][0]), int(self.default_scale[1][1])))
        print("Default 2 scale   :  {0} {1}".format(int(self.default_scale[2][0]), int(self.default_scale[1][1])))
        print("Default 3 scale   :  {0} {1}".format(int(self.default_scale[3][0]), int(self.default_scale[1][1])))
        print("Resolution List   :  {0}".format(self.scale_list))


    def get_image_path(self, folder_name, frame_index, side):
        # image_name = key_frame + ".jpg"
        # image_name = "{:010d}{}".format(frame_index, ".jpg")
        # image_name = str(frame_index).zfill(10) + ".jpg"
        # image_name = ''.join(['0' * (10 - len(str(frame_index)))]) + str(frame_index) + ".jpg"
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
        # image_instance = Image.open(image_path)
        # image_instance = image_instance.convert("RGB")
        # numpy_image    = np.array(image_instance)

        # if do_flip == True:
        #     numpy_image = self.flip_image(numpy_image)
        # return numpy_image
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                image_instance = img.convert('RGB')
                numpy_image    = np.array(image_instance)

                if do_flip == True:
                    numpy_image = self.flip_image(numpy_image)
                return numpy_image

    def load_point(self, calib_path, point_path, do_flip):
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
        """
        depth = Point2Depth(calib_path = calib_path, point_path = point_path)
        depth = skimage.transform.resize(depth, (1242, 375)[::-1], order = 0, preserve_range = True, mode = "constant")
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
        # depth = self.depth_resize(image = depth)
        # depth = np.reshape(depth["image"], (depth["image"].shape[0], depth["image"].shape[1], 1))
        
        if do_flip == True:
            depth = self.flip_image(depth)
        return depth


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
        tensor_image = self.image2tensor(image = numpy_image)
        return tensor_image[self.augment_key]


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

    def preprocessing_point(self, input_data, folder_name, key_frame, do_flip):
        """
        1. 캘리브레이션, 포인트 클라우드 파일 로드
        2. 포인트 클라우드 -> 뎁스 이미지로 변환한 데이터를 로드
        3. input_data 딕셔너리에 키 프레임의 뎁스 이미지 데이터 저장
        """
        calib_path, point_path   = self.get_point_path(folder_name, key_frame)
        depth                    = self.load_point(calib_path, point_path, do_flip)
        input_data[("depth", 0)] = depth
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
    
        # input_data 딕셔너리를 지정하고, folder_name, key_frame, side 여부, do_flip으로 이미지 전처리와 뎁스 전처리
        input_data = {}
        input_data = self.preprocessing_image(input_data, folder_name, key_frame, side, do_flip)        
        input_data = self.preprocessing_point(input_data, folder_name, key_frame, do_flip)

        if do_auge:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    self.recolor_image(input_data[("color", frame_id, scale)]) for scale in self.scales})
        else:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    input_data[("color", frame_id, scale)] for scale in self.scales})
        
        # input_data에 포함된 모든 키의 값을 torch.tensor 타입으로 변환
        input_data.update({key: self.numpy2tensor(input_data[key]) for key in input_data})

        # 원본 이미지를 스케일링한 비율만큼 K, inv_K도 동일하게 스케일링
        input_data = self.preprocessing_intrinsic(input_data)
        return input_data

    def __len__(self):
        return len(self.filename)