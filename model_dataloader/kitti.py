import os
import numpy as np
from natsort import natsorted



def read_file(file_path, types):
    file  = open(file_path + "/" + types + ".txt", "r")
    lines = file.readlines()
    lines = [line.rstrip(" \n") for line in lines]
    return lines


def make_path(lines):
    path_list  = [line.split("_drive")[0] + "/" + line for line in lines]
    return path_list


class KITTIRawDataset(object):
    def __init__(self, data_path: str, file_path: str, mode: str, use_point: bool):
        """
        Args:
        data_path: "./dataset/kitti-master"
        file_path: "./dataset/kitti_splits"
        types: "train" or "valid
        use_point: True or False
                   포인트 클라우드 데이터르 포함한다면 포인트 클라우드 인덱싱에 맞춘다.
                   그렇지 않고, 이미지만 필요하다면 포인트 클라우드 인덱싱을 맞출 필요 없다.
                   그것을 명시하는 기준
        """
        lines  = read_file(file_path, mode)
        self.file_path  = make_path(lines)
        self.data_path  = data_path
        self.use_point  = use_point # True or False

        self.l_path     = "image_02/data"
        self.r_path     = "image_03/data"
        self.p_path     = "velodyne_points/data"
        self.cam2cam    = "calib_cam_to_cam.txt"
        self.imu2velo   = "calib_imu_to_velo.txt"
        self.velo2cam   = "calib_velo_to_cam.txt"

        # class KITTI를 상속받아도 될 듯 하지만, 같은 코드 안에 있다면 이렇게 사용해도 무방
        self.dataname_dict = self.search()


    def search(self):
        dataname_dict = {}
        # 2011_09_26/2011_09_26_drive_0001_sync or 2011_09_28/2011_09_26_drive_0009_sync ...
        for index, yyyy_mm_dd in enumerate(self.file_path):
            """
            딕셔너리에 들어갈 키 값 지정
            "l" : 왼쪽 이미지
            "r" : 오른쪽 이미지
            "p" : 포인트 클라우드
            "v2c" : 벨로다인에서 카메라 좌표계로 옮기는 정보가 있는 파일
            "c2c"  : 카메라 좌표에서 카메라 좌표로 옮기는 파일 # 확인 필요
            """
            category = index+1
            dataname_dict[category]        = {}
            dataname_dict[category]["l"]   = []
            dataname_dict[category]["r"]   = []
            dataname_dict[category]["p"]   = []
            dataname_dict[category]["v2c"] = []
            dataname_dict[category]["c2c"] = []
            
            if self.use_point == True:
                # "./dataset/kitti_splits/2011_09_26/2011_09_26_drive_0001_sync/image_02/data
                # "./dataset/kitti_splits/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data
                common_l_path = self.data_path + "/" + yyyy_mm_dd + "/" + self.l_path
                common_r_path = self.data_path + "/" + yyyy_mm_dd + "/" + self.r_path
                common_p_path = self.data_path + "/" + yyyy_mm_dd + "/" + self.p_path
                cam2cam_path  = self.data_path + "/" + yyyy_mm_dd.split("/")[0] + "/" + self.cam2cam
                velo2cam_path = self.data_path + "/" + yyyy_mm_dd.split("/")[0] + "/" + self.velo2cam

                # point cloud 파일 이름에 맞는 left, right 이미지 파일을 로드
                for velo_filename in os.listdir(common_p_path): # filename: 0000000000.bin, 0000000001.bin ... ...
                    # 포인트 클라우드와 같은 이름의 파일만 (즉 같은 타임스탬프) "l", "r", "p"에 추가, 이때 cam2cam, velo2cam도 추가
                    dataname_dict[category]["l"].append(common_l_path + "/" + velo_filename.split(".")[0] + ".jpg")
                    dataname_dict[category]["r"].append(common_r_path + "/" + velo_filename.split(".")[0] + ".jpg")
                    dataname_dict[category]["p"].append(common_p_path + "/" + velo_filename)
                    dataname_dict[category]["c2c"].append(cam2cam_path)
                    dataname_dict[category]["v2c"].append(velo2cam_path)

            elif self.use_point == False:
                common_l_path = self.data_path + "/" + yyyy_mm_dd + "/" + self.l_path
                common_r_path = self.data_path + "/" + yyyy_mm_dd + "/" + self.r_path
                cam2cam_path  = self.data_path + "/" + yyyy_mm_dd.split("/")[0] + "/" + self.cam2cam # Not use velo2cam file
                
                for filename in zip(os.listdir(common_l_path), os.listdir(common_r_path)):
                    dataname_dict[category]["l"].append(common_l_path + "/" + filename)
                    dataname_dict[category]["r"].append(common_r_path + "/" + filename)
                    dataname_dict[category]["c2c"].append(cam2cam_path)
            
            dataname_dict[category]["l"]   = natsorted(dataname_dict[category]["l"])
            dataname_dict[category]["r"]   = natsorted(dataname_dict[category]["r"])
            dataname_dict[category]["p"]   = natsorted(dataname_dict[category]["p"])
            dataname_dict[category]["v2c"] = natsorted(dataname_dict[category]["v2c"])
            dataname_dict[category]["c2c"] = natsorted(dataname_dict[category]["c2c"])
        return dataname_dict


    def mono_sequence(self, types, num_frames):
        """
        cf) [02: l, 03: r]
        Args:
            type: "l" or "r" or "all", 어떤 타입의 사진을 할당 받을지 선택
            num_frames: 몇 개의 프레임 시퀀스를 이룰지 단위 수를 지정
        """
        if types not in ["l", "r", "all"]:
            raise "모노 시퀀스 생성의 타입은 'l', 'r', 'all' 만 입력 가능"
        
        if type(num_frames) is not int:
            raise "프레임 수는 반드시 정수만 입력"

        sequence = {}
        sequence["image"] = []
        sequence["c2c"]   = []
        if self.use_point == True:
            sequence["point"] = []
            sequence["v2c"]   = []

            if types == "all":
                for _, yyyy_mm_dd in enumerate(self.dataname_dict): 
                    for index in range(len(self.dataname_dict[yyyy_mm_dd]["l"]) - (num_frames-1)):
                        l_pair = tuple(self.dataname_dict[yyyy_mm_dd]["l"][index + num] for num in range(num_frames))
                        r_pair = tuple(self.dataname_dict[yyyy_mm_dd]["r"][index + num] for num in range(num_frames))
                        p_pair = tuple(self.dataname_dict[yyyy_mm_dd]["p"][index + num] for num in range(num_frames))

                        sequence["image"].append(l_pair)
                        sequence["point"].append(p_pair)
                        sequence["v2c"].append(self.dataname_dict[yyyy_mm_dd]["v2c"][index])
                        sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])
                        sequence["image"].append(r_pair)
                        sequence["point"].append(p_pair)
                        sequence["v2c"].append(self.dataname_dict[yyyy_mm_dd]["v2c"][index])
                        sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])
            else:
                for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                    for index in range(len(self.dataname_dict[yyyy_mm_dd][types])-(num_frames-1)):
                        image_pair = tuple(self.dataname_dict[yyyy_mm_dd][types][index + num] for num in range(num_frames))
                        point_pair = tuple(self.dataname_dict[yyyy_mm_dd]["p"][index + num] for num in range(num_frames))

                        sequence["image"].append(image_pair)
                        sequence["point"].append(point_pair)
                        sequence["v2c"].append(self.dataname_dict[yyyy_mm_dd]["v2c"][index])
                        sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])

        elif self.use_point == False:

            if types == "all":
                for _, yyyy_mm_dd in enumerate(self.dataname_dict): 
                    for index in range(len(self.dataname_dict[yyyy_mm_dd]["l"])-(num_frames-1)):
                        l_pair = tuple(self.dataname_dict[yyyy_mm_dd]["l"][index + num] for num in range(num_frames))
                        r_pair = tuple(self.dataname_dict[yyyy_mm_dd]["r"][index + num] for num in range(num_frames))
                        sequence["image"].append(l_pair)
                        sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])
                        sequence["image"].append(r_pair)
                        sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])
            else:
                for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                    for index in range(len(self.dataname_dict[yyyy_mm_dd][types])-(num_frames-1)):
                        image_pair = tuple(self.dataname_dict[yyyy_mm_dd][types][index + num] for num in range(num_frames))

                        sequence["image"].append(image_pair)
                        sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])
        return sequence


    def stereo_sequence(self, num_frames):
        """
        Args:
            type: "left" or "right", 어떤 사진을 할당 받을지 선택
            num_frames: 몇 개의 프레임 시퀀스를 이룰지 단위 수를 지정
        """
        if type(num_frames) is not int:
            raise "프레임 수는 반드시 정수만 입력"

        sequence        = {}
        sequence["l"]   = []
        sequence["r"]   = []
        sequence["c2c"] = []

        if self.use_point == True:
            sequence["p"]   = []
            sequence["v2c"] = []

            for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                for index in range(len(self.dataname_dict[yyyy_mm_dd]["l"])-(num_frames-1)):
                    l_pair = tuple(self.dataname_dict[yyyy_mm_dd]["l"][index + num] for num in range(num_frames))
                    r_pair = tuple(self.dataname_dict[yyyy_mm_dd]["r"][index + num] for num in range(num_frames))
                    p_pair = tuple(self.dataname_dict[yyyy_mm_dd]["p"][index + num] for num in range(num_frames))
                    
                    sequence["l"].append(l_pair)
                    sequence["r"].append(r_pair)
                    sequence["p"].append(p_pair)
                    sequence["v2c"].append(self.dataname_dict[yyyy_mm_dd]["v2c"][index])
                    sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])

        elif self.use_point == False:
            for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                for index in range(len(self.dataname_dict[yyyy_mm_dd]["l"])-(num_frames-1)):
                    l_pair = tuple(self.dataname_dict[yyyy_mm_dd]["l"][index + num] for num in range(num_frames))
                    r_pair = tuple(self.dataname_dict[yyyy_mm_dd]["r"][index + num] for num in range(num_frames))
                    sequence["l"].append(l_pair)
                    sequence["r"].append(r_pair)
                    sequence["c2c"].append(self.dataname_dict[yyyy_mm_dd]["c2c"][index])
        return sequence        



class GetKITTI(object):
    def __init__(self, data_path: str, file_path: str, mode: str, use_point: bool, types: str, num_frames: int):
        """
        Args:
        data_path: "./dataset/kitti-master"
        file_path: "./dataset/kitti_splits"
        mode:      "train" or "valid
        use_point: True or False
        type: "l", "r", "s", "all"
        """
        self.data_path  = data_path
        self.file_path  = file_path
        self.mode       = mode
        self.use_point  = use_point
        self.types      = types
        self.num_frames = num_frames

        self.KITTI      = KITTIRawDataset(data_path, file_path, mode, use_point)
    
    def item(self):
        if self.types == "s":
            self.sequence = self.KITTI.stereo_sequence(self.num_frames)

        elif self.types == "l" or "r" or "all":
            self.sequence = self.KITTI.mono_sequence(self.types, self.num_frames)
        return self.sequence