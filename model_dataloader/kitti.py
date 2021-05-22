import os
import numpy as np
from natsort import natsorted


class KITTI(object):
    def __init__(self, path):
        """
        Args:
        path: "./dataset/raw_images/train" or "./dataset/raw_image/val"
              키티 데이터 폴더가 들어있는 train or val 경로를 입력
        이 클래스는 train 또는 val 폴더 경로를 입력 받으면,
        그 안의 l 이미지, r 이미지, LiDAR 이미지의 파일 명들을 딕셔너리로 반환한다.

        def search
            return:
                dict = {"1":
                            {"l": [이미지1, 이미지2, 이미지3, 이미지4, 이미지5 ... ...],
                             "r": [이미지1, 이미지2, 이미지3, 이미지4, 이미지5 ... ...],
                             "p": [이미지2, 이미지3, 이미지4]
                            }
                        }
        
        이러한 형태로 반환하는 이유는 이미지는 l, r의 pair가 정확하게 맞지만,
        포인트 클라우드 데이터는 때때로 안 맞는 인덱싱이 존재해서 향후 핸들링의 편의를 위함이다.
        """
        self.path   = path
        self.l_path = "image_02/data"
        self.r_path = "image_03/data"
        self.p_path = "velodyne_points/data"

    def search(self):
        dataname_dict = {}

        for index, yyyy_mm_dd in enumerate(os.listdir(self.path)):
            category = str(index+1)
            dataname_dict[category]      = {}
            dataname_dict[category]["l"] = []
            dataname_dict[category]["r"] = []
            dataname_dict[category]["p"] = []

            for filename in os.listdir(self.path + "/" + yyyy_mm_dd + "/" + self.l_path):
                dataname_dict[category]["l"].append(filename)

            for filename in os.listdir(self.path + "/" + yyyy_mm_dd + "/" + self.r_path):
                dataname_dict[category]["r"].append(filename)

            for filename in os.listdir(self.path + "/" + yyyy_mm_dd + "/" + self.p_path):
                dataname_dict[category]["p"].append(filename)
        
        """
        return test
        index = 3
        print(len(dataset[str(index)]["l"]))
        print(len(dataset[str(index)]["r"]))
        print(len(dataset[str(index)]["p"]))

        print(dataset[str(index)]["l"][100])
        print(dataset[str(index)]["r"][100])
        print(dataset[str(index)]["p"][100])
        """
        return dataname_dict


class KITTIRawDataset(object):
    def __init__(self, path, use_point):
        """
        Args:
        path: "./dataset/raw_images/train"
              or 
              "./dataset/raw_image/val"
              키티 데이터 폴더가 들어있는 train or val 경로를 입력

        use_point: 포인트 클라우드 데이터르 포함한다면 포인트 클라우드 인덱싱에 맞춘다.
                   그렇지 않고, 이미지만 필요하다면 포인트 클라우드 인덱싱을 맞출 필요 없다.
                   그것을 명시하는 기준

        return
        mono_sequence = {"image": [(이미지1, 이미지2), (이미지2, 이미지3), (이미지3, 이미지4) ...]
                         "point": [(이미지1의 포인트, velo2cam 경로, cam2cam 경로),
                                    이미지2의 포인트, vwlo2cam 경로, cam2cam 경로),]
            
        """
        self.path       = path
        
        self.use_point  = use_point # True or Fals
        self.l_path     = "image_02/data"
        self.r_path     = "image_03/data"
        self.p_path     = "velodyne_points/data"

        # velo2cam and cam2cam path ~ "./dataset" + "/" + "calib_path" + "/" + "2011_09_26" + "/" + "calib_cam_to_cam.txt"
        self.dataset, _ = self.path.split("/raw_image") # self.path 이름에서 "./dataset" 만 분리
        self.calib_path = "calib_path"
        self.velo2cam   = "calib_velo_to_cam.txt"
        self.cam2cam    = "calib_cam_to_cam.txt"

        """
        class KITTI를 상속받아도 될 듯 하지만, 같은 코드 안에 있다면 이렇게 사용해도 무방
        """
        self.dataname_dict = self.search()


    def search(self):
        dataname_dict = {}
        for index, yyyy_mm_dd in enumerate(os.listdir(self.path)):
            category = index+1
            """
            딕셔너리에 들어갈 키 값 지정
            "l" : 왼쪽 이미지
            "r" : 오른쪽 이미지
            "p" : 포인트 클라우드
            "velo2cam" : 벨로다인에서 카메라 좌표계로 옮기는 정보가 있는 파일
            "cam2cam"  : 카메라 좌표에서 카메라 좌표로 옮기는 파일 # 확인 필요
            """
            dataname_dict[category]      = {}
            dataname_dict[category]["l"] = []
            dataname_dict[category]["r"] = []
            dataname_dict[category]["p"] = []
            dataname_dict[category]["velo2cam"] = []
            dataname_dict[category]["cam2cam"]  = []
            
            if self.use_point == True:
                # "./dataset/train/2011_09_26_drive_0001_sync/image_02/data"
                common_l_path = self.path + "/" + yyyy_mm_dd + "/" + self.l_path
                common_r_path = self.path + "/" + yyyy_mm_dd + "/" + self.r_path
                common_p_path = self.path + "/" + yyyy_mm_dd + "/" + self.p_path

                # point cloud 파일 이름에 맞는 left, right 이미지 파일을 로드
                for filename in os.listdir(self.path + "/" + yyyy_mm_dd + "/" + self.p_path):
                    # "./dataset/calib_pth/2011_09_26/"
                    velo2cam_path = self.dataset + "/" + self.calib_path + "/" + yyyy_mm_dd.split("_drive")[0] + "/" + self.velo2cam
                    cam2cam_path  = self.dataset + "/" + self.calib_path + "/" + yyyy_mm_dd.split("_drive")[0] + "/" + self.cam2cam

                    # 포인트 클라우드와 같은 이름의 파일만 (즉 같은 타임스탬프) "l", "r", "p"에 추가
                    dataname_dict[category]["l"].append(common_l_path + "/" + filename.split(".")[0] + ".jpg")
                    dataname_dict[category]["r"].append(common_r_path + "/" + filename.split(".")[0] + ".jpg")
                    dataname_dict[category]["p"].append(common_p_path + "/" + filename)
                    dataname_dict[category]["velo2cam"].append(velo2cam_path)
                    dataname_dict[category]["cam2cam"].append(cam2cam_path)

            elif self.use_point == False:
                # "./dataset/train/2011_09_26_drive_0001_sync/image_02/data"
                common_l_path = self.path + "/" + yyyy_mm_dd + "/" + self.l_path
                common_r_path = self.path + "/" + yyyy_mm_dd + "/" + self.r_path
                
                for filename in os.listdir(self.path + "/" + yyyy_mm_dd + "/" + self.l_path):
                    # "./dataset/train/2011_09_26_drive_0001_sync/image_02/data/0000000000.jpg"
                    dataname_dict[category]["l"].append(common_l_path + "/" + filename)

                for filename in os.listdir(self.path + "/" + yyyy_mm_dd + "/" + self.r_path):
                    cam2cam_path  = self.dataset + "/" + self.calib_path + "/" + yyyy_mm_dd.split("_drive")[0] + "/" + self.cam2cam
                    dataname_dict[category]["r"].append(common_r_path + "/" + filename)
                    dataname_dict[category]["cam2cam"].append(cam2cam_path)
            
            dataname_dict[category]["l"]        = natsorted(dataname_dict[category]["l"])
            dataname_dict[category]["r"]        = natsorted(dataname_dict[category]["r"])
            dataname_dict[category]["p"]        = natsorted(dataname_dict[category]["p"])
            dataname_dict[category]["velo2cam"] = natsorted(dataname_dict[category]["velo2cam"])
            dataname_dict[category]["cam2cam"]  = natsorted(dataname_dict[category]["cam2cam"])
        return dataname_dict


    def mono_sequence(self, types, num_frames):
        """
        Args:
            type: "left" or "right", 어떤 사진을 할당 받을지 선택
            num_frames: 몇 개의 프레임 시퀀스를 이룰지 단위 수를 지정
        """
        if types == "left":
            mono_type = "l"
        elif types == "right":
            mono_type = "r"
        else:
            raise "mono sequence is left or right type, If you need stereo sequence, \
                   Dont use def mono_sequence and Go def stereo_sequence"

        sequence = {}
        
        if self.use_point == True:
            sequence["image"] = []
            sequence["point"] = []
            sequence["v2c"]   = []
            sequence["c2c"]   = []

            for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                for index in range(len(self.dataname_dict[yyyy_mm_dd][mono_type])-(num_frames-1)):
                    if num_frames == 2:
                        sequence["image"].append((
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+1]))
                        sequence["point"].append((
                                        self.dataname_dict[yyyy_mm_dd]["p"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+1]))
                        sequence["v2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["velo2cam"][index])
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 3:
                        sequence["image"].append((
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+1], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+2]))
                        sequence["point"].append((
                                        self.dataname_dict[yyyy_mm_dd]["p"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+1],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+2]))
                        sequence["v2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["velo2cam"][index])
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 5:
                        sequence["image"].append((
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+1], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+2], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+3],
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+4]))
                        sequence["point"].append((
                                        self.dataname_dict[yyyy_mm_dd]["p"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+1],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+2],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+3],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+4]))
                        sequence["v2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["velo2cam"][index])
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])
                    else:
                        raise "pair is 2 or 3 or 5"

        elif self.use_point == False:
            sequence["image"] = []
            sequence["c2c"]   = []

            for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                for index in range(len(self.dataname_dict[yyyy_mm_dd][mono_type])-(num_frames-1)):
                    if num_frames == 2:
                        sequence["image"].append((
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+1]))
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 3:
                        sequence["image"].append((
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+1], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+2]))
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 5:
                        sequence["image"].append((
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+1], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+2], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+3], 
                                        self.dataname_dict[yyyy_mm_dd][mono_type][index+4]))
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])
                    else:
                        raise "pair is 2 or 3 or 5"

        """
        return test
        pprint(len(mono_sequence["image"]))
        pprint(len(mono_sequence["point"]))
        pprint(mono_sequence["image"][:2])
        pprint(mono_sequence["point"][:2])
        """
        return sequence


    def stereo_sequence(self, num_frames):
        """
        Args:
            type: "left" or "right", 어떤 사진을 할당 받을지 선택
            num_frames: 몇 개의 프레임 시퀀스를 이룰지 단위 수를 지정
        """
        sequence           = {}
        sequence["stereo"] = []

        if self.use_point == True:
            sequence["v2c"]   = []
            sequence["c2c"]   = []
            for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                for index in range(len(self.dataname_dict[yyyy_mm_dd]["l"])-(num_frames-1)):
                    if num_frames == 1:
                        sequence["stereo"].append((
                                        self.dataname_dict[yyyy_mm_dd]["l"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index]))
                        sequence["v2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["velo2cam"][index])
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 2:
                        sequence["stereo"].append((
                                        self.dataname_dict[yyyy_mm_dd]["l"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index],
                                        self.dataname_dict[yyyy_mm_dd]["l"][index+1], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index+1],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+1]))
                        sequence["v2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["velo2cam"][index])
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 3:
                        sequence["stereo"].append((
                                        self.dataname_dict[yyyy_mm_dd]["l"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index],
                                        self.dataname_dict[yyyy_mm_dd]["l"][index+1], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index+1],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+1],
                                        self.dataname_dict[yyyy_mm_dd]["l"][index+2], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index+2],
                                        self.dataname_dict[yyyy_mm_dd]["p"][index+2]))
                        sequence["v2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["velo2cam"][index])
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])
                    else:
                        raise "pair is 1 or 2 or 3"

        elif self.use_point == False:
            for _, yyyy_mm_dd in enumerate(self.dataname_dict):
                for index in range(len(self.dataname_dict[yyyy_mm_dd]["l"])-(num_frames-1)):
                    if num_frames == 1:
                        sequence["stereo"].append((
                                        self.dataname_dict[yyyy_mm_dd]["l"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index]))
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 2:
                        sequence["stereo"].append((
                                        self.dataname_dict[yyyy_mm_dd]["l"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index],
                                        self.dataname_dict[yyyy_mm_dd]["l"][index+1], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index+1]))
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])

                    elif num_frames == 3:
                        sequence["stereo"].append((
                                        self.dataname_dict[yyyy_mm_dd]["l"][index], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index],
                                        self.dataname_dict[yyyy_mm_dd]["l"][index+1], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index+1],
                                        self.dataname_dict[yyyy_mm_dd]["l"][index+2], 
                                        self.dataname_dict[yyyy_mm_dd]["r"][index+2]))
                        sequence["c2c"].append(
                                        self.dataname_dict[yyyy_mm_dd]["cam2cam"][index])
                    else:
                        raise "pair is 1 or 2 or 3"
        """
        return test
        pprint(len(stereo_sequence["stereo"]))
        pprint(stereo_sequence["stereo"][12312])
        """
        return sequence        



class GetKITTI(object):
    def __init__(self, 
                path: str, 
                option: str,
                use_point: bool, 
                types: str,
                num_frames: int):
        """
        Args:
            path: "./dataset/raw_images"
            option: "train" or "val"
            use_point: True or False
            type: "stereo", "left", "right"
        """
        self.path       = path
        self.option     = option
        self.use_point  = use_point
        self.type       = types
        self.KITTI      = KITTIRawDataset(self.path + "/" + self.option, self.use_point)
        self.num_frames = num_frames
    
    def item(self):
        if self.type == "stereo":
            self.sequence = self.KITTI.stereo_sequence(self.num_frames)

        elif self.type == "left" or self.type == "right":
            self.sequence = self.KITTI.mono_sequence(self.type, self.num_frames)

        return self.sequence