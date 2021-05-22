# -*- coding: utf-8 -*-
import os
import numpy as np
from collections import Counter

import torch
import matplotlib.pyplot as plt


"""
포인트 클라우드로부터 뎁스 맵을 추출하는 모듈이 있거나, 그외 여러가지 유틸리티 모음
"""
def load_velodyne_points(filename):
    """
    벨로다인 포인트 클라우드 파일을 읽어들이는 함수
    Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """
    벨로다인 캘리브레이션 파일을 읽어들이는 함수
    Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """
    Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def Point2Depth(velo2cam_path, cam2cam_path, point_path, cam = 2, vel_depth = True):
    """
    벨로다인 파일을 읽어서 뎁스 맵을 만드는 함수
    example)
    points_path  = "./dataset/raw_image/train/2011_09_26_drive_0017_sync/velodyne_points/data/0000000042.bin"
    calib_path   = {"2011_09_26": "./dataset/calib_path/2011_09_26",
                    "2011_09_28": "./dataset/calib_path/2011_09_28",
                    "2011_09_29": "./dataset/calib_path/2011_09_29",
                    "2011_09_30": "./dataset/calib_path/2011_09_30",
                    "2011_10_03": "./dataset/calib_path/2011_10_03",}
    depth = generate_depth_map(calib_path["2011_09_26"], points_path, 2, False)

    GT depth information
    np.max ~ 80
    np.min ~ 0
    shape: [375, 1242]
    """
    """
    1. load calibration files
    """
    cam2cam  = read_calib_file(os.path.join(cam2cam_path))
    velo2cam = read_calib_file(os.path.join(velo2cam_path))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    """
    2. 이미지 모양을 획득 (375, 1242)
    """
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    

    """
    3.
    3차원 포인트 점을 카메라 좌표계로 변환하고 다시 K를 곱해서 이미지로 사영시키는 수식
    먼저 4x4 항등행렬을 선언하고 여기서 3x3 부분은 회전 행렬을 붙인다. (R_rect_00)
    그리고 모션 벡터를 cam2cam의 P_rect_0 성분을 불러와서 둘을 np.dot한다.
    마지막으로 velo2cam 매트릭스를 np.dot하면 벨로다인 포인트 -> 이미지로 사영하는 매트릭스를 만듬
    """
    R_cam2rect = np.eye(4)                                  # 4x4 항등행렬
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3) # 회전 운동
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)     # 모션 벡터
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    
    """
    4.
    벨로다인 포인트 클라우드를 불러오고, x, y, z, 1의 homogenous 좌표계로 만듬
    load velodyne points and remove all behind image plane (approximation)
    each row of the velodyne data is forward, left, up, reflectance
    """
    velo = load_velodyne_points(point_path)
    velo = velo[velo[:, 0] >= 0, :]


    """
    5.
    벨로다인 포인트 homogenous 값을 카메라의 이미지 좌표에 사영하는 계산과정 이미지 = 사영행렬 * 3차원 벨로다인 포인트
    """
    velo_pts_im        = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] # shape is (포인트 갯수, x, y, 1 값)

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]


    """
    check if in bounds
    use minus 1 to get the exact same value as KITTI matlab code
    1. velo_path_im.shape는 3개 (x, y, 1) 성분이 61021개 있다. 여기의 x, y 좌표에서 1씩 빼준 것을 다시 velo_pts_im[:, 0] and [:, 1]에 대입
    2. 그리고 x 좌표가 0 이상이고 y 좌표가 0 이상인 값만 유효한 인덱스로 취급한다.
    3. 그리고 val_ind 이면서 동시에 velo_pts_im 좌표의 위치가 이미지의 크기보다 작은 것만 다시 val_inds로 할당 (그래야만 이미지에 좌표가 잘 맺히므로)
    4. 마지막으로 그 유효한 좌표의 위치, 즉 True만 velo_pts_im로 취급
    """
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds          = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds          = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im       = velo_pts_im[val_inds, :]

    depth = np.zeros((im_shape[:2])) # 이미지로 사영, 375, 1245 사이즈의 zero map을 만듬
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    
    """
    마지막
    find the duplicate points and choose the closest depth
    중복된 값을 제거
    """
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts   = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    return depth


#################################################################################################################################################
# ################################################################################################################################################
def tensor2numpy(tensor):
    return tensor.numpy()


def numpy2tensor(numpy):
    return torch.from_numpy(numpy)


def show_image(image, option = "torch", size = (10, 4), cmap = "magma"):
    plt.rcParams["figure.figsize"] = size
    """
    토치나 텐서플로우 형태의 이미지를 받아서 이미지를 띄우는 함수
    Args: tensor type
        Pytorch:    [B, N, H, W]
        Tensorflow: [B, H, W, C]
    """

    if option == "torch":
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        elif len(image.shape) == 4:
            image = np.squeeze(image, axis = 0)
            image = np.transpose(image, (1, 2, 0))

    elif option == "tensorflow": # N H W C
        if len(image.shape) == 3:
            pass
        elif len(image.shape) == 4:
            image = np.squeeze(image, axis = 3)

    """
    uint8 -> float32로 바꾸면 엄청 큰 정수 단위의 float32가 됨
    따리서 255.로 나누어 주는 것이 중요
    그리고 cv2.imread로 불러온 이미지를 plt.imshow로 띄울때는 cv2.COLOR_BGR2RGB
    """
    plt.imshow(image, cmap = cmap, vmax = np.percentile(image, 95))
    plt.show()
    
def sample_dataset(dataloader):
    sample = 0
    for index, data in enumerate(dataloader):
        sample = data
        if index == 0:
            break
    return sample
