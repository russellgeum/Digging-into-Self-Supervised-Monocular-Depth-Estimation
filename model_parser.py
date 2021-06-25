import argparse

def main():
    parser = argparse.ArgumentParser(description = "Input optional guidance for training")
    parser.add_argument("--datapath",
                        default = "./dataset/kitti",
                        type = str,
                        help = "훈련 폴더가 있는 곳")
    parser.add_argument("--splits",
                        default = "./splits_kitti",
                        type = str,
                        help = "검증 폴더가 있는 곳")
    parser.add_argument("--load_path",
                        default = None,
                        type = str,
                        help = "로드 파일이 있는 경로")
    parser.add_argument("--epoch",
                        default = 25,
                        type = int,
                        help = "모델 에포크 수")
    parser.add_argument("--batch",
                        default = 20,
                        type = int,
                        help = "모델 배치 사이즈")
    parser.add_argument("--prepetch",
                        default = 2,
                        type = int,
                        help = "데이터 로더의 prefetch_factor")
    parser.add_argument("--num_workers",
                        default = 12,
                        type = int,
                        help = "데이터 로더의 num_workers")
    parser.add_argument("--learning_rate",
                        default = 1e-4, 
                        help = "모델 러닝 레이트") # 0.001이 아니라 0.0001로 할 것 (1e-4 = 0.0001)
    parser.add_argument("--scheduler_step",
                        default = 12,
                        type = int,
                        help = "모델 스케줄러 스텝 값")
    parser.add_argument("--smooth_const",
                        default = 1e-3,
                        help = "smooth loss loss의 가중치 값")
    
    """
    original size: (375, 1242)
    scale 0:       (320, 1024)
    scale 1:       (192, 640)
    scale 2:       (96, 320)
    scale 3:       (48, 160)
    """
    parser.add_argument("--scale_factor",
                        default = 1,
                        type = int,
                        help = "스케일 팩터 (오리지널 이미지 크기로부터)")
    parser.add_argument("--scales",
                        default = range(4),
                        type = str,
                        help = "뎁스 디코더의 레인지 범위")
    parser.add_argument("--min_depth",
                        default = 0.1,
                        type = float,
                        help = "최소 깊이")
    parser.add_argument("--max_depth",
                        default = 100.0,
                        type = float,
                        help = "최대 깊이")
    
    """
    포즈 네트워크 옵션, 항상 키 프레임은 맨 앞에
    ex) 
    [1, 0, 2] -> [0, -1, 1]: 1
    [2, 0, 1, 3, 4] -> [0, -2, -1, 1, 2]: 2
    [3, 0, 1, 2, 4, 5, 6] -> [0, -3, -2, -1, 1, 2, 3]: 3
    """
    parser.add_argument("--frame_idx",
                        default = [2, 0, 1, 3, 4],
                        type = str,
                        help = "프레임 리스트의 아이디")
    parser.add_argument("--frame_ids",
                        default = [0, -2, -1, 1, 2],
                        type = str,
                        help = "프레임 리스트의 아이디")
    parser.add_argument("--key_frame",
                        default = 2,
                        type = int,
                        help = "키 프레임의 인덱스를 입력, 프레임 수가 3이면 1, 5이면 2, 7이면 3")
    parser.add_argument("--pose_frames",
                        default = 2,
                        type = str,
                        help = "포즈 네트워크에 입력될 프레임 수")

    # 뎁스 인코더, 디코더 옵션
    parser.add_argument("--num_layers",
                        default = 18,
                        type = int,
                        help = "Resnet 모델 버전 18, 34 중 18이 기본")
    parser.add_argument("--pose_type",
                        default = "shared",
                        type = str,
                        help = "포즈 네트워크의 타입: posecnn, shared, separate")
    parser.add_argument("--weight_init",
                        default = True,
                        type = str,
                        help = "이미지넷 사전 학습 모델 사용 여부")

    # 오토 마스킹 옵션
    parser.add_argument("--use_automasking",
                        default = True,
                        type = bool,
                        help = "오토마스킹 사용 여부")
    args = parser.parse_args()
    return args