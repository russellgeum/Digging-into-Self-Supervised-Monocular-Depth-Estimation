import argparse

def main():
    parser = argparse.ArgumentParser(description = "Input optional guidance for training")
    parser.add_argument("--train_path",
                        default = "./dataset/raw_image",
                        type = str,
                        help = "훈련 폴더가 있는 곳")
    parser.add_argument("--valid_path",
                        default = "./dataset/raw_image",
                        type = str,
                        help = "검증 폴더가 있는 곳")
    parser.add_argument("--load_path",
                        default = None,
                        type = str,
                        help = "로드 파일이 있는 경로")
    parser.add_argument("--num_workers",
                        default = 4,
                        type = int,
                        help = "데이터 로더의 num_workers 인자")

    parser.add_argument("--epoch",
                        default = 50,
                        type = int,
                        help = "모델 에포크 수")
    parser.add_argument("--batch",
                        default = 16,
                        type = int,
                        help = "모델 배치 사이즈")
    parser.add_argument("--learning_rate",
                        default = 0.0015,
                        help = "모델 러닝 레이트")
    parser.add_argument("--scheduler_step",
                        default = 0.96,
                        type = float,
                        help = "모델 스케줄러 스텝")
    parser.add_argument("--min_depth",
                        default = 0.01,
                        type = float,
                        help = "최소 깊이")
    parser.add_argument("--max_depth",
                        default = 80.0,
                        type = float,
                        help = "최대 깊이")
    
    # 디폴트 스케일에서 얼마나 사이즈를 줄일지 결정
    parser.add_argument("--scale",
                        default = 1,
                        type = int,
                        help = "이미지 스케일")

    # 뎁스 인코더, 디코더 옵션
    parser.add_argument("--num_layers",
                        default = 18,
                        type = int,
                        help = "Resnet 모델 버전 18, 34 중 18이 기본")
    parser.add_argument("--pose_type",
                        default = "shared",
                        type = str,
                        help = "포즈 네트워크의 타입")
    parser.add_argument("--weight_init",
                        default = True,
                        type = str,
                        help = "이미지넷 사전 학습 모델 사용 여부")
    parser.add_argument("--scales",
                        default = range(4),
                        type = str,
                        help = "뎁스 디코더의 레인지 범위")

    # 포즈 네트워크 옵션
    parser.add_argument("--frame_ids",
                        default = [0, -2, -1, 1, 2],
                        type = str,
                        help = "프레임 리스트의 아이디")
    parser.add_argument("--pose_frames",
                        default = 2,
                        type = str,
                        help = "포즈 네트워크의 프레임 수")

    # 마스크 네트워크 옵션
    parser.add_argument("--use_automasking",
                        default = True,
                        type = bool,
                        help = "오토마스킹 사용 여부")
    args = parser.parse_args()
    return args