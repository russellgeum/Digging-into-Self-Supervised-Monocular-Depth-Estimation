# Intro
Implementation of "Digging into Self-Supervised Monocular Depth Estimation (ICCV 2019)" (NOT official)  
This repo follow up new KITTI depth bechmark split (But except Person videos)  
- [KITTI Depth Comletion Evaluation](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)  
- [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)  
# Requirements  
```
cv2
numpy
torch
albumentations
```
# Folder  
```
dataset/
    calib_path/
    raw_image/
model_dataloader/
model_layer/
model_loss/
model_parser.py
model_logger.py
model_train.py
model_utility.py
```
# Usage
```
python model_train.py
```
# Will...
1. Evaluation  
2. Eigen split?  
# Reference  
[Offical Code](https://github.com/nianticlabs/monodepth2)  