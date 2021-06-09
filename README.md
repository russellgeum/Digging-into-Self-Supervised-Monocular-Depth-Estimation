# Intro (NOT official)
Implementation of "Digging into Self-Supervised Monocular Depth Estimation (ICCV 2019)"  
- [Digging into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260)  
This repo follow up new KITTI depth bechmark split (But except Person videos)  
- [KITTI Depth Comletion Evaluation](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)  
- [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)  
# Requirements  
```
tqdm
natsorted
numpy
Pillow
torch >= 1.7.1
albumentations == 0.5.2
```
# Sample
![image](https://github.com/Doyosae/Digging_Into_Self-Supervised_Monocular_Depth_Estimation/blob/main/model_save/sample/image.gif)  
![disp](https://github.com/Doyosae/Digging_Into_Self-Supervised_Monocular_Depth_Estimation/blob/main/model_save/sample/disp.gif)  
# Folder  
```
dataset/
    calib_path/
    raw_image/
model_dataloader/
model_layer/
model_loss/
model_save/
model_logger.py
model_parser.py
model_train.py
model_utility.py
```
# Usage
```
python model_train.py
```
# Will...
1. metric problem  
2. Eigen split?  
# Reference  
[Offical Code](https://github.com/nianticlabs/monodepth2)  