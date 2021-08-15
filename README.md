# monodepth2 (NOT official)
Implementation of "Digging into Self-Supervised Monocular Depth Estimation (ICCV 2019)"  
This repo follow up new KITTI depth bechmark split (But except Person videos)  
- [Digging into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260)  
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
![image](https://github.com/Doyosae/Digging_Into_Self-Supervised_Monocular_Depth_Estimation/blob/main/sample/image.gif)  
![disp](https://github.com/Doyosae/Digging_Into_Self-Supervised_Monocular_Depth_Estimation/blob/main/sample/disp.gif)  
# Folder  
```
dataset/
    2011_09_26/
    ...
    ...
model_loader/
model_layer/
model_loss/
model_tool/
model_save/
model_utility.py
model_option.py
model_train.py
model_test.py
```
# Usage
### Setup KITTI Dataset
1. Download Raw KITTI Dataset
```
wget -i splits_kitti/archives2download.txt -P dataset/  
```
2. Unzip Dataset
```
cd dataset
unzip "*.zip"
```
### Install pakages
1. Install moreutils and parallel  
```
apt-get update -y
apt-get install moreutils
or
apt-get install -y moreutils
```
2. Convert from png to jpg
```
find dataset/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
### Train and Test
1. Training
```
python model_train.py --pose_type separate --datatype kitti_eigen_zhou
python model_train.py --pose_type separate --datatype kitti_benchmark
```
2. Test
```
python model_test.py
```
# Evaluation
```
kitti eigen splits test
monodepth2 original
abs_rel   sqrt_rel  rmse      rmse_log  a1        a2        a3
0.116     0.939     4.904     0.194     0.877     0.959     0.981

my code
abs_rel   sqrt_rel  rmse      rmse_log  a1        a2        a3
0.116     0.906     4.884     0.193     0.874     0.958     0.981
```
# Reference  
[Offical Code](https://github.com/nianticlabs/monodepth2)  
[KITTI Dataset](https://github.com/Doyosae/KITTIDataset)  
[Cityscapes Dataset](https://github.com/Doyosae/CityscapesDataset)
