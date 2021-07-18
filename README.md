# Intro (NOT official)
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
model_dataloader/
model_layer/
model_loss/
model_save/
model_test.py
model_train.py
model_parser.py
model_utility.py
```
# Usage
1. Download Raw KITTI Dataset
```
wget -i splits_kitti/archives2download.txt -P dataset/  
```
2. Unzip Dataset
```
cd dataset
unzip "*.zip"
```
3. Install moreutils and parallel  
```
apt-get update -y
apt-get install moreutils
or
apt-get install -y moreutils
```
4. Convert from png to jpg
```
find dataset/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
5. Train model
```
python model_train.py --datatype kitti_eigen_zhou --pose_type separate

python model_train.py --datatype kitti_benchmark --pose_type separate
```
6. Test model
```
(for kitti_eigen_benchmark)
python model_test.py
```
# Eval
```
kitti_eigen_test
abs_rel 0.115  sq_rel 0.828  rmse 4.695  rmse_log 0.200  a1 0.882  a2 0.961  a3 0.981

kitti_eigen_benchmark_test
abs_rel 0.115  sq_rel 0.818  rmse 4.675  rmse_log 0.198  a1 0.879  a2 0.962  a3 0.981
```
# Reference  
[Offical Code](https://github.com/nianticlabs/monodepth2)  
[KITTI Dataset](https://github.com/Doyosae/KITTIDataset)  
[Cityscapes Dataset](https://github.com/Doyosae/CityscapesDataset)
