# Plate_Recognition_lprnet

## 简介

本项目在RK3588上部署车牌识别任务，将为分为三个工程部分

使用yolov5进行车牌检测-Plate_Detection_yolov5

使用PLRNet进行车牌识别-Plate_Recognition_lprnet

使用RKNN-toolkit进行模型转换推理部署-PLR_rknn

## 准备工作

### 创建conda虚拟环境

```
conda create --name plr python=3.7.11
```

### 激活环境

```
conda activate plr
```

下载仓库源文件，本项目的yolo模型基于yolov5的[956be8e642b5c10af4a1533e09084ca32ff4f21f](https://github.com/ultralytics/yolov5.git)版本

### 安装依赖环境

```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple 
```

### 准备数据集

本实验使用的数据来自[CCPD](https://github.com/detectRecog/CCPD)，下载的数据集为CCPD2019，用于蓝牌为主的车牌检测（电车车牌检测可以下载CCPD2020）。车牌的数据集的组成结构如下表。

| file folder name |    description     | number |
| :--------------: | :----------------: | :----: |
|    ccpd_blur     |      车牌模糊      | 20611  |
|  ccpd_challenge  |    车牌难以检测    | 50003  |
|     ccpd_db      |    车牌曝光不一    | 10132  |
|     ccpd_fn      |    车牌距离差大    | 20967  |
|     ccpd_np      |       无车牌       |  3036  |
|   ccpd_rotate    |     车牌旋转大     | 10053  |
|    ccpd_tilt     |     车牌倾角大     | 30216  |
|   ccpd_weather   | 车牌在雨雪雾天气中 |  9999  |

数据集文件命名如下

```
0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg
车牌区域占整个画面的比例-水平倾角_垂直倾角-车牌左上角右下角-车牌四个角点-车牌号码-亮度-模糊度.jpg
其中车牌号码由以下字典组成
省份：[“皖”, “沪”, “津”, “渝”, “冀”, “晋”, “蒙”, “辽”, “吉”, “黑”, “苏”, “浙”, “京”, “闽”, “赣”, “鲁”, “豫”, “鄂”, “湘”, “粤”, “桂”, “琼”, “川”, “贵”, “云”, “藏”, “陕”, “甘”, “青”, “宁”, “新”]
地市：[‘A’, ‘B’, ‘C’, ‘D’, ‘E’, ‘F’, ‘G’, ‘H’, ‘J’, ‘K’, ‘L’, ‘M’, ‘N’, ‘P’, ‘Q’, ‘R’, ‘S’, ‘T’, ‘U’, ‘V’, ‘W’,‘X’, ‘Y’, ‘Z’]
车牌字典：[‘A’, ‘B’, ‘C’, ‘D’, ‘E’, ‘F’, ‘G’, ‘H’, ‘J’, ‘K’, ‘L’, ‘M’, ‘N’, ‘P’, ‘Q’, ‘R’, ‘S’, ‘T’, ‘U’, ‘V’, ‘W’, ‘X’,‘Y’, ‘Z’, ‘0’, ‘1’, ‘2’, ‘3’, ‘4’, ‘5’, ‘6’, ‘7’, ‘8’, ‘9’]
```

### 数据集解析

为了获得训练PLRNET的训练文件，对每一张图片进行解析。

```
cd plrnet
python tools/ccpd2lpr.py
```

得到所有图片裁剪后区域以及重命名得到的PLRNET训练文件后，进行数据集划分

```
python tools/split_train_val.py
```

最终格式如下图

```
- datasets
| - croped_CCPD
  |	- train
    | - images
  |	- val
| - 数据集2
- data
- models
```

PLRNET使用重命名后文件进行训练

```
京N3FT96.jpg
```

## 训练模型

使用划分好的数据集进行训练

```
python train_lprnet.py
```

## 验证模型

推理

```
python detect.py --weights /runs/train/exp1/best.pt --source /your_val_source
```

转onnx

```
python export_lprnet.py --weights /runs/train/exp1/best.pth
```

最终得到用于其他平台上部署的onnx模型。