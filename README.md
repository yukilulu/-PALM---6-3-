# AI-Studio-项目标题

## 项目描述
简要描述项目

## 项目结构
```
-|data
-|work
-README.MD
-飞桨常规赛：PALM眼底彩照视盘探测与分割 - 6月第3名方案.ipynb
```
### 飞桨常规赛：PALM眼底彩照视盘探测与分割
### [链接地址](https://aistudio.baidu.com/aistudio/competition/detail/87)
常规赛简介

飞桨（PaddlePaddle）以百度多年的深度学习技术研究和业务应用为基础，是中国首个开源开放、技术领先、功能完备的产业级深度学习平台。更多飞桨资讯，点击此处查看。

飞桨常规赛由百度飞桨于 2019 年发起，面向全球 AI 开发者，赛题范围广，涵盖领域多。常规赛旨在通过长期发布的经典比赛项目，为开发者提供学习锻炼机会，助力大家在飞桨大赛中获得骄人成绩。

参赛选手需使用飞桨框架，基于特定赛题下的真实行业数据完成并提交任务。常规赛采取月度评比方式，为打破历史最高记录选手和当月有资格参与月度评奖的前 10 名选手提供飞桨特别礼包奖励。更多惊喜，更多收获，尽在飞桨常规赛。

赛题介绍
本赛题原型为ISBI2019PALM眼科大赛。 近视已成为全球公共卫生负担。在近视患者中，约35%为高度近视。近视导致眼轴长度的延长，可能引起视网膜和脉络膜的病理改变。随着近视屈光度的增加，高度近视将发展为病理性近视，其特点是病理改变的形成:(1)后极，包括镶嵌型眼底、后葡萄肿、视网膜脉络膜变性等;(2)视盘，包括乳头旁萎缩、倾斜等;(3)近视性黄斑，包括漆裂、福氏斑、CNV等。病理性近视对患者造成不可逆的视力损害。因此，早期诊断和定期随访非常重要。
![](https://ai-studio-static-online.cdn.bcebos.com/27fabb47f8af452087086140c147338180bcea215bee48c1b5c7f6e270c8ff2d)
视网膜由黄斑向鼻侧约3mm处有一直径约1.5mm、境界清楚的淡红色圆盘状结构，称为视神经盘，简称视盘。视盘是眼底图像的一个重要特征，对其进行准确、快速地定位与分割对利用眼底图像进行疾病辅助诊断具有重要意义。
 ### 比赛任务
该任务目的是对眼底图像的视盘进行检测，若存在视盘结构，需从眼底图像中分割出视盘区域；若无视盘结构，分割结果直接置全背景。
![](https://ai-studio-static-online.cdn.bcebos.com/c584ee43a27947b6b8908464fa1d90386490eca0ddcd43ae86d9e5251e569071)

### 数据集介绍
本次常规赛提供的金标准由中山大学中山眼科中心的7名眼科医生手工进行视盘像素级标注，之后由另一位高级专家将它们融合为最终的标注结果。存储为BMP图像，与对应的眼底图像大小相同，标签为0代表视盘（黑色区域）；标签为255代表其他（白色区域）。

训练数据集

文件名称：Train
Train文件夹里有fundus_images文件夹和Disc_Masks文件夹。

fundus_images文件夹内包含800张眼底彩照，分辨率为1444×1444，或2124×2056。命名形如H0001.jpg、N0001.jpg、P0001.jpg和V0001.jpg。

Disc_Masks文件夹内包含fundus_images里眼底彩照的视盘分割金标准，大小与对应的眼底彩照一致。命名前缀和对应的fundus_images文件夹里的图像命名一致，后缀为bmp。

测试数据集

文件名称：PALM-Testing400-Images

包含400张眼底彩照，命名形如T0001.jpg。

### 比赛思路

使用UnetAttention进行检测，得到预测结果后，如果预测结果出现多个不连通的区域，通过面积筛选保留最大的面积。
!git clone -b release/v0.6.0 https://gitee.com/paddlepaddle/PaddleSeg.git 

# 下载依赖项，保证PaddleSeg正常运行
%cd PaddleSeg
%pwd
!pip install -r requirements.txt

#解压数据
!unzip -o data/data86770/seg.zip -d /home/aistudio/work

### 生成train.txt 和val.txt
import random
import os
random.seed(2020)
mask_dir  = '/home/aistudio/work/seg/Train/masks'
img_dir = '/home/aistudio/work/seg/Train/fundus_image'
path_list = list()
for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img)
    mask_path = os.path.join(mask_dir,img.replace('jpg', 'png'))
    path_list.append((img_path, mask_path))
random.shuffle(path_list)
ratio = 0.7
train_f = open('/home/aistudio/work/seg/Train/train.txt','w') 
val_f = open('/home/aistudio/work/seg/Train/val.txt' ,'w')

for i ,content in enumerate(path_list):
    img, mask = content
    text = img + ' ' + mask + '\n'
    if i < len(path_list) * ratio:
        train_f.write(text)
    else:
        val_f.write(text)
train_f.close()
val_f.close()

### 配置文件如下
#### batch_size可以适当调大
```
batch_size: 4
iters: 16000

train_dataset:
  type: Dataset
  dataset_root: /home/aistudio/work/seg/Train/
  train_path: /home/aistudio/work/seg/Train/train.txt
  num_classes: 2
  transforms:
    - type: Resize
      target_size: [512, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
    - type: Normalize
  mode: train

val_dataset:
  type: Dataset
  dataset_root: /home/aistudio/work/seg/Train/
  val_path: /home/aistudio/work/seg/Train/val.txt
  num_classes: 2
  transforms:
    - type: Resize
      target_size: [512, 512]
    - type: Normalize
  mode: val


optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

learning_rate:
  value: 0.00125
  decay:
    type: poly
    power: 0.9
    end_lr: 0.0

loss:
  types:
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: DiceLoss
      coef: [4.0, 2.0]
  coef: [1]

model:
  type: AttentionUNet
  num_classes: 2
  pretrained: attentionunet_13000.pdparams

```

### 开始训练
%cd /home/aistudio/PaddleSeg

### 验证
!python val.py --config configs/attentionunet_PALM.yml  --model_path output_attentionunet_PALMoutput/best_model/model.pdparams 

### 预测
!python predict.py \
       --config configs/attentionunet_PALM.yml \
       --model_path output_attentionunet_PALMoutput/best_model/model.pdparams \
       --image_path /home/aistudio/work/seg/test \
       --save_dir output_attentionunet_PALMoutput/result

### 生成结果
import os 
import cv2
result_path = '/home/aistudio/PaddleSeg/output_attentionunet_PALMoutput/result/pseudo_color_prediction'
dist_path = '/home/aistudio/Disc_Segmentation'
for img_name in os.listdir(result_path):
    img_path = os.path.join(result_path, img_name)
    img = cv2.imread(img_path)
    g  = img[:,:,1]
    ret, result = cv2.threshold(g, 127,255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(dist_path,img_name), result)

### 假如预测中出现多个不连通的区域，只保留最大的区域
import os 
import cv2
import matplotlib.pyplot as plt
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

result_path = '/home/aistudio/PaddleSeg/output_attentionunet_PALMoutput/result/pseudo_color_prediction'
dist_path = '/home/aistudio/Disc_Segmentation'
for img_name in os.listdir(result_path):
    img_path = os.path.join(result_path, img_name)
    img = cv2.imread(img_path)
    g  = img[:,:,1]
    ret, threshold = cv2.threshold(g, 127,255, cv2.THRESH_BINARY)


    contours, hierarch = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=cnt_area, reverse=True)
    if len(contours) > 1:
        for i in range(1,len(contours)):
            cv2.drawContours(threshold, [contours[i]], 0, 0, -1)
    _,result = cv2.threshold(threshold, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(dist_path, img_name), result)
    
### 总结
本次参数
模型构建思路及调优过程（可具体包括：思路框架图、思路步骤详述、模型应用+调优过程）

【模型】UnetAttention

【数据增强】图片大小512x512，水平翻转，对比度随机改变等数据增强

【对预测结果进行处理】假如预测中出现多个不连通的区域，只保留最大的区域

【提高】其实可以尝试多一些数据增强，但是我在本次中没有尝试其他的

【参考】https://aistudio.baidu.com/aistudio/projectdetail/2184492?forkThirdPart=1
