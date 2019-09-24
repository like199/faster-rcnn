# Faster-Rcnn

# README
# 使用TensorFlow Object Detection API进行目标检测
本文介绍如何在TensorFlow中训练faster-Rcnn进行人脸识别，并在CoCo数据集上对训练好的网络进行验证。<br />说明：

- Object Detection API 来自[models](https://github.com/tensorflow/models)
- faster-Rcnn训练模型来自[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- faster-Rcnn论文见：[Faster-Rcnn](https://arxiv.org/abs/1506.01497)
## TensorFlow Object Detection API的构建
构建一镜像并搭建tensorFlow Object Detection API（基础镜像tensorflow:19.07-py3）
```bash
docker build docker build docker build -t benchmark/faster-rcnn:v1 
```
启动一个容器：
```bash
docker run -it \
-v /host_dataset:/dataset \
-v /host_model:/model \
-v /host_output/faster-rcnn:/output/faster-rcnn \
--net="host" benchmark/faster-rcnn:v1 /bin/bash
```
可以使用`-v`参数，分别将宿主机上的数据集目录，模型目录，输出结果目录挂在到docker中。

## [](https://github.com/xxmyjk/modelzoo/tree/tf-facenet/tensorflow/face_recognition/facenet#%E5%87%86%E5%A4%87%E6%95%B0%E6%8D%AE%E9%9B%86)数据集准备：
请下载mscoco2017数据集<br />分别下载如下内容：<br />[train_images](http://images.cocodataset.org/zips/train2017.zip),[val_images](http://images.cocodataset.org/zips/val2017.zip),[test_images](http://images.cocodataset.org/zips/test2017.zip),[annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)<br />在/dataset目录下新建mscoco_2017 并按照如下文件目录存放文件<br />/dataset<br />└──/mscoco_2017<br />├──/annotations<br />│    ├──instances_train2017.json,instances_val2017.json,sample_10_instances_val2017.json<br />│    └──sample_1_instances_train2017.json,sample_1_instances_val2017.json<br />├──/test2017<br />│    └──测试集图片<br />├──/train2017<br />│    └──训练集图片<br />└──/val2017<br />└──验证集图片
## 数据预处理：
进入 ~/scripts/coco 运行  sh create_coco_tfrecord.sh<br />tfrecord 格式的文件将存储在/dataset/tfrecord/coco目录下
## 模型准备
从[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)中选择需要训练的模型。
## 训练和验证

### 设置环境变量
首先，请在镜像里设置相应的环境变量,示例如下：
```bash
export MODEL_NAME="faster_rcnn_inception_v2_coco"
export BATCH_SIZE=1
export TRAIN_DATA_FILE='"/dataset/tfrecord/coco/coco_train.record-?????-of-00100"'
export LABEL_FILE='"/dataset/tfrecord/coco/mscoco_label_map.pbtxt"'
export EVAL_DATA_FILE='"/dataset/tfrecord/coco/coco_val.record-?????-of-00010"'
export MODEL_DIR='"/model/mask_rcnn_inception_v2_coco/model.ckpt"'
```
- MODEL_NAME：欲训练/评估的模型名称
- BATCH_SIZE：batch_size数
- TRAIN_DATA_FILE：训练数据集
- LABEL_FILE：数据的标签
- EVAL_DATA_FILE：评估数据集
- MODEL_DIR：欲训练/评估模型路径
### [](https://github.com/xxmyjk/modelzoo/tree/tf-facenet/tensorflow/face_recognition/facenet#%E8%AE%BE%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)[](https://github.com/xxmyjk/modelzoo/tree/tf-facenet/tensorflow/face_recognition/facenet#%E7%94%A8casia-webface%E6%95%B0%E6%8D%AE%E9%9B%86%E8%AE%AD%E7%BB%83facenet%E7%BD%91%E7%BB%9C)用CoCo数据集训练faster-Rcnn网络
运行训练的命令为：

单机单卡：
```bash
cd ~ && sh scripts/train_1n1c.sh 
```
单机四卡
```bash
cd ~ && sh scripts/train_1n4c.sh 
```
### 在CoCo数据集上进行评估

评估模型：
```bash
cd ~ && sh scripts/eval.sh 
```
```
由于tf默认占用所有的显卡，因此要指定显卡运行程序时需要使用nvidia驱动的参数"CUDA_VISIBLE_DEVICES= "，比如：- CUDA_VISIBLE_DEVICES=0 用0号显卡。 


