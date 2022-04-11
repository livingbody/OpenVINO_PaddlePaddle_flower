# 一、鲜花识别
## 1.数据集简介

Oxford 102 Flowers Dataset 是一个花卉集合数据集，主要用于图像分类，它分为 102 个类别共计 102 种花，其中每个类别包含 40 到 258 张图像。

该数据集由牛津大学工程科学系于 2008 年发布，相关论文有《Automated flower classification over a large number of classes》。


![](https://ai-studio-static-online.cdn.bcebos.com/04ce6cfe9c5045a88060ec9c52b263cac308ba10c6c5488a9c1302f2357545e9)

在文件夹下已经生成用于训练和测试的三个.txt文件：train.txt（训练集，1020张图）、valid.txt（验证集，1020张图）、test.txt(6149)。文件中每行格式：图像相对路径 图像的label_id（注意：中间有空格）。

## 2.PaddleClas简介
PaddleClas目前已经是 release2.3了，和以前有脱胎换骨的差别，所以需要重新熟悉。

地址: [https://gitee.com/paddlepaddle/PaddleClas](https://gitee.com/paddlepaddle/PaddleClas)

configs已经移动到了ppcls目录
部署为单独的deploy目录


![](https://ai-studio-static-online.cdn.bcebos.com/4463dc1148a0474fb177d690b93f0d98a37522eef38c498abe73add218635bdd)

![](https://ai-studio-static-online.cdn.bcebos.com/7131057f06f346199371c377bf027eb8afb268a37ab445148dec1b2001d00cff)

## 3.OpenVINO 2022.1 部署支持
OpenVINO™ 是开源的AI预测部署工具箱，支持多种格式，对飞桨支持友好，目前无需转换即可使用。

![](https://ai-studio-static-online.cdn.bcebos.com/86a03e270e4c430e8bf05fb4ad58ac8e72a947e37e1348f38d96fea7d25c5382)

## 4.OpenVINO 2022.1 工作流程
![](https://ai-studio-static-online.cdn.bcebos.com/00d8525cd0a947eb844363dcdaeee3be1940f622039245e6a5e3fd6bef1ecaea)



```python
# 解压缩数据集
!tar -xvf  data/data19852/flowers102.tar -C ./data/ >log.log
```

# 二、PaddleClas准备


```python
# 下载最新版
!git clone https://gitee.com/paddlepaddle/PaddleClas/ --depth=1
```


```python
%cd PaddleClas/
!pip install -r requirements.txt >log.log
```

    /home/aistudio/PaddleClas


# 三、OpenVINO安装


```python
!pip uninstall openvino -y
```

    [33mWARNING: Skipping openvino as it is not installed.[0m[33m
    [0m


```python
!pip install openvino-dev==2022.1.0
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting openvino-dev==2022.1.0
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/f2/99/9e55ddb1abce5ff7def768470810044a6de48a5da99b9195498ad75dfe8a/openvino_dev-2022.1.0-7019-py3-none-any.whl (5.8 MB)
    Collecting networkx~=2.6
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/e9/93/aa6613aa70d6eb4868e667068b5a11feca9645498fd31b954b6c4bb82fa5/networkx-2.6.3-py3-none-any.whl (1.9 MB)
    Collecting imagecodecs~=2021.11.20
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/98/d8/ddeda8911128955d911698e9df6127ddf62937be62d1fad2102a2521d677/imagecodecs-2021.11.20-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (31.0 MB)
    Collecting pyclipper>=1.2.1
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/c5/fa/2c294127e4f88967149a68ad5b3e43636e94e3721109572f8f17ab15b772/pyclipper-1.3.0.post2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (603 kB)
    Collecting numpy<1.20,>=1.16.6
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/08/d6/a6aaa29fea945bc6c61d11f6e0697b325ff7446de5ffd62c2fa02f627048/numpy-1.19.5-cp37-cp37m-manylinux2010_x86_64.whl (14.8 MB)
    Collecting requests>=2.25.1
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/2d/61/08076519c80041bc0ffa1a8af0cbd3bf3e2b62af10435d269a9d0f40564d/requests-2.27.1-py2.py3-none-any.whl (63 kB)
    Collecting progress>=1.5
      Using cached https://pypi.tuna.tsinghua.edu.cn/packages/2a/68/d8412d1e0d70edf9791cbac5426dc859f4649afc22f2abbeb0d947cf70fd/progress-1.6.tar.gz (7.8 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [31mERROR: Could not find a version that satisfies the requirement openvino==2022.1.0 (from openvino-dev) (from versions: 2021.2, 2021.3.0, 2021.4.0, 2021.4.1, 2021.4.2)[0m[31m
    [0m[31mERROR: No matching distribution found for openvino==2022.1.0[0m[31m
    [0m[?25h


```python
!pip install -U pip --user >log.log
!pip install openvino==2022.1.0
#  -i https://pypi.org/simple
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    [31mERROR: Could not find a version that satisfies the requirement openvino==2022.1 (from versions: 2021.2, 2021.3.0, 2021.4.0, 2021.4.1, 2021.4.2)[0m[31m
    [0m[31mERROR: No matching distribution found for openvino==2022.1[0m[31m
    [0m


```python
!pip list|grep openvino
```

# 三、模型训练

## 1.修改imagenet_dataset.py
目录： \ppcls\data\dataloader\imagenet_dataset.py

修改原因是目录这块存在bug，注释：
* assert os.path.exists(self._cls_path)
* assert os.path.exists(self._img_root)

添加

* self._cls_path=os.path.join(self._img_root,self._cls_path)

否则不能使用相对路径

```
class ImageNetDataset(CommonDataset):
    def _load_anno(self, seed=None):
        会对目录进行检测，如果cls_path使用相对目录，就会报错，在此注释掉，并修改为self._cls_path=os.path.join(self._img_root,self._cls_path)
        # assert os.path.exists(self._cls_path)
        # assert os.path.exists(self._img_root)
        self._cls_path=os.path.join(self._img_root,self._cls_path)
        print('self._cls_path',self._cls_path)
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(" ")
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(int(l[1]))
                assert os.path.exists(self.images[-1])
```

## 2.修改配置文件



```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # gpu或cpu配置
  device: gpu
  # 分类数量
  class_num: 102
  # 保存间隔
  save_interval: 5
  # 是否再训练立案过程中进行eval
  eval_during_train: True
  # eval间隔
  eval_interval: 5
  # 训练轮数
  epochs: 20
  # 打印batch step设置
  print_batch_step: 10
  # 是否使用visualdl
  use_visualdl: False
  # used for static mode and model export
  image_shape: [3, 224, 224]
  # 保存地址
  save_inference_dir: ./inference

# model architecture
Arch:
  name: ResNet50_vd
 
# loss function config for traing/eval process
Loss:
  Train:
    - CELoss:
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Cosine
    learning_rate: 0.0125
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    coeff: 0.00001


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
      cls_label_path: train.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 256
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      image_root: /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
      cls_label_path: valid.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 256
      drop_last: False
      shuffle: False
    loader:
      num_workers: 4
      use_shared_memory: True

Infer:
  infer_imgs: /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    topk: 5
    class_id_map_file: /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00030.jpg

Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]

```

* -c 参数是指定训练的配置文件路径，训练的具体超参数可查看yaml文件
* yaml文Global.device 参数设置为cpu，即使用CPU进行训练（若不设置，此参数默认为True）
* yaml文件中epochs参数设置为20，说明对整个数据集进行20个epoch迭代，预计训练20分钟左右（不同CPU，训练时间略有不同），此时训练模型不充分。若提高训练模型精度，请将此参数设大，如40，训练时间也会相应延长

## 3.配置说明
### 3.1 全局配置(Global)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| checkpoints | 断点模型路径，用于恢复训练 | null | str |
| pretrained_model | 预训练模型路径 | null | str |
| output_dir | 保存模型路径 | "./output/" | str |
| save_interval | 每隔多少个epoch保存模型 | 1 | int |
| eval_during_train| 是否在训练时进行评估 | True | bool |
| eval_interval | 每隔多少个epoch进行模型评估 | 1 | int |
| epochs | 训练总epoch数 |  | int |
| print_batch_step | 每隔多少个mini-batch打印输出 | 10 | int |
| use_visualdl | 是否是用visualdl可视化训练过程 | False | bool |
| image_shape | 图片大小 | [3，224，224] | list, shape: (3,) |
| save_inference_dir | inference模型的保存路径 | "./inference" | str |
| eval_mode | eval的模式 | "classification" | "retrieval" |

### 3.2 结构(Arch)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 模型结构名字 | ResNet50 | PaddleClas提供的模型结构 |
| class_num | 分类数 | 1000 | int |
| pretrained | 预训练模型 | False | bool， str |

### 3.3 损失函数（Loss）

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| CELoss | 交叉熵损失函数 | —— | —— |
| CELoss.weight | CELoss的在整个Loss中的权重 | 1.0 | float |
| CELoss.epsilon | CELoss中label_smooth的epsilon值 | 0.1 | float，0-1之间 |

### 3.4 优化器(Optimizer)

| 参数名字 | 具体含义 | 默认值 | 可选值 |
|:---:|:---:|:---:|:---:|
| name | 优化器方法名 | "Momentum" | "RmsProp"等其他优化器 |
| momentum | momentum值 | 0.9 | float |
| lr.name | 学习率下降方式 | "Cosine" | "Linear"、"Piecewise"等其他下降方式 |
| lr.learning_rate | 学习率初始值 | 0.1 | float |
| lr.warmup_epoch | warmup轮数 | 0 | int，如5 |
| regularizer.name | 正则化方法名 | "L2" | ["L1", "L2"] |
| regularizer.coeff | 正则化系数 | 0.00007 | float |



## 4.训练


```python
!pwd
!cp ~/ResNet50_vd.yaml  ./ppcls/configs/quick_start/ResNet50_vd.yaml 
!cp ~/imagenet_dataset.py ./ppcls/data/dataloader/imagenet_dataset.py
```

    /home/aistudio/PaddleClas



```python
# GPU设置
!export CUDA_VISIBLE_DEVICES=0

# -o Arch.pretrained=True 使用预训练模型，当选择为True时，预训练权重会自动下载到本地
!python tools/train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml -o Arch.pretrained=True
```

    A new filed (pretrained) detected!
    [2022/04/04 17:51:03] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2022/04/04 17:51:03] root INFO: Arch : 
    [2022/04/04 17:51:03] root INFO:     name : ResNet50_vd
    [2022/04/04 17:51:03] root INFO:     pretrained : True
    [2022/04/04 17:51:03] root INFO: DataLoader : 
    [2022/04/04 17:51:03] root INFO:     Eval : 
    [2022/04/04 17:51:03] root INFO:         dataset : 
    [2022/04/04 17:51:03] root INFO:             cls_label_path : valid.txt
    [2022/04/04 17:51:03] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/04 17:51:03] root INFO:             name : ImageNetDataset
    [2022/04/04 17:51:03] root INFO:             transform_ops : 
    [2022/04/04 17:51:03] root INFO:                 DecodeImage : 
    [2022/04/04 17:51:03] root INFO:                     channel_first : False
    [2022/04/04 17:51:03] root INFO:                     to_rgb : True
    [2022/04/04 17:51:03] root INFO:                 ResizeImage : 
    [2022/04/04 17:51:03] root INFO:                     resize_short : 256
    [2022/04/04 17:51:03] root INFO:                 CropImage : 
    [2022/04/04 17:51:03] root INFO:                     size : 224
    [2022/04/04 17:51:03] root INFO:                 NormalizeImage : 
    [2022/04/04 17:51:03] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2022/04/04 17:51:03] root INFO:                     order : 
    [2022/04/04 17:51:03] root INFO:                     scale : 1.0/255.0
    [2022/04/04 17:51:03] root INFO:                     std : [0.229, 0.224, 0.225]
    [2022/04/04 17:51:03] root INFO:         loader : 
    [2022/04/04 17:51:03] root INFO:             num_workers : 4
    [2022/04/04 17:51:03] root INFO:             use_shared_memory : True
    [2022/04/04 17:51:03] root INFO:         sampler : 
    [2022/04/04 17:51:03] root INFO:             batch_size : 128
    [2022/04/04 17:51:03] root INFO:             drop_last : False
    [2022/04/04 17:51:03] root INFO:             name : DistributedBatchSampler
    [2022/04/04 17:51:03] root INFO:             shuffle : False
    [2022/04/04 17:51:03] root INFO:     Train : 
    [2022/04/04 17:51:03] root INFO:         dataset : 
    [2022/04/04 17:51:03] root INFO:             cls_label_path : train.txt
    [2022/04/04 17:51:03] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/04 17:51:03] root INFO:             name : ImageNetDataset
    [2022/04/04 17:51:03] root INFO:             transform_ops : 
    [2022/04/04 17:51:03] root INFO:                 DecodeImage : 
    [2022/04/04 17:51:03] root INFO:                     channel_first : False
    [2022/04/04 17:51:03] root INFO:                     to_rgb : True
    [2022/04/04 17:51:03] root INFO:                 RandCropImage : 
    [2022/04/04 17:51:03] root INFO:                     size : 224
    [2022/04/04 17:51:03] root INFO:                 RandFlipImage : 
    [2022/04/04 17:51:03] root INFO:                     flip_code : 1
    [2022/04/04 17:51:03] root INFO:                 NormalizeImage : 
    [2022/04/04 17:51:03] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2022/04/04 17:51:03] root INFO:                     order : 
    [2022/04/04 17:51:03] root INFO:                     scale : 1.0/255.0
    [2022/04/04 17:51:03] root INFO:                     std : [0.229, 0.224, 0.225]
    [2022/04/04 17:51:03] root INFO:         loader : 
    [2022/04/04 17:51:03] root INFO:             num_workers : 4
    [2022/04/04 17:51:03] root INFO:             use_shared_memory : True
    [2022/04/04 17:51:03] root INFO:         sampler : 
    [2022/04/04 17:51:03] root INFO:             batch_size : 128
    [2022/04/04 17:51:03] root INFO:             drop_last : False
    [2022/04/04 17:51:03] root INFO:             name : DistributedBatchSampler
    [2022/04/04 17:51:03] root INFO:             shuffle : True
    [2022/04/04 17:51:03] root INFO: Global : 
    [2022/04/04 17:51:03] root INFO:     checkpoints : None
    [2022/04/04 17:51:03] root INFO:     class_num : 102
    [2022/04/04 17:51:03] root INFO:     device : gpu
    [2022/04/04 17:51:03] root INFO:     epochs : 20
    [2022/04/04 17:51:03] root INFO:     eval_during_train : True
    [2022/04/04 17:51:03] root INFO:     eval_interval : 5
    [2022/04/04 17:51:03] root INFO:     image_shape : [3, 224, 224]
    [2022/04/04 17:51:03] root INFO:     output_dir : ./output/
    [2022/04/04 17:51:03] root INFO:     pretrained_model : None
    [2022/04/04 17:51:03] root INFO:     print_batch_step : 10
    [2022/04/04 17:51:03] root INFO:     save_inference_dir : ./inference
    [2022/04/04 17:51:03] root INFO:     save_interval : 5
    [2022/04/04 17:51:03] root INFO:     use_visualdl : False
    [2022/04/04 17:51:03] root INFO: Infer : 
    [2022/04/04 17:51:03] root INFO:     PostProcess : 
    [2022/04/04 17:51:03] root INFO:         class_id_map_file : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00030.jpg
    [2022/04/04 17:51:03] root INFO:         name : Topk
    [2022/04/04 17:51:03] root INFO:         topk : 5
    [2022/04/04 17:51:03] root INFO:     batch_size : 10
    [2022/04/04 17:51:03] root INFO:     infer_imgs : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/04 17:51:03] root INFO:     transforms : 
    [2022/04/04 17:51:03] root INFO:         DecodeImage : 
    [2022/04/04 17:51:03] root INFO:             channel_first : False
    [2022/04/04 17:51:03] root INFO:             to_rgb : True
    [2022/04/04 17:51:03] root INFO:         ResizeImage : 
    [2022/04/04 17:51:03] root INFO:             resize_short : 256
    [2022/04/04 17:51:03] root INFO:         CropImage : 
    [2022/04/04 17:51:03] root INFO:             size : 224
    [2022/04/04 17:51:03] root INFO:         NormalizeImage : 
    [2022/04/04 17:51:03] root INFO:             mean : [0.485, 0.456, 0.406]
    [2022/04/04 17:51:03] root INFO:             order : 
    [2022/04/04 17:51:03] root INFO:             scale : 1.0/255.0
    [2022/04/04 17:51:03] root INFO:             std : [0.229, 0.224, 0.225]
    [2022/04/04 17:51:03] root INFO:         ToCHWImage : None
    [2022/04/04 17:51:03] root INFO: Loss : 
    [2022/04/04 17:51:03] root INFO:     Eval : 
    [2022/04/04 17:51:03] root INFO:         CELoss : 
    [2022/04/04 17:51:03] root INFO:             weight : 1.0
    [2022/04/04 17:51:03] root INFO:     Train : 
    [2022/04/04 17:51:03] root INFO:         CELoss : 
    [2022/04/04 17:51:03] root INFO:             weight : 1.0
    [2022/04/04 17:51:03] root INFO: Metric : 
    [2022/04/04 17:51:03] root INFO:     Eval : 
    [2022/04/04 17:51:03] root INFO:         TopkAcc : 
    [2022/04/04 17:51:03] root INFO:             topk : [1, 5]
    [2022/04/04 17:51:03] root INFO:     Train : 
    [2022/04/04 17:51:03] root INFO:         TopkAcc : 
    [2022/04/04 17:51:03] root INFO:             topk : [1, 5]
    [2022/04/04 17:51:03] root INFO: Optimizer : 
    [2022/04/04 17:51:03] root INFO:     lr : 
    [2022/04/04 17:51:03] root INFO:         learning_rate : 0.0125
    [2022/04/04 17:51:03] root INFO:         name : Cosine
    [2022/04/04 17:51:03] root INFO:         warmup_epoch : 5
    [2022/04/04 17:51:03] root INFO:     momentum : 0.9
    [2022/04/04 17:51:03] root INFO:     name : Momentum
    [2022/04/04 17:51:03] root INFO:     regularizer : 
    [2022/04/04 17:51:03] root INFO:         coeff : 1e-05
    [2022/04/04 17:51:03] root INFO:         name : L2
    [2022/04/04 17:51:03] root INFO: profiler_options : None
    [2022/04/04 17:51:03] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
    [2022/04/04 17:51:03] root WARNING: The Global.class_num will be deprecated. Please use Arch.class_num instead. Arch.class_num has been set to 102.
    self._cls_path /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/train.txt
    self._cls_path /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/valid.txt
    [2022/04/04 17:51:03] root WARNING: 'TopkAcc' metric can not be used when setting 'batch_transform_ops' in config. The 'TopkAcc' metric has been removed.
    W0404 17:51:03.718078   846 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0404 17:51:03.723338   846 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    [2022/04/04 17:51:08] root INFO: unique_endpoints {''}
    [2022/04/04 17:51:08] root INFO: Found /home/aistudio/.paddleclas/weights/ResNet50_vd_pretrained.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1301: UserWarning: Skip loading for fc.weight. fc.weight receives a shape [2048, 1000], but the expected shape is [2048, 102].
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1301: UserWarning: Skip loading for fc.bias. fc.bias receives a shape [1000], but the expected shape is [102].
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2022/04/04 17:51:09] root WARNING: The training strategy in config files provided by PaddleClas is based on 4 gpus. But the number of gpus is 1 in current training. Please modify the stategy (learning rate, batch size and so on) if use config files in PaddleClas to train.
    [2022/04/04 17:51:12] root INFO: [Train][Epoch 1/20][Iter: 0/8]lr: 0.00031, CELoss: 4.64081, loss: 4.64081, batch_cost: 3.14893s, reader_cost: 2.57331, ips: 40.64869 images/sec, eta: 0:08:23
    [2022/04/04 17:51:16] root INFO: [Train][Epoch 1/20][Avg]CELoss: 4.64772, loss: 4.64772
    [2022/04/04 17:51:16] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:51:19] root INFO: [Train][Epoch 2/20][Iter: 0/8]lr: 0.00281, CELoss: 4.61437, loss: 4.61437, batch_cost: 1.05037s, reader_cost: 0.63829, ips: 121.86160 images/sec, eta: 0:02:39
    [2022/04/04 17:51:22] root INFO: [Train][Epoch 2/20][Avg]CELoss: 4.58869, loss: 4.58869
    [2022/04/04 17:51:23] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:51:26] root INFO: [Train][Epoch 3/20][Iter: 0/8]lr: 0.00531, CELoss: 4.55260, loss: 4.55260, batch_cost: 1.19191s, reader_cost: 0.78459, ips: 107.39076 images/sec, eta: 0:02:51
    [2022/04/04 17:51:30] root INFO: [Train][Epoch 3/20][Avg]CELoss: 4.45869, loss: 4.45869
    [2022/04/04 17:51:30] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:51:33] root INFO: [Train][Epoch 4/20][Iter: 0/8]lr: 0.00781, CELoss: 4.32414, loss: 4.32414, batch_cost: 0.95354s, reader_cost: 0.56087, ips: 134.23697 images/sec, eta: 0:02:09
    [2022/04/04 17:51:36] root INFO: [Train][Epoch 4/20][Avg]CELoss: 4.16781, loss: 4.16781
    [2022/04/04 17:51:37] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:51:39] root INFO: [Train][Epoch 5/20][Iter: 0/8]lr: 0.01031, CELoss: 3.81278, loss: 3.81278, batch_cost: 0.97704s, reader_cost: 0.57012, ips: 131.00846 images/sec, eta: 0:02:05
    [2022/04/04 17:51:43] root INFO: [Train][Epoch 5/20][Avg]CELoss: 3.64121, loss: 3.64121
    [2022/04/04 17:51:46] root INFO: [Eval][Epoch 5][Iter: 0/8]CELoss: 3.10458, loss: 3.10458, top1: 0.62500, top5: 0.84375, batch_cost: 2.88314s, reader_cost: 2.69916, ips: 44.39597 images/sec
    [2022/04/04 17:51:49] root INFO: [Eval][Epoch 5][Avg]CELoss: 3.06763, loss: 3.06763, top1: 0.58627, top5: 0.83039
    [2022/04/04 17:51:49] root INFO: Already save model in ./output/ResNet50_vd/best_model
    [2022/04/04 17:51:49] root INFO: [Eval][Epoch 5][best metric: 0.586274507933972]
    [2022/04/04 17:51:50] root INFO: Already save model in ./output/ResNet50_vd/epoch_5
    [2022/04/04 17:51:50] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:51:53] root INFO: [Train][Epoch 6/20][Iter: 0/8]lr: 0.01250, CELoss: 3.23097, loss: 3.23097, batch_cost: 1.12798s, reader_cost: 0.72043, ips: 113.47702 images/sec, eta: 0:02:15
    [2022/04/04 17:51:56] root INFO: [Train][Epoch 6/20][Avg]CELoss: 2.82732, loss: 2.82732
    [2022/04/04 17:51:57] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:00] root INFO: [Train][Epoch 7/20][Iter: 0/8]lr: 0.01233, CELoss: 2.31577, loss: 2.31577, batch_cost: 0.97315s, reader_cost: 0.58090, ips: 131.53180 images/sec, eta: 0:01:48
    [2022/04/04 17:52:03] root INFO: [Train][Epoch 7/20][Avg]CELoss: 2.02123, loss: 2.02123
    [2022/04/04 17:52:04] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:06] root INFO: [Train][Epoch 8/20][Iter: 0/8]lr: 0.01189, CELoss: 1.60169, loss: 1.60169, batch_cost: 0.96181s, reader_cost: 0.56401, ips: 133.08299 images/sec, eta: 0:01:40
    [2022/04/04 17:52:10] root INFO: [Train][Epoch 8/20][Avg]CELoss: 1.38111, loss: 1.38111
    [2022/04/04 17:52:10] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:13] root INFO: [Train][Epoch 9/20][Iter: 0/8]lr: 0.01121, CELoss: 1.10794, loss: 1.10794, batch_cost: 1.00229s, reader_cost: 0.59248, ips: 127.70792 images/sec, eta: 0:01:36
    [2022/04/04 17:52:16] root INFO: [Train][Epoch 9/20][Avg]CELoss: 0.93859, loss: 0.93859
    [2022/04/04 17:52:17] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:20] root INFO: [Train][Epoch 10/20][Iter: 0/8]lr: 0.01031, CELoss: 0.78641, loss: 0.78641, batch_cost: 1.16687s, reader_cost: 0.75940, ips: 109.69474 images/sec, eta: 0:01:42
    [2022/04/04 17:52:23] root INFO: [Train][Epoch 10/20][Avg]CELoss: 0.66623, loss: 0.66623
    [2022/04/04 17:52:27] root INFO: [Eval][Epoch 10][Iter: 0/8]CELoss: 0.75807, loss: 0.75807, top1: 0.85938, top5: 0.95312, batch_cost: 3.28170s, reader_cost: 3.07438, ips: 39.00414 images/sec
    [2022/04/04 17:52:29] root INFO: [Eval][Epoch 10][Avg]CELoss: 0.66149, loss: 0.66149, top1: 0.89118, top5: 0.97647
    [2022/04/04 17:52:30] root INFO: Already save model in ./output/ResNet50_vd/best_model
    [2022/04/04 17:52:30] root INFO: [Eval][Epoch 10][best metric: 0.8911764738606471]
    [2022/04/04 17:52:30] root INFO: Already save model in ./output/ResNet50_vd/epoch_10
    [2022/04/04 17:52:31] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:33] root INFO: [Train][Epoch 11/20][Iter: 0/8]lr: 0.00923, CELoss: 0.42259, loss: 0.42259, batch_cost: 0.98574s, reader_cost: 0.57787, ips: 129.85227 images/sec, eta: 0:01:18
    [2022/04/04 17:52:37] root INFO: [Train][Epoch 11/20][Avg]CELoss: 0.48286, loss: 0.48286
    [2022/04/04 17:52:37] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:40] root INFO: [Train][Epoch 12/20][Iter: 0/8]lr: 0.00803, CELoss: 0.42716, loss: 0.42716, batch_cost: 1.01280s, reader_cost: 0.57595, ips: 126.38195 images/sec, eta: 0:01:12
    [2022/04/04 17:52:44] root INFO: [Train][Epoch 12/20][Avg]CELoss: 0.35100, loss: 0.35100
    [2022/04/04 17:52:44] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:47] root INFO: [Train][Epoch 13/20][Iter: 0/8]lr: 0.00674, CELoss: 0.27117, loss: 0.27117, batch_cost: 0.95948s, reader_cost: 0.56412, ips: 133.40572 images/sec, eta: 0:01:01
    [2022/04/04 17:52:50] root INFO: [Train][Epoch 13/20][Avg]CELoss: 0.31605, loss: 0.31605
    [2022/04/04 17:52:51] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:52:54] root INFO: [Train][Epoch 14/20][Iter: 0/8]lr: 0.00543, CELoss: 0.28394, loss: 0.28394, batch_cost: 1.13772s, reader_cost: 0.72433, ips: 112.50533 images/sec, eta: 0:01:03
    [2022/04/04 17:52:57] root INFO: [Train][Epoch 14/20][Avg]CELoss: 0.26122, loss: 0.26122
    [2022/04/04 17:52:58] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:53:01] root INFO: [Train][Epoch 15/20][Iter: 0/8]lr: 0.00416, CELoss: 0.16402, loss: 0.16402, batch_cost: 1.09271s, reader_cost: 0.68246, ips: 117.14036 images/sec, eta: 0:00:52
    [2022/04/04 17:53:04] root INFO: [Train][Epoch 15/20][Avg]CELoss: 0.21305, loss: 0.21305
    [2022/04/04 17:53:08] root INFO: [Eval][Epoch 15][Iter: 0/8]CELoss: 0.53277, loss: 0.53277, top1: 0.88281, top5: 0.94531, batch_cost: 3.47772s, reader_cost: 3.29045, ips: 36.80568 images/sec
    [2022/04/04 17:53:10] root INFO: [Eval][Epoch 15][Avg]CELoss: 0.42379, loss: 0.42379, top1: 0.91863, top5: 0.98039
    [2022/04/04 17:53:11] root INFO: Already save model in ./output/ResNet50_vd/best_model
    [2022/04/04 17:53:11] root INFO: [Eval][Epoch 15][best metric: 0.9186274477079803]
    [2022/04/04 17:53:11] root INFO: Already save model in ./output/ResNet50_vd/epoch_15
    [2022/04/04 17:53:11] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:53:14] root INFO: [Train][Epoch 16/20][Iter: 0/8]lr: 0.00298, CELoss: 0.30324, loss: 0.30324, batch_cost: 0.98888s, reader_cost: 0.57842, ips: 129.43957 images/sec, eta: 0:00:39
    [2022/04/04 17:53:18] root INFO: [Train][Epoch 16/20][Avg]CELoss: 0.22449, loss: 0.22449
    [2022/04/04 17:53:18] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:53:21] root INFO: [Train][Epoch 17/20][Iter: 0/8]lr: 0.00195, CELoss: 0.19888, loss: 0.19888, batch_cost: 0.95664s, reader_cost: 0.56364, ips: 133.80189 images/sec, eta: 0:00:30
    [2022/04/04 17:53:24] root INFO: [Train][Epoch 17/20][Avg]CELoss: 0.24154, loss: 0.24154
    [2022/04/04 17:53:25] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:53:28] root INFO: [Train][Epoch 18/20][Iter: 0/8]lr: 0.00110, CELoss: 0.26926, loss: 0.26926, batch_cost: 1.14401s, reader_cost: 0.73260, ips: 111.88671 images/sec, eta: 0:00:27
    [2022/04/04 17:53:31] root INFO: [Train][Epoch 18/20][Avg]CELoss: 0.23758, loss: 0.23758
    [2022/04/04 17:53:32] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:53:35] root INFO: [Train][Epoch 19/20][Iter: 0/8]lr: 0.00048, CELoss: 0.31946, loss: 0.31946, batch_cost: 0.94231s, reader_cost: 0.54846, ips: 135.83640 images/sec, eta: 0:00:15
    [2022/04/04 17:53:38] root INFO: [Train][Epoch 19/20][Avg]CELoss: 0.20997, loss: 0.20997
    [2022/04/04 17:53:39] root INFO: Already save model in ./output/ResNet50_vd/latest
    [2022/04/04 17:53:42] root INFO: [Train][Epoch 20/20][Iter: 0/8]lr: 0.00010, CELoss: 0.16348, loss: 0.16348, batch_cost: 1.05688s, reader_cost: 0.64513, ips: 121.11092 images/sec, eta: 0:00:08
    [2022/04/04 17:53:45] root INFO: [Train][Epoch 20/20][Avg]CELoss: 0.20710, loss: 0.20710
    [2022/04/04 17:53:48] root INFO: [Eval][Epoch 20][Iter: 0/8]CELoss: 0.51250, loss: 0.51250, top1: 0.89062, top5: 0.95312, batch_cost: 3.19902s, reader_cost: 2.99184, ips: 40.01228 images/sec
    [2022/04/04 17:53:51] root INFO: [Eval][Epoch 20][Avg]CELoss: 0.40260, loss: 0.40260, top1: 0.92059, top5: 0.98235
    [2022/04/04 17:53:52] root INFO: Already save model in ./output/ResNet50_vd/best_model
    [2022/04/04 17:53:52] root INFO: [Eval][Epoch 20][best metric: 0.9205882329566806]
    [2022/04/04 17:53:52] root INFO: Already save model in ./output/ResNet50_vd/epoch_20
    [2022/04/04 17:53:52] root INFO: Already save model in ./output/ResNet50_vd/latest


训练日志如下

```
[2021/10/31 01:53:47] root INFO: [Train][Epoch 16/20][Iter: 0/4]lr: 0.00285, top1: 0.93750, top5: 0.96484, CELoss: 0.36489, loss: 0.36489, batch_cost: 1.48066s, reader_cost: 0.68550, ips: 172.89543 images/sec, eta: 0:00:29
[2021/10/31 01:53:49] root INFO: [Train][Epoch 16/20][Avg]top1: 0.95098, top5: 0.97745, CELoss: 0.31581, loss: 0.31581
[2021/10/31 01:53:53] root INFO: [Train][Epoch 17/20][Iter: 0/4]lr: 0.00183, top1: 0.94531, top5: 0.97656, CELoss: 0.32916, loss: 0.32916, batch_cost: 1.47958s, reader_cost: 0.68473, ips: 173.02266 images/sec, eta: 0:00:23
[2021/10/31 01:53:55] root INFO: [Train][Epoch 17/20][Avg]top1: 0.95686, top5: 0.98137, CELoss: 0.29560, loss: 0.29560
[2021/10/31 01:53:58] root INFO: [Train][Epoch 18/20][Iter: 0/4]lr: 0.00101, top1: 0.93750, top5: 0.98047, CELoss: 0.31542, loss: 0.31542, batch_cost: 1.47524s, reader_cost: 0.68058, ips: 173.53117 images/sec, eta: 0:00:17
[2021/10/31 01:54:01] root INFO: [Train][Epoch 18/20][Avg]top1: 0.94608, top5: 0.98627, CELoss: 0.29086, loss: 0.29086
[2021/10/31 01:54:04] root INFO: [Train][Epoch 19/20][Iter: 0/4]lr: 0.00042, top1: 0.97266, top5: 0.98438, CELoss: 0.24642, loss: 0.24642, batch_cost: 1.47376s, reader_cost: 0.67916, ips: 173.70590 images/sec, eta: 0:00:11
[2021/10/31 01:54:07] root INFO: [Train][Epoch 19/20][Avg]top1: 0.94608, top5: 0.97941, CELoss: 0.30998, loss: 0.30998
[2021/10/31 01:54:10] root INFO: [Train][Epoch 20/20][Iter: 0/4]lr: 0.00008, top1: 0.98047, top5: 0.98438, CELoss: 0.20209, loss: 0.20209, batch_cost: 1.47083s, reader_cost: 0.67647, ips: 174.05180 images/sec, eta: 0:00:05
[2021/10/31 01:54:13] root INFO: [Train][Epoch 20/20][Avg]top1: 0.95784, top5: 0.98922, CELoss: 0.25974, loss: 0.25974
[2021/10/31 01:54:16] root INFO: [Eval][Epoch 20][Iter: 0/4]CELoss: 0.47912, loss: 0.47912, top1: 0.91797, top5: 0.96094, batch_cost: 3.26175s, reader_cost: 3.02034, ips: 78.48538 images/sec
[2021/10/31 01:54:17] root INFO: [Eval][Epoch 20][Avg]CELoss: 0.54982, loss: 0.54982, top1: 0.88922, top5: 0.96667
[2021/10/31 01:54:18] root INFO: Already save model in ./output/ResNet50_vd/best_model
[2021/10/31 01:54:18] root INFO: [Eval][Epoch 20][best metric: 0.8892156844045601]
[2021/10/31 01:54:18] root INFO: Already save model in ./output/ResNet50_vd/epoch_20
[2021/10/31 01:54:18] root INFO: Already save model in ./output/ResNet50_vd/latest
```

可见日志输出比较混乱，没有以前那么清晰，最好使用visualdl来查看训练情况

# 四、模型导出


```python
!python tools/export_model.py \
    -c ./ppcls/configs/quick_start/ResNet50_vd.yaml \
    -o Global.pretrained_model=./output/ResNet50_vd/best_model \
    -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

    [2022/04/04 18:13:38] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2022/04/04 18:13:38] root INFO: Arch : 
    [2022/04/04 18:13:38] root INFO:     name : ResNet50_vd
    [2022/04/04 18:13:38] root INFO: DataLoader : 
    [2022/04/04 18:13:38] root INFO:     Eval : 
    [2022/04/04 18:13:38] root INFO:         dataset : 
    [2022/04/04 18:13:38] root INFO:             cls_label_path : valid.txt
    [2022/04/04 18:13:38] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/04 18:13:38] root INFO:             name : ImageNetDataset
    [2022/04/04 18:13:38] root INFO:             transform_ops : 
    [2022/04/04 18:13:38] root INFO:                 DecodeImage : 
    [2022/04/04 18:13:38] root INFO:                     channel_first : False
    [2022/04/04 18:13:38] root INFO:                     to_rgb : True
    [2022/04/04 18:13:38] root INFO:                 ResizeImage : 
    [2022/04/04 18:13:38] root INFO:                     resize_short : 256
    [2022/04/04 18:13:38] root INFO:                 CropImage : 
    [2022/04/04 18:13:38] root INFO:                     size : 224
    [2022/04/04 18:13:38] root INFO:                 NormalizeImage : 
    [2022/04/04 18:13:38] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2022/04/04 18:13:38] root INFO:                     order : 
    [2022/04/04 18:13:38] root INFO:                     scale : 1.0/255.0
    [2022/04/04 18:13:38] root INFO:                     std : [0.229, 0.224, 0.225]
    [2022/04/04 18:13:38] root INFO:         loader : 
    [2022/04/04 18:13:38] root INFO:             num_workers : 4
    [2022/04/04 18:13:38] root INFO:             use_shared_memory : True
    [2022/04/04 18:13:38] root INFO:         sampler : 
    [2022/04/04 18:13:38] root INFO:             batch_size : 128
    [2022/04/04 18:13:38] root INFO:             drop_last : False
    [2022/04/04 18:13:38] root INFO:             name : DistributedBatchSampler
    [2022/04/04 18:13:38] root INFO:             shuffle : False
    [2022/04/04 18:13:38] root INFO:     Train : 
    [2022/04/04 18:13:38] root INFO:         dataset : 
    [2022/04/04 18:13:38] root INFO:             cls_label_path : train.txt
    [2022/04/04 18:13:38] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/04 18:13:38] root INFO:             name : ImageNetDataset
    [2022/04/04 18:13:38] root INFO:             transform_ops : 
    [2022/04/04 18:13:38] root INFO:                 DecodeImage : 
    [2022/04/04 18:13:38] root INFO:                     channel_first : False
    [2022/04/04 18:13:38] root INFO:                     to_rgb : True
    [2022/04/04 18:13:38] root INFO:                 RandCropImage : 
    [2022/04/04 18:13:38] root INFO:                     size : 224
    [2022/04/04 18:13:38] root INFO:                 RandFlipImage : 
    [2022/04/04 18:13:38] root INFO:                     flip_code : 1
    [2022/04/04 18:13:38] root INFO:                 NormalizeImage : 
    [2022/04/04 18:13:38] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2022/04/04 18:13:38] root INFO:                     order : 
    [2022/04/04 18:13:38] root INFO:                     scale : 1.0/255.0
    [2022/04/04 18:13:38] root INFO:                     std : [0.229, 0.224, 0.225]
    [2022/04/04 18:13:38] root INFO:         loader : 
    [2022/04/04 18:13:38] root INFO:             num_workers : 4
    [2022/04/04 18:13:38] root INFO:             use_shared_memory : True
    [2022/04/04 18:13:38] root INFO:         sampler : 
    [2022/04/04 18:13:38] root INFO:             batch_size : 128
    [2022/04/04 18:13:38] root INFO:             drop_last : False
    [2022/04/04 18:13:38] root INFO:             name : DistributedBatchSampler
    [2022/04/04 18:13:38] root INFO:             shuffle : True
    [2022/04/04 18:13:38] root INFO: Global : 
    [2022/04/04 18:13:38] root INFO:     checkpoints : None
    [2022/04/04 18:13:38] root INFO:     class_num : 102
    [2022/04/04 18:13:38] root INFO:     device : gpu
    [2022/04/04 18:13:38] root INFO:     epochs : 20
    [2022/04/04 18:13:38] root INFO:     eval_during_train : True
    [2022/04/04 18:13:38] root INFO:     eval_interval : 5
    [2022/04/04 18:13:38] root INFO:     image_shape : [3, 224, 224]
    [2022/04/04 18:13:38] root INFO:     output_dir : ./output/
    [2022/04/04 18:13:38] root INFO:     pretrained_model : ./output/ResNet50_vd/best_model
    [2022/04/04 18:13:38] root INFO:     print_batch_step : 10
    [2022/04/04 18:13:38] root INFO:     save_inference_dir : ./deploy/models/class_ResNet50_vd_ImageNet_infer
    [2022/04/04 18:13:38] root INFO:     save_interval : 5
    [2022/04/04 18:13:38] root INFO:     use_visualdl : False
    [2022/04/04 18:13:38] root INFO: Infer : 
    [2022/04/04 18:13:38] root INFO:     PostProcess : 
    [2022/04/04 18:13:38] root INFO:         class_id_map_file : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00030.jpg
    [2022/04/04 18:13:38] root INFO:         name : Topk
    [2022/04/04 18:13:38] root INFO:         topk : 5
    [2022/04/04 18:13:38] root INFO:     batch_size : 10
    [2022/04/04 18:13:38] root INFO:     infer_imgs : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/04 18:13:38] root INFO:     transforms : 
    [2022/04/04 18:13:38] root INFO:         DecodeImage : 
    [2022/04/04 18:13:38] root INFO:             channel_first : False
    [2022/04/04 18:13:38] root INFO:             to_rgb : True
    [2022/04/04 18:13:38] root INFO:         ResizeImage : 
    [2022/04/04 18:13:38] root INFO:             resize_short : 256
    [2022/04/04 18:13:38] root INFO:         CropImage : 
    [2022/04/04 18:13:38] root INFO:             size : 224
    [2022/04/04 18:13:38] root INFO:         NormalizeImage : 
    [2022/04/04 18:13:38] root INFO:             mean : [0.485, 0.456, 0.406]
    [2022/04/04 18:13:38] root INFO:             order : 
    [2022/04/04 18:13:38] root INFO:             scale : 1.0/255.0
    [2022/04/04 18:13:38] root INFO:             std : [0.229, 0.224, 0.225]
    [2022/04/04 18:13:38] root INFO:         ToCHWImage : None
    [2022/04/04 18:13:38] root INFO: Loss : 
    [2022/04/04 18:13:38] root INFO:     Eval : 
    [2022/04/04 18:13:38] root INFO:         CELoss : 
    [2022/04/04 18:13:38] root INFO:             weight : 1.0
    [2022/04/04 18:13:38] root INFO:     Train : 
    [2022/04/04 18:13:38] root INFO:         CELoss : 
    [2022/04/04 18:13:38] root INFO:             weight : 1.0
    [2022/04/04 18:13:38] root INFO: Metric : 
    [2022/04/04 18:13:38] root INFO:     Eval : 
    [2022/04/04 18:13:38] root INFO:         TopkAcc : 
    [2022/04/04 18:13:38] root INFO:             topk : [1, 5]
    [2022/04/04 18:13:38] root INFO:     Train : 
    [2022/04/04 18:13:38] root INFO:         TopkAcc : 
    [2022/04/04 18:13:38] root INFO:             topk : [1, 5]
    [2022/04/04 18:13:38] root INFO: Optimizer : 
    [2022/04/04 18:13:38] root INFO:     lr : 
    [2022/04/04 18:13:38] root INFO:         learning_rate : 0.0125
    [2022/04/04 18:13:38] root INFO:         name : Cosine
    [2022/04/04 18:13:38] root INFO:         warmup_epoch : 5
    [2022/04/04 18:13:38] root INFO:     momentum : 0.9
    [2022/04/04 18:13:38] root INFO:     name : Momentum
    [2022/04/04 18:13:38] root INFO:     regularizer : 
    [2022/04/04 18:13:38] root INFO:         coeff : 1e-05
    [2022/04/04 18:13:38] root INFO:         name : L2
    [2022/04/04 18:13:38] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
    [2022/04/04 18:13:38] root WARNING: The Global.class_num will be deprecated. Please use Arch.class_num instead. Arch.class_num has been set to 102.
    W0404 18:13:38.957692  2099 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0404 18:13:38.962862  2099 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and



```python
!ls ./deploy/models/class_ResNet50_vd_ImageNet_infer -la
```

    total 93944
    drwxr-xr-x 2 aistudio aistudio     4096 Apr  4 18:13 .
    drwxr-xr-x 3 aistudio aistudio     4096 Apr  4 18:13 ..
    -rw-r--r-- 1 aistudio aistudio 95165295 Apr  4 18:13 inference.pdiparams
    -rw-r--r-- 1 aistudio aistudio    23453 Apr  4 18:13 inference.pdiparams.info
    -rw-r--r-- 1 aistudio aistudio   996386 Apr  4 18:13 inference.pdmodel


# 五、OpenVINO预测

## 1.导入OPenVINO包


```python
!pip install openvino>log.log
```


```python
# model download
from pathlib import Path
import os
import urllib.request
import tarfile

# inference
import openvino
from openvino.runtime import Core

# preprocessing
import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Layout, Type, AsyncInferQueue, PartialShape

# results visualization
import time
import json
from IPython.display import Image
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    /tmp/ipykernel_2281/571563012.py in <module>
          7 # inference
          8 import openvino
    ----> 9 from openvino.runtime import Core
         10 
         11 # preprocessing


    ModuleNotFoundError: No module named 'openvino.runtime'



```python
dir(openvino)
```




    ['__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__path__',
     '__spec__']



## 1.预测


```python
from PIL import Image
img=Image.open('/home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg')
img
```




![png](output_27_0.png)




```python
# 预测
!python3 tools/infer.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml -o Infer.infer_imgs=/home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg  -o Global.pretrained_model=output/ResNet50_vd/best_model
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/10/31 02:02:53] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/10/31 02:02:53] root INFO: Arch : 
    [2021/10/31 02:02:53] root INFO:     name : ResNet50_vd
    [2021/10/31 02:02:53] root INFO: DataLoader : 
    [2021/10/31 02:02:53] root INFO:     Eval : 
    [2021/10/31 02:02:53] root INFO:         dataset : 
    [2021/10/31 02:02:53] root INFO:             cls_label_path : valid.txt
    [2021/10/31 02:02:53] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2021/10/31 02:02:53] root INFO:             name : ImageNetDataset
    [2021/10/31 02:02:53] root INFO:             transform_ops : 
    [2021/10/31 02:02:53] root INFO:                 DecodeImage : 
    [2021/10/31 02:02:53] root INFO:                     channel_first : False
    [2021/10/31 02:02:53] root INFO:                     to_rgb : True
    [2021/10/31 02:02:53] root INFO:                 ResizeImage : 
    [2021/10/31 02:02:53] root INFO:                     resize_short : 256
    [2021/10/31 02:02:53] root INFO:                 CropImage : 
    [2021/10/31 02:02:53] root INFO:                     size : 224
    [2021/10/31 02:02:53] root INFO:                 NormalizeImage : 
    [2021/10/31 02:02:53] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/10/31 02:02:53] root INFO:                     order : 
    [2021/10/31 02:02:53] root INFO:                     scale : 1.0/255.0
    [2021/10/31 02:02:53] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/10/31 02:02:53] root INFO:         loader : 
    [2021/10/31 02:02:53] root INFO:             num_workers : 4
    [2021/10/31 02:02:53] root INFO:             use_shared_memory : True
    [2021/10/31 02:02:53] root INFO:         sampler : 
    [2021/10/31 02:02:53] root INFO:             batch_size : 256
    [2021/10/31 02:02:53] root INFO:             drop_last : False
    [2021/10/31 02:02:53] root INFO:             name : DistributedBatchSampler
    [2021/10/31 02:02:53] root INFO:             shuffle : False
    [2021/10/31 02:02:53] root INFO:     Train : 
    [2021/10/31 02:02:53] root INFO:         dataset : 
    [2021/10/31 02:02:53] root INFO:             cls_label_path : train.txt
    [2021/10/31 02:02:53] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2021/10/31 02:02:53] root INFO:             name : ImageNetDataset
    [2021/10/31 02:02:53] root INFO:             transform_ops : 
    [2021/10/31 02:02:53] root INFO:                 DecodeImage : 
    [2021/10/31 02:02:53] root INFO:                     channel_first : False
    [2021/10/31 02:02:53] root INFO:                     to_rgb : True
    [2021/10/31 02:02:53] root INFO:                 RandCropImage : 
    [2021/10/31 02:02:53] root INFO:                     size : 224
    [2021/10/31 02:02:53] root INFO:                 RandFlipImage : 
    [2021/10/31 02:02:53] root INFO:                     flip_code : 1
    [2021/10/31 02:02:53] root INFO:                 NormalizeImage : 
    [2021/10/31 02:02:53] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/10/31 02:02:53] root INFO:                     order : 
    [2021/10/31 02:02:53] root INFO:                     scale : 1.0/255.0
    [2021/10/31 02:02:53] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/10/31 02:02:53] root INFO:         loader : 
    [2021/10/31 02:02:53] root INFO:             num_workers : 4
    [2021/10/31 02:02:53] root INFO:             use_shared_memory : True
    [2021/10/31 02:02:53] root INFO:         sampler : 
    [2021/10/31 02:02:53] root INFO:             batch_size : 256
    [2021/10/31 02:02:53] root INFO:             drop_last : False
    [2021/10/31 02:02:53] root INFO:             name : DistributedBatchSampler
    [2021/10/31 02:02:53] root INFO:             shuffle : True
    [2021/10/31 02:02:53] root INFO: Global : 
    [2021/10/31 02:02:53] root INFO:     checkpoints : None
    [2021/10/31 02:02:53] root INFO:     class_num : 102
    [2021/10/31 02:02:53] root INFO:     device : gpu
    [2021/10/31 02:02:53] root INFO:     epochs : 20
    [2021/10/31 02:02:53] root INFO:     eval_during_train : True
    [2021/10/31 02:02:53] root INFO:     eval_interval : 5
    [2021/10/31 02:02:53] root INFO:     image_shape : [3, 224, 224]
    [2021/10/31 02:02:53] root INFO:     output_dir : ./output/
    [2021/10/31 02:02:53] root INFO:     pretrained_model : output/ResNet50_vd/best_model
    [2021/10/31 02:02:53] root INFO:     print_batch_step : 10
    [2021/10/31 02:02:53] root INFO:     save_inference_dir : ./inference
    [2021/10/31 02:02:53] root INFO:     save_interval : 5
    [2021/10/31 02:02:53] root INFO:     use_visualdl : False
    [2021/10/31 02:02:53] root INFO: Infer : 
    [2021/10/31 02:02:53] root INFO:     PostProcess : 
    [2021/10/31 02:02:53] root INFO:         class_id_map_file : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00030.jpg
    [2021/10/31 02:02:53] root INFO:         name : Topk
    [2021/10/31 02:02:53] root INFO:         topk : 5
    [2021/10/31 02:02:53] root INFO:     batch_size : 10
    [2021/10/31 02:02:53] root INFO:     infer_imgs : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg
    [2021/10/31 02:02:53] root INFO:     transforms : 
    [2021/10/31 02:02:53] root INFO:         DecodeImage : 
    [2021/10/31 02:02:53] root INFO:             channel_first : False
    [2021/10/31 02:02:53] root INFO:             to_rgb : True
    [2021/10/31 02:02:53] root INFO:         ResizeImage : 
    [2021/10/31 02:02:53] root INFO:             resize_short : 256
    [2021/10/31 02:02:53] root INFO:         CropImage : 
    [2021/10/31 02:02:53] root INFO:             size : 224
    [2021/10/31 02:02:53] root INFO:         NormalizeImage : 
    [2021/10/31 02:02:53] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/10/31 02:02:53] root INFO:             order : 
    [2021/10/31 02:02:53] root INFO:             scale : 1.0/255.0
    [2021/10/31 02:02:53] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/10/31 02:02:53] root INFO:         ToCHWImage : None
    [2021/10/31 02:02:53] root INFO: Loss : 
    [2021/10/31 02:02:53] root INFO:     Eval : 
    [2021/10/31 02:02:53] root INFO:         CELoss : 
    [2021/10/31 02:02:53] root INFO:             weight : 1.0
    [2021/10/31 02:02:53] root INFO:     Train : 
    [2021/10/31 02:02:53] root INFO:         CELoss : 
    [2021/10/31 02:02:53] root INFO:             weight : 1.0
    [2021/10/31 02:02:53] root INFO: Metric : 
    [2021/10/31 02:02:53] root INFO:     Eval : 
    [2021/10/31 02:02:53] root INFO:         TopkAcc : 
    [2021/10/31 02:02:53] root INFO:             topk : [1, 5]
    [2021/10/31 02:02:53] root INFO:     Train : 
    [2021/10/31 02:02:53] root INFO:         TopkAcc : 
    [2021/10/31 02:02:53] root INFO:             topk : [1, 5]
    [2021/10/31 02:02:53] root INFO: Optimizer : 
    [2021/10/31 02:02:53] root INFO:     lr : 
    [2021/10/31 02:02:53] root INFO:         learning_rate : 0.0125
    [2021/10/31 02:02:53] root INFO:         name : Cosine
    [2021/10/31 02:02:53] root INFO:         warmup_epoch : 5
    [2021/10/31 02:02:53] root INFO:     momentum : 0.9
    [2021/10/31 02:02:53] root INFO:     name : Momentum
    [2021/10/31 02:02:53] root INFO:     regularizer : 
    [2021/10/31 02:02:53] root INFO:         coeff : 1e-05
    [2021/10/31 02:02:53] root INFO:         name : L2
    [2021/10/31 02:02:53] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
    W1031 02:02:53.626825  7656 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1031 02:02:53.631515  7656 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    [{'class_ids': [37, 35, 76, 28, 50], 'scores': [0.45998, 0.13054, 0.03585, 0.03207, 0.02833], 'file_name': '/home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg', 'label_names': []}]


可见，更新后的版本日志输出比较混乱。

最终输出
```
[{'class_ids': [37, 35, 76, 28, 50], 'scores': [0.45998, 0.13054, 0.03585, 0.03207, 0.02833], 'file_name': '/home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg', 'label_names': []}]
```

显示的是top5的概率，可见分类为37类。由于没有设置label_names，所以这处为空。


## 2.使用inference模型进行模型推理

### 2.1 inference格式转换
通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。


```python
%cd ~/PaddleClas/
!python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/ResNet50_vd.yaml \
    -o Global.pretrained_model=output/ResNet50_vd/best_model
```

    /home/aistudio/PaddleClas
    [2022/04/11 22:22:21] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2022/04/11 22:22:21] root INFO: Arch : 
    [2022/04/11 22:22:21] root INFO:     name : ResNet50_vd
    [2022/04/11 22:22:21] root INFO: DataLoader : 
    [2022/04/11 22:22:21] root INFO:     Eval : 
    [2022/04/11 22:22:21] root INFO:         dataset : 
    [2022/04/11 22:22:21] root INFO:             cls_label_path : valid.txt
    [2022/04/11 22:22:21] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/11 22:22:21] root INFO:             name : ImageNetDataset
    [2022/04/11 22:22:21] root INFO:             transform_ops : 
    [2022/04/11 22:22:21] root INFO:                 DecodeImage : 
    [2022/04/11 22:22:21] root INFO:                     channel_first : False
    [2022/04/11 22:22:21] root INFO:                     to_rgb : True
    [2022/04/11 22:22:21] root INFO:                 ResizeImage : 
    [2022/04/11 22:22:21] root INFO:                     resize_short : 256
    [2022/04/11 22:22:21] root INFO:                 CropImage : 
    [2022/04/11 22:22:21] root INFO:                     size : 224
    [2022/04/11 22:22:21] root INFO:                 NormalizeImage : 
    [2022/04/11 22:22:21] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2022/04/11 22:22:21] root INFO:                     order : 
    [2022/04/11 22:22:21] root INFO:                     scale : 1.0/255.0
    [2022/04/11 22:22:21] root INFO:                     std : [0.229, 0.224, 0.225]
    [2022/04/11 22:22:21] root INFO:         loader : 
    [2022/04/11 22:22:21] root INFO:             num_workers : 4
    [2022/04/11 22:22:21] root INFO:             use_shared_memory : True
    [2022/04/11 22:22:21] root INFO:         sampler : 
    [2022/04/11 22:22:21] root INFO:             batch_size : 128
    [2022/04/11 22:22:21] root INFO:             drop_last : False
    [2022/04/11 22:22:21] root INFO:             name : DistributedBatchSampler
    [2022/04/11 22:22:21] root INFO:             shuffle : False
    [2022/04/11 22:22:21] root INFO:     Train : 
    [2022/04/11 22:22:21] root INFO:         dataset : 
    [2022/04/11 22:22:21] root INFO:             cls_label_path : train.txt
    [2022/04/11 22:22:21] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/11 22:22:21] root INFO:             name : ImageNetDataset
    [2022/04/11 22:22:21] root INFO:             transform_ops : 
    [2022/04/11 22:22:21] root INFO:                 DecodeImage : 
    [2022/04/11 22:22:21] root INFO:                     channel_first : False
    [2022/04/11 22:22:21] root INFO:                     to_rgb : True
    [2022/04/11 22:22:21] root INFO:                 RandCropImage : 
    [2022/04/11 22:22:21] root INFO:                     size : 224
    [2022/04/11 22:22:21] root INFO:                 RandFlipImage : 
    [2022/04/11 22:22:21] root INFO:                     flip_code : 1
    [2022/04/11 22:22:21] root INFO:                 NormalizeImage : 
    [2022/04/11 22:22:21] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2022/04/11 22:22:21] root INFO:                     order : 
    [2022/04/11 22:22:21] root INFO:                     scale : 1.0/255.0
    [2022/04/11 22:22:21] root INFO:                     std : [0.229, 0.224, 0.225]
    [2022/04/11 22:22:21] root INFO:         loader : 
    [2022/04/11 22:22:21] root INFO:             num_workers : 4
    [2022/04/11 22:22:21] root INFO:             use_shared_memory : True
    [2022/04/11 22:22:21] root INFO:         sampler : 
    [2022/04/11 22:22:21] root INFO:             batch_size : 128
    [2022/04/11 22:22:21] root INFO:             drop_last : False
    [2022/04/11 22:22:21] root INFO:             name : DistributedBatchSampler
    [2022/04/11 22:22:21] root INFO:             shuffle : True
    [2022/04/11 22:22:21] root INFO: Global : 
    [2022/04/11 22:22:21] root INFO:     checkpoints : None
    [2022/04/11 22:22:21] root INFO:     class_num : 102
    [2022/04/11 22:22:21] root INFO:     device : gpu
    [2022/04/11 22:22:21] root INFO:     epochs : 20
    [2022/04/11 22:22:21] root INFO:     eval_during_train : True
    [2022/04/11 22:22:21] root INFO:     eval_interval : 5
    [2022/04/11 22:22:21] root INFO:     image_shape : [3, 224, 224]
    [2022/04/11 22:22:21] root INFO:     output_dir : ./output/
    [2022/04/11 22:22:21] root INFO:     pretrained_model : output/ResNet50_vd/best_model
    [2022/04/11 22:22:21] root INFO:     print_batch_step : 10
    [2022/04/11 22:22:21] root INFO:     save_inference_dir : ./inference
    [2022/04/11 22:22:21] root INFO:     save_interval : 5
    [2022/04/11 22:22:21] root INFO:     use_visualdl : False
    [2022/04/11 22:22:21] root INFO: Infer : 
    [2022/04/11 22:22:21] root INFO:     PostProcess : 
    [2022/04/11 22:22:21] root INFO:         class_id_map_file : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00030.jpg
    [2022/04/11 22:22:21] root INFO:         name : Topk
    [2022/04/11 22:22:21] root INFO:         topk : 5
    [2022/04/11 22:22:21] root INFO:     batch_size : 10
    [2022/04/11 22:22:21] root INFO:     infer_imgs : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2022/04/11 22:22:21] root INFO:     transforms : 
    [2022/04/11 22:22:21] root INFO:         DecodeImage : 
    [2022/04/11 22:22:21] root INFO:             channel_first : False
    [2022/04/11 22:22:21] root INFO:             to_rgb : True
    [2022/04/11 22:22:21] root INFO:         ResizeImage : 
    [2022/04/11 22:22:21] root INFO:             resize_short : 256
    [2022/04/11 22:22:21] root INFO:         CropImage : 
    [2022/04/11 22:22:21] root INFO:             size : 224
    [2022/04/11 22:22:21] root INFO:         NormalizeImage : 
    [2022/04/11 22:22:21] root INFO:             mean : [0.485, 0.456, 0.406]
    [2022/04/11 22:22:21] root INFO:             order : 
    [2022/04/11 22:22:21] root INFO:             scale : 1.0/255.0
    [2022/04/11 22:22:21] root INFO:             std : [0.229, 0.224, 0.225]
    [2022/04/11 22:22:21] root INFO:         ToCHWImage : None
    [2022/04/11 22:22:21] root INFO: Loss : 
    [2022/04/11 22:22:21] root INFO:     Eval : 
    [2022/04/11 22:22:21] root INFO:         CELoss : 
    [2022/04/11 22:22:21] root INFO:             weight : 1.0
    [2022/04/11 22:22:21] root INFO:     Train : 
    [2022/04/11 22:22:21] root INFO:         CELoss : 
    [2022/04/11 22:22:21] root INFO:             weight : 1.0
    [2022/04/11 22:22:21] root INFO: Metric : 
    [2022/04/11 22:22:21] root INFO:     Eval : 
    [2022/04/11 22:22:21] root INFO:         TopkAcc : 
    [2022/04/11 22:22:21] root INFO:             topk : [1, 5]
    [2022/04/11 22:22:21] root INFO:     Train : 
    [2022/04/11 22:22:21] root INFO:         TopkAcc : 
    [2022/04/11 22:22:21] root INFO:             topk : [1, 5]
    [2022/04/11 22:22:21] root INFO: Optimizer : 
    [2022/04/11 22:22:21] root INFO:     lr : 
    [2022/04/11 22:22:21] root INFO:         learning_rate : 0.0125
    [2022/04/11 22:22:21] root INFO:         name : Cosine
    [2022/04/11 22:22:21] root INFO:         warmup_epoch : 5
    [2022/04/11 22:22:21] root INFO:     momentum : 0.9
    [2022/04/11 22:22:21] root INFO:     name : Momentum
    [2022/04/11 22:22:21] root INFO:     regularizer : 
    [2022/04/11 22:22:21] root INFO:         coeff : 1e-05
    [2022/04/11 22:22:21] root INFO:         name : L2
    [2022/04/11 22:22:21] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
    [2022/04/11 22:22:21] root WARNING: The Global.class_num will be deprecated. Please use Arch.class_num instead. Arch.class_num has been set to 102.
    W0411 22:22:21.514081   314 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0411 22:22:21.518855   314 device_context.cc:422] device: 0, cuDNN Version: 7.6.


文件保存到了PaddleClas/inference/目录

```
inference.pdiparams  
inference.pdiparams.info  
inference.pdmodel
```

### 2.2 使用inference进行预测


```python
%cd deploy
!python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=/home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.Topk.class_id_map_file=None
```

    /home/aistudio/PaddleClas/deploy
    2021-10-31 02:34:50 INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    2021-10-31 02:34:50 INFO: Global : 
    2021-10-31 02:34:50 INFO:     batch_size : 1
    2021-10-31 02:34:50 INFO:     cpu_num_threads : 10
    2021-10-31 02:34:50 INFO:     enable_benchmark : True
    2021-10-31 02:34:50 INFO:     enable_mkldnn : True
    2021-10-31 02:34:50 INFO:     enable_profile : False
    2021-10-31 02:34:50 INFO:     gpu_mem : 8000
    2021-10-31 02:34:50 INFO:     infer_imgs : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00033.jpg
    2021-10-31 02:34:50 INFO:     inference_model_dir : ../inference/
    2021-10-31 02:34:50 INFO:     ir_optim : True
    2021-10-31 02:34:50 INFO:     use_fp16 : False
    2021-10-31 02:34:50 INFO:     use_gpu : True
    2021-10-31 02:34:50 INFO:     use_tensorrt : False
    2021-10-31 02:34:50 INFO: PostProcess : 
    2021-10-31 02:34:50 INFO:     SavePreLabel : 
    2021-10-31 02:34:50 INFO:         save_dir : ./pre_label/
    2021-10-31 02:34:50 INFO:     Topk : 
    2021-10-31 02:34:50 INFO:         class_id_map_file : None
    2021-10-31 02:34:50 INFO:         topk : 5
    2021-10-31 02:34:50 INFO:     main_indicator : Topk
    2021-10-31 02:34:50 INFO: PreProcess : 
    2021-10-31 02:34:50 INFO:     transform_ops : 
    2021-10-31 02:34:50 INFO:         ResizeImage : 
    2021-10-31 02:34:50 INFO:             resize_short : 256
    2021-10-31 02:34:50 INFO:         CropImage : 
    2021-10-31 02:34:50 INFO:             size : 224
    2021-10-31 02:34:50 INFO:         NormalizeImage : 
    2021-10-31 02:34:50 INFO:             channel_num : 3
    2021-10-31 02:34:50 INFO:             mean : [0.485, 0.456, 0.406]
    2021-10-31 02:34:50 INFO:             order : 
    2021-10-31 02:34:50 INFO:             scale : 0.00392157
    2021-10-31 02:34:50 INFO:             std : [0.229, 0.224, 0.225]
    2021-10-31 02:34:50 INFO:         ToCHWImage : None
    image_00033.jpg:	class id(s): [37, 35, 76, 28, 50], score(s): [0.46, 0.13, 0.04, 0.03, 0.03], label_name(s): []



```python

```

# 五、模型评估


```python
!python tools/eval.py \
        -c ./ppcls/configs/quick_start/ResNet50_vd.yaml \
        -o Global.pretrained_model=output/ResNet50_vd/best_model
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/10/31 02:28:12] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/10/31 02:28:12] root INFO: Arch : 
    [2021/10/31 02:28:12] root INFO:     name : ResNet50_vd
    [2021/10/31 02:28:12] root INFO: DataLoader : 
    [2021/10/31 02:28:12] root INFO:     Eval : 
    [2021/10/31 02:28:12] root INFO:         dataset : 
    [2021/10/31 02:28:12] root INFO:             cls_label_path : valid.txt
    [2021/10/31 02:28:12] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2021/10/31 02:28:12] root INFO:             name : ImageNetDataset
    [2021/10/31 02:28:12] root INFO:             transform_ops : 
    [2021/10/31 02:28:12] root INFO:                 DecodeImage : 
    [2021/10/31 02:28:12] root INFO:                     channel_first : False
    [2021/10/31 02:28:12] root INFO:                     to_rgb : True
    [2021/10/31 02:28:12] root INFO:                 ResizeImage : 
    [2021/10/31 02:28:12] root INFO:                     resize_short : 256
    [2021/10/31 02:28:12] root INFO:                 CropImage : 
    [2021/10/31 02:28:12] root INFO:                     size : 224
    [2021/10/31 02:28:12] root INFO:                 NormalizeImage : 
    [2021/10/31 02:28:12] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/10/31 02:28:12] root INFO:                     order : 
    [2021/10/31 02:28:12] root INFO:                     scale : 1.0/255.0
    [2021/10/31 02:28:12] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/10/31 02:28:12] root INFO:         loader : 
    [2021/10/31 02:28:12] root INFO:             num_workers : 4
    [2021/10/31 02:28:12] root INFO:             use_shared_memory : True
    [2021/10/31 02:28:12] root INFO:         sampler : 
    [2021/10/31 02:28:12] root INFO:             batch_size : 256
    [2021/10/31 02:28:12] root INFO:             drop_last : False
    [2021/10/31 02:28:12] root INFO:             name : DistributedBatchSampler
    [2021/10/31 02:28:12] root INFO:             shuffle : False
    [2021/10/31 02:28:12] root INFO:     Train : 
    [2021/10/31 02:28:12] root INFO:         dataset : 
    [2021/10/31 02:28:12] root INFO:             cls_label_path : train.txt
    [2021/10/31 02:28:12] root INFO:             image_root : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2021/10/31 02:28:12] root INFO:             name : ImageNetDataset
    [2021/10/31 02:28:12] root INFO:             transform_ops : 
    [2021/10/31 02:28:12] root INFO:                 DecodeImage : 
    [2021/10/31 02:28:12] root INFO:                     channel_first : False
    [2021/10/31 02:28:12] root INFO:                     to_rgb : True
    [2021/10/31 02:28:12] root INFO:                 RandCropImage : 
    [2021/10/31 02:28:12] root INFO:                     size : 224
    [2021/10/31 02:28:12] root INFO:                 RandFlipImage : 
    [2021/10/31 02:28:12] root INFO:                     flip_code : 1
    [2021/10/31 02:28:12] root INFO:                 NormalizeImage : 
    [2021/10/31 02:28:12] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/10/31 02:28:12] root INFO:                     order : 
    [2021/10/31 02:28:12] root INFO:                     scale : 1.0/255.0
    [2021/10/31 02:28:12] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/10/31 02:28:12] root INFO:         loader : 
    [2021/10/31 02:28:12] root INFO:             num_workers : 4
    [2021/10/31 02:28:12] root INFO:             use_shared_memory : True
    [2021/10/31 02:28:12] root INFO:         sampler : 
    [2021/10/31 02:28:12] root INFO:             batch_size : 256
    [2021/10/31 02:28:12] root INFO:             drop_last : False
    [2021/10/31 02:28:12] root INFO:             name : DistributedBatchSampler
    [2021/10/31 02:28:12] root INFO:             shuffle : True
    [2021/10/31 02:28:12] root INFO: Global : 
    [2021/10/31 02:28:12] root INFO:     checkpoints : None
    [2021/10/31 02:28:12] root INFO:     class_num : 102
    [2021/10/31 02:28:12] root INFO:     device : gpu
    [2021/10/31 02:28:12] root INFO:     epochs : 20
    [2021/10/31 02:28:12] root INFO:     eval_during_train : True
    [2021/10/31 02:28:12] root INFO:     eval_interval : 5
    [2021/10/31 02:28:12] root INFO:     image_shape : [3, 224, 224]
    [2021/10/31 02:28:12] root INFO:     output_dir : ./output/
    [2021/10/31 02:28:12] root INFO:     pretrained_model : output/ResNet50_vd/best_model
    [2021/10/31 02:28:12] root INFO:     print_batch_step : 10
    [2021/10/31 02:28:12] root INFO:     save_inference_dir : ./inference
    [2021/10/31 02:28:12] root INFO:     save_interval : 5
    [2021/10/31 02:28:12] root INFO:     use_visualdl : False
    [2021/10/31 02:28:12] root INFO: Infer : 
    [2021/10/31 02:28:12] root INFO:     PostProcess : 
    [2021/10/31 02:28:12] root INFO:         class_id_map_file : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/jpg/image_00030.jpg
    [2021/10/31 02:28:12] root INFO:         name : Topk
    [2021/10/31 02:28:12] root INFO:         topk : 5
    [2021/10/31 02:28:12] root INFO:     batch_size : 10
    [2021/10/31 02:28:12] root INFO:     infer_imgs : /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/
    [2021/10/31 02:28:12] root INFO:     transforms : 
    [2021/10/31 02:28:12] root INFO:         DecodeImage : 
    [2021/10/31 02:28:12] root INFO:             channel_first : False
    [2021/10/31 02:28:12] root INFO:             to_rgb : True
    [2021/10/31 02:28:12] root INFO:         ResizeImage : 
    [2021/10/31 02:28:12] root INFO:             resize_short : 256
    [2021/10/31 02:28:12] root INFO:         CropImage : 
    [2021/10/31 02:28:12] root INFO:             size : 224
    [2021/10/31 02:28:12] root INFO:         NormalizeImage : 
    [2021/10/31 02:28:12] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/10/31 02:28:12] root INFO:             order : 
    [2021/10/31 02:28:12] root INFO:             scale : 1.0/255.0
    [2021/10/31 02:28:12] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/10/31 02:28:12] root INFO:         ToCHWImage : None
    [2021/10/31 02:28:12] root INFO: Loss : 
    [2021/10/31 02:28:12] root INFO:     Eval : 
    [2021/10/31 02:28:12] root INFO:         CELoss : 
    [2021/10/31 02:28:12] root INFO:             weight : 1.0
    [2021/10/31 02:28:12] root INFO:     Train : 
    [2021/10/31 02:28:12] root INFO:         CELoss : 
    [2021/10/31 02:28:12] root INFO:             weight : 1.0
    [2021/10/31 02:28:12] root INFO: Metric : 
    [2021/10/31 02:28:12] root INFO:     Eval : 
    [2021/10/31 02:28:12] root INFO:         TopkAcc : 
    [2021/10/31 02:28:12] root INFO:             topk : [1, 5]
    [2021/10/31 02:28:12] root INFO:     Train : 
    [2021/10/31 02:28:12] root INFO:         TopkAcc : 
    [2021/10/31 02:28:12] root INFO:             topk : [1, 5]
    [2021/10/31 02:28:12] root INFO: Optimizer : 
    [2021/10/31 02:28:12] root INFO:     lr : 
    [2021/10/31 02:28:12] root INFO:         learning_rate : 0.0125
    [2021/10/31 02:28:12] root INFO:         name : Cosine
    [2021/10/31 02:28:12] root INFO:         warmup_epoch : 5
    [2021/10/31 02:28:12] root INFO:     momentum : 0.9
    [2021/10/31 02:28:12] root INFO:     name : Momentum
    [2021/10/31 02:28:12] root INFO:     regularizer : 
    [2021/10/31 02:28:12] root INFO:         coeff : 1e-05
    [2021/10/31 02:28:12] root INFO:         name : L2
    [2021/10/31 02:28:12] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
    self._cls_path /home/aistudio/data/oxford-102-flowers/oxford-102-flowers/valid.txt
    W1031 02:28:12.918593 11295 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1031 02:28:12.923264 11295 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    [2021/10/31 02:28:21] root INFO: [Eval][Epoch 0][Iter: 0/4]CELoss: 0.47912, loss: 0.47912, top1: 0.91797, top5: 0.96094, batch_cost: 3.39170s, reader_cost: 2.99661, ips: 75.47843 images/sec
    [2021/10/31 02:28:22] root INFO: [Eval][Epoch 0][Avg]CELoss: 0.54982, loss: 0.54982, top1: 0.88922, top5: 0.96667


eval结果
```
W1031 02:28:12.923264 11295 device_context.cc:422] device: 0, cuDNN Version: 7.6.
[2021/10/31 02:28:21] root INFO: [Eval][Epoch 0][Iter: 0/4]CELoss: 0.47912, loss: 0.47912, top1: 0.91797, top5: 0.96094, batch_cost: 3.39170s, reader_cost: 2.99661, ips: 75.47843 images/sec
[2021/10/31 02:28:22] root INFO: [Eval][Epoch 0][Avg]CELoss: 0.54982, loss: 0.54982, top1: 0.88922, top5: 0.96667
```


```python

```
