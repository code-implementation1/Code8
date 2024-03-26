# Contents

* [Contents](#contents)
    * [SpineNet RetinaNet Description](#retinanet-description)
        * [Model Architecture](#model-architecture)
        * [Dataset](#dataset)
    * [Environment Requirements](#environment-requirements)
    * [Quick Start](#quick-start)
        * [Prepare the model](#prepare-the-model)
        * [Run the scripts](#run-the-scripts)
    * [Script Description](#script-description)
        * [Script and Sample Code](#script-and-sample-code)
        * [Script Parameters](#script-parameters)
    * [Training](#training)
        * [Training Process](#training-process)
        * [Transfer Training](#transfer-training)
        * [Distribute training](#distribute-training)
    * [Evaluation](#evaluation)
        * [Evaluation Process](#evaluation-process)
            * [Evaluation on GPU](#evaluation-on-gpu)
        * [Evaluation result](#evaluation-result)
    * [Inference](#inference)
        * [Inference Process](#inference-process)
            * [Inference on GPU](#inference-on-gpu)
        * [Inference result](#inference-result)
   * [Model Description](#model-description)
        * [Performance](#performance)
   * [Description of Random Situation](#description-of-random-situation)
   * [ModelZoo Homepage](#modelzoo-homepage)

## [SpineNet RetinaNet Description](#contents)

SpineNet is a convolution backbone with scale-permuted intermediate features
and cross-scale connections that is learned on an object detection task by
Neural Architecture Search. Using similar building blocks, SpineNet models
outperform ResNet-FPN models at various scales.

The architecture of the proposed backbone model consists of a fixed stem
network followed by a learned scale permuted network. A stem network is
designed with scale decreased architecture. Blocks in the stem network can be
candidate inputs for the following scale-permuted network.

[Paper](https://arxiv.org/abs/1912.05027): Xianzhi Du, Tsung-Yi Lin,
Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V. Le, Xiaodan Song.
Computer Vision and Pattern Recognition (CVPR), 2020 (In press).

### [Model Architecture](#contents)

**Overview of the pipeline of SpineNet RetinaNet:**
The RetinaNet algorithm is derived from the paper "Focal Loss for Dense Object
Detection" of Facebook AI Research in 2018. The biggest contribution of this
paper is that Focal Loss is proposed to solve the problem of class imbalance,
thereby creating RetinaNet (one-stage object detection algorithm), an object
detection network with accuracy higher than that of the classical two-stage
Faster-RCNN.

RetinaNet result prediction pipeline:

1. SpineNet backbone.
2. RetinaHead.
3. multiclass NMS (reduce number of proposed boxes and omit objects with low
 confidence).

RetinaNet result training pipeline:

1. SpineNet backbone.
2. RetinaHead.
3. Assigner.
4. Classification + Localization losses.

### [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [COCO-2017](https://cocodataset.org/#download)

* Dataset size: 25.4G
    * Train: 18.0G，118287 images
    * Val: 777.1M，5000 images
    * Test: 6.2G，40670 images
    * Annotations: 474.7M, represented in 3 JSON files for each subset.
* Data format: image and json files.
    * Note: Data will be processed in dataset.py

## [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset COCO-2017.
* Install third-parties requirements:

```text
numpy~=1.21.2
opencv-python~=4.5.4.58
pycocotools>=2.0.5
matplotlib
seaborn
pandas
tqdm==4.64.1
```

* We use COCO-2017 as training dataset in this example by default, and you
 can also use your own datasets. Dataset structure:

```log
.
└── coco-2017
    ├── train
    │   ├── data
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   └── labels.json
    ├── validation
    │   ├── data
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   └── labels.json
    └── test
        ├── data
        │    ├── 000000000001.jpg
        │    ├── 000000000002.jpg
        │    └── ...
        └── labels.json
```

## [Quick Start](#contents)

### [Prepare the model](#contents)

1. Prepare yaml config file. Create file and copy content from
 `default_config.yaml` to created file.
2. Change data settings: experiment folder (`train_outputs`), image size
 settings (`img_width`, `img_height`, etc.), subsets folders (`train_dataset`,
 `val_dataset`), information about categories etc.
3. Change the backbone settings.
4. Change other training hyperparameters (learning rate, regularization,
 augmentations etc.).

### [Run the scripts](#contents)

After installing MindSpore via the official website, you can start training and
evaluation as follows:

* running on GPU

```shell
# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]

# run eval on GPU
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (Optional)[PREDICTION_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```log
SpineNet
└── retinanet
    ├── configs
    │   ├── retinanet_spinenet_143_1280.yaml                    ## SpineNet143 configuration.
    │   ├── retinanet_spinenet_49_640.yaml                      ## SpineNet49 (640x640) configuration.
    │   ├── retinanet_spinenet_49_896.yaml                      ## SpineNet49 (896x896) configuration.
    │   ├── retinanet_spinenet_49s_640.yaml                     ## SpineNet49s configuration.
    │   └── retinanet_spinenet_96_1024.yaml                     ## SpineNet96 configuration.
    ├── scripts
    │   ├── run_distribute_train_gpu.sh                         ## Bash script for distributed training on gpu.
    │   ├── run_eval_gpu.sh                                     ## Bash script for eval on gpu.
    │   ├── run_infer_gpu.sh                                    ## Bash script for gpu model inference.
    │   └── run_standalone_train_gpu.sh                         ## Bash script for training on gpu.
    ├── src
    │   ├── blocks
    │   │   ├── anchor_generator
    │   │   │   ├── anchor_generator.py                     ## Anchor generator.
    │   │   │   └── __init__.py
    │   │   ├── assigners_samplers
    │   │   │   ├── assigner_pseudo_sampler.py              ## Wrapper for assigner.
    │   │   │   ├── __init__.py
    │   │   │   └── max_iou_assigner.py                     ## MaxIOUAssigner.
    │   │   ├── backbones
    │   │   │   ├── __init__.py
    │   │   │   └── spinenet.py                             ## Implemented SpineNet backbone.
    │   │   ├── bbox_coders
    │   │   │   ├── bbox_coder.py                           ## Bounding box coder.
    │   │   │   └── __init__.py
    │   │   ├── dense_heads
    │   │   │   ├── __init__.py
    │   │   │   └── retina_sepbn_head.py                    ## RetinaHead.
    │   │   ├── initialization
    │   │   │   ├── initialization.py                       ## Weight initialization functional.
    │   │   │   └── __init__.py
    │   │   ├── layers
    │   │   │   ├── conv_module.py                          ## Convolution module.
    │   │   │   └── __init__.py
    │   │   ├── losses
    │   │   │   ├── focal_loss.py                           ## FocalLoss.
    │   │   │   └── __init__.py
    │   │   ├── __init__.py
    │   │   └── retinanet.py                                  ## RetinaNet model
    │   ├── model_utils
    │   │   ├── config.py                                 ## Configuration file parsing utils.
    │   │   ├── device_adapter.py                         ## File to adapt current used devices.
    │   │   ├── __init__.py
    │   │   ├── local_adapter.py                          ## File to work with local devices.
    │   │   └── moxing_adapter.py                         ## File to work with model arts devices.
    │   ├── callback.py                                         ## Callbacks.
    │   ├── common.py                                           ## Common functional with common setting.
    │   ├── dataset.py                                          ## Images loading and preprocessing.
    │   ├── detecteval.py                                       ## DetectEval class to analyze predictions
    │   ├── eval_utils.py                                       ## Evaluation metrics utilities.
    │   ├── __init__.py
    │   ├── lr_schedule.py                                      ## Optimizer settings.
    │   ├── mlflow_funcs.py                                     ## mlflow utilities.
    │   └── network_define.py                                   ## Model wrappers for training.
    ├── draw_predictions.py                                       ## Draw results of infer.py script on image.
    ├── eval.py                                                   ## Run models evaluation.
    ├── infer.py                                                  ## Make predictions for models.
    ├── __init__.py
    ├── README.md                                                 ## RetinaNet definition.
    ├── requirements.txt                                          ## Dependencies.
    └── train.py                                                  ## Train script.
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```shell
train_outputs: '/data/retinanet_models'
brief: 'gpu-1'
device_target: GPU
mode: 'graph'
# ==============================================================================
detector: 'retinanet'

# backbone
backbone:
  type: 'spinenet'                                                ## Backbone type
  arch: '49'                                                      ## Backbone architecture
  norm_cfg:                                                       ## Batch normalization parameter
    type: 'BN'
    momentum: 0.01
    eps: 0.001

# dense bbox head retinanet
bbox_head:
  num_ins: 5                                                      ## Number of stages
  in_channels: 256                                                ## Number of input channels
  stacked_convs: 4                                                ## Number of convolution layers per stage
  feat_channels: 256                                              ## Number of input channels
  bbox_coder:
    target_means: [0.0, 0.0, 0.0, 0.0]                            ## Bounding box coder parameter
    target_stds: [1.0, 1.0, 1.0, 1.0]                             ## Bounding box coder parameter
  loss_cls:
    gamma: 2.0                                                    ## Focal loss gamma parameter
    alpha: 0.25                                                   ## Focal loss alpha parameter
    loss_weight: 1.0                                              ## Classification loss weight
  loss_bbox:
    beta: 0.11                                                    ## Smooth loss beta parameter
    loss_weight: 1.0                                              ## Localization loss weight
  anchor_generator:
    octave_base_scale: 3
    scales_per_octave: 3
    ratios: [0.5, 1.0, 2.0]
    strides: [8, 16, 32, 64, 128]
  norm_cfg:
    type: 'BN'
    momentum: 0.01
    eps: 0.001

train_cfg:
  assigner:
    pos_iou_thr: 0.5                                            ## IoU threshold for negative bboxes
    neg_iou_thr: 0.5                                            ## IoU threshold for positive bboxes
    min_pos_iou: 0.00001                                        ## Minimum iou for a bbox to be considered as a positive bbox
    match_low_quality: True                                     ## Allow low quality matches

test_cfg:
  nms_pre: 1000                                                 ## max number of samples per level
  min_bbox_size: 0                                              ## min bboxes size
  score_thr: 0.05                                               ## Confidence threshold
  max_per_img: 100                                              ## Max number of output bboxes
  nms:
    iou_threshold: 0.5                                          ## IOU threshold

# optimizer
opt_type: 'sgd'                                                 ## Optimizer type (sgd or adam)
lr: 0.14                                                        ## Base learning rate
min_lr: 0.0000001                                               ## Minimum learning rate
momentum: 0.9                                                   ## Optimizer parameter
weight_decay: 0.00004                                           ## Regularization
warmup_step: 4000                                               ## Number of warmup steps
warmup_ratio: 0.1                                               ## Initial learning rate = base_lr * warmup_ratio
lr_steps: [320, 340]                                            ## Epochs numbers when learning rate is divided by 10 (for multistep lr_type)
lr_type: 'multistep'                                            ## Learning rate scheduling type
grad_clip: 0                                                    ## Gradient clipping (set 0 to turn off)


# train
num_gts: 100                                                   ## Train batch size
batch_size: 16                                                 ## Train batch size
accumulate_step: 1                                             ## artificial batch size multiplier
test_batch_size: 1                                             ## Test batch size
loss_scale: 256                                                ## Loss scale
epoch_size: 350                                                ## Number of epochs
run_eval: 1                                                    ## Whether evaluation or not
eval_every: 1                                                  ## Evaluation interval
enable_graph_kernel: 0                                         ## Turn on kernel fusion
finetune: 0                                                    ## Turn on finetune (for transfer learning)
datasink: 0                                                    ## Turn on data sink mode
pre_trained: ''                                                ## Path to pretrained model weights

#distribution training
run_distribute: 0                                              ## Turn on distributed training
device_id: 0                                                   ##
device_num: 1                                                  ## Number of devices (only if distributed training turned on)
rank_id: 0                                                     ##

# Number of threads used to process the dataset in parallel
num_parallel_workers: 6
# Parallelize Python operations with multiple worker processes
python_multiprocessing: 0
# dataset setting
train_dataset: '/data/coco-2017/train/'
val_dataset: '/home/coco-2017/validation/'
coco_classes: ['background', 'person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
num_classes: 81
train_dataset_num: 0
train_dataset_divider: 0

# images
img_width: 640                                                          ## Input images width
img_height: 640                                                         ## Input images height
divider: 64                                                             ## Automatically make width and height are dividable by divider
img_mean: [123.675, 116.28, 103.53]                                     ## Image normalization parameters
img_std: [58.395, 57.12, 57.375]                                        ## Image normalization parameters
to_rgb: 1                                                               ## RGB or BGR
keep_ratio: 1                                                           ## Keep ratio in original images

# augmentation
flip_ratio: 0.5                                                         ## Probability of image horizontal flip
expand_ratio: 0.0                                                       ## Probability of image expansion

# callbacks
save_every: 100                                                         ## Save model every <n> steps
keep_checkpoint_max: 5                                                  ## Max number of saved periodical checkpoints
keep_best_checkpoints_max: 5                                            ## Max number of saved best checkpoints
 ```

## [Training](#contents)

To train the model, run `train.py`.

### [Training process](#contents)

Standalone training mode:

```bash
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for these scripts.

* `CONFIG_PATH`: path to config file.
* `TRAIN_DATA`: path to train dataset.
* `VAL_DATA`: path to validation dataset.
* `TRAIN_OUT`: path to folder with training experiments.
* `BRIEF`: short experiment name.
* `PRETRAINED_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

Training result will be stored in the current path, whose folder name is "LOG".
Under this, you can find checkpoint files together with result like the
following in log.

```log
loading annotations into memory...
Done (t=0.46s)
creating index...
index created!
[2023-09-20T19:33:54.705] INFO: epoch: 1, step: 100, loss: 1.095697,  lr: 0.000011
[2023-09-20T19:34:40.903] INFO: epoch: 1, step: 200, loss: 1.023709,  lr: 0.000011
[2023-09-20T19:35:27.140] INFO: epoch: 1, step: 300, loss: 0.927682,  lr: 0.000012
[2023-09-20T19:36:13.411] INFO: epoch: 1, step: 400, loss: 1.027271,  lr: 0.000012
[2023-09-20T19:36:59.697] INFO: epoch: 1, step: 500, loss: 0.910358,  lr: 0.000013
[2023-09-20T19:37:46.138] INFO: epoch: 1, step: 600, loss: 0.935139,  lr: 0.000013
[2023-09-20T19:38:32.567] INFO: epoch: 1, step: 700, loss: 0.914457,  lr: 0.000014
[2023-09-20T19:39:19.112] INFO: epoch: 1, step: 800, loss: 0.875220,  lr: 0.000015
[2023-09-20T19:40:06.311] INFO: epoch: 1, step: 900, loss: 0.915169,  lr: 0.000015
[2023-09-20T19:40:52.961] INFO: epoch: 1, step: 1000, loss: 0.901838,  lr: 0.000016
[2023-09-20T19:41:39.623] INFO: epoch: 1, step: 1100, loss: 0.861881,  lr: 0.000016
[2023-09-20T19:42:27.146] INFO: epoch: 1, step: 1200, loss: 0.823656,  lr: 0.000017
[2023-09-20T19:43:14.359] INFO: epoch: 1, step: 1300, loss: 0.868174,  lr: 0.000017
[2023-09-20T19:44:01.158] INFO: epoch: 1, step: 1400, loss: 0.781430,  lr: 0.000018
[2023-09-20T19:44:47.740] INFO: epoch: 1, step: 1500, loss: 0.799121,  lr: 0.000018
[2023-09-20T19:45:34.215] INFO: epoch: 1, step: 1600, loss: 0.853432,  lr: 0.000019
[2023-09-20T19:46:20.696] INFO: epoch: 1, step: 1700, loss: 0.790925,  lr: 0.000020
[2023-09-20T19:47:07.294] INFO: epoch: 1, step: 1800, loss: 0.804789,  lr: 0.000020
[2023-09-20T19:47:53.931] INFO: epoch: 1, step: 1900, loss: 0.772267,  lr: 0.000021
[2023-09-20T19:48:40.504] INFO: epoch: 1, step: 2000, loss: 0.815480,  lr: 0.000021
[2023-09-20T19:49:27.008] INFO: epoch: 1, step: 2100, loss: 0.799377,  lr: 0.000022
[2023-09-20T19:50:13.474] INFO: epoch: 1, step: 2200, loss: 0.822944,  lr: 0.000022
[2023-09-20T19:51:01.143] INFO: epoch: 1, step: 2300, loss: 0.836283,  lr: 0.000023
[2023-09-20T19:51:48.669] INFO: epoch: 1, step: 2400, loss: 0.762010,  lr: 0.000024
[2023-09-20T19:52:35.697] INFO: epoch: 1, step: 2500, loss: 0.841734,  lr: 0.000024
[2023-09-20T20:06:06.616] INFO: Eval epoch time: 810912.117 ms, per step time: 162.182 ms
[2023-09-20T20:06:47.154] INFO: Result metrics for epoch 1: {'bbox_mAP': 0.3501140280926751, 'loss': 0.8704017491340638}
[2023-09-20T20:06:47.163] INFO: Train epoch time: 2098438.281 ms, per step time: 839.375 ms
[2023-09-20T20:07:35.521] INFO: epoch: 2, step: 100, loss: 0.833123,  lr: 0.000025
[2023-09-20T20:08:23.612] INFO: epoch: 2, step: 200, loss: 0.769798,  lr: 0.000025
[2023-09-20T20:09:11.178] INFO: epoch: 2, step: 300, loss: 0.835747,  lr: 0.000026
[2023-09-20T20:09:59.157] INFO: epoch: 2, step: 400, loss: 0.687634,  lr: 0.000026
[2023-09-20T20:10:46.806] INFO: epoch: 2, step: 500, loss: 0.785105,  lr: 0.000027
[2023-09-20T20:11:34.987] INFO: epoch: 2, step: 600, loss: 0.747699,  lr: 0.000027
[2023-09-20T20:12:22.667] INFO: epoch: 2, step: 700, loss: 0.817359,  lr: 0.000028
[2023-09-20T20:13:09.060] INFO: epoch: 2, step: 800, loss: 0.806359,  lr: 0.000029
[2023-09-20T20:13:57.719] INFO: epoch: 2, step: 900, loss: 0.757138,  lr: 0.000029
[2023-09-20T20:14:45.648] INFO: epoch: 2, step: 1000, loss: 0.826554,  lr: 0.000030
[2023-09-20T20:15:34.211] INFO: epoch: 2, step: 1100, loss: 0.834632,  lr: 0.000030
[2023-09-20T20:16:21.716] INFO: epoch: 2, step: 1200, loss: 0.767957,  lr: 0.000031
[2023-09-20T20:17:08.982] INFO: epoch: 2, step: 1300, loss: 0.733946,  lr: 0.000031
[2023-09-20T20:17:56.129] INFO: epoch: 2, step: 1400, loss: 0.740913,  lr: 0.000032
[2023-09-20T20:18:43.047] INFO: epoch: 2, step: 1500, loss: 0.717972,  lr: 0.000033
[2023-09-20T20:19:30.535] INFO: epoch: 2, step: 1600, loss: 0.763332,  lr: 0.000033
[2023-09-20T20:20:17.598] INFO: epoch: 2, step: 1700, loss: 0.755662,  lr: 0.000034
[2023-09-20T20:21:04.793] INFO: epoch: 2, step: 1800, loss: 0.688449,  lr: 0.000034
[2023-09-20T20:21:52.030] INFO: epoch: 2, step: 1900, loss: 0.644879,  lr: 0.000035
[2023-09-20T20:22:39.950] INFO: epoch: 2, step: 2000, loss: 0.749005,  lr: 0.000035
[2023-09-20T20:23:27.993] INFO: epoch: 2, step: 2100, loss: 0.783134,  lr: 0.000036
[2023-09-20T20:24:15.949] INFO: epoch: 2, step: 2200, loss: 0.762729,  lr: 0.000036
[2023-09-20T20:25:02.594] INFO: epoch: 2, step: 2300, loss: 0.723209,  lr: 0.000037
[2023-09-20T20:25:49.253] INFO: epoch: 2, step: 2400, loss: 0.736347,  lr: 0.000038
[2023-09-20T20:26:35.867] INFO: epoch: 2, step: 2500, loss: 0.706169,  lr: 0.000038
[2023-09-20T20:39:22.338] INFO: Eval epoch time: 765935.462 ms, per step time: 153.187 ms
[2023-09-20T20:40:03.516] INFO: Result metrics for epoch 2: {'bbox_mAP': 0.3604730881081456, 'loss': 0.7589940050363541}
[2023-09-20T20:40:03.525] INFO: Train epoch time: 1996358.991 ms, per step time: 798.544 ms
[2023-09-20T20:40:49.399] INFO: epoch: 3, step: 100, loss: 0.757715,  lr: 0.000039
[2023-09-20T20:41:35.553] INFO: epoch: 3, step: 200, loss: 0.712256,  lr: 0.000039
[2023-09-20T20:42:21.749] INFO: epoch: 3, step: 300, loss: 0.733487,  lr: 0.000040
[2023-09-20T20:43:07.988] INFO: epoch: 3, step: 400, loss: 0.722612,  lr: 0.000040
[2023-09-20T20:43:54.353] INFO: epoch: 3, step: 500, loss: 0.690824,  lr: 0.000041
[2023-09-20T20:44:40.706] INFO: epoch: 3, step: 600, loss: 0.667887,  lr: 0.000041
[2023-09-20T20:45:27.068] INFO: epoch: 3, step: 700, loss: 0.741430,  lr: 0.000042
[2023-09-20T20:46:13.398] INFO: epoch: 3, step: 800, loss: 0.742308,  lr: 0.000043
[2023-09-20T20:46:59.690] INFO: epoch: 3, step: 900, loss: 0.769529,  lr: 0.000043
[2023-09-20T20:47:46.014] INFO: epoch: 3, step: 1000, loss: 0.698827,  lr: 0.000044
[2023-09-20T20:48:32.453] INFO: epoch: 3, step: 1100, loss: 0.775155,  lr: 0.000044
[2023-09-20T20:49:18.902] INFO: epoch: 3, step: 1200, loss: 0.693283,  lr: 0.000045
[2023-09-20T20:50:05.329] INFO: epoch: 3, step: 1300, loss: 0.713412,  lr: 0.000045
[2023-09-20T20:50:51.749] INFO: epoch: 3, step: 1400, loss: 0.675248,  lr: 0.000046
[2023-09-20T20:51:38.229] INFO: epoch: 3, step: 1500, loss: 0.711307,  lr: 0.000047
[2023-09-20T20:52:24.736] INFO: epoch: 3, step: 1600, loss: 0.673766,  lr: 0.000047
[2023-09-20T20:53:11.313] INFO: epoch: 3, step: 1700, loss: 0.708616,  lr: 0.000048
[2023-09-20T20:53:58.650] INFO: epoch: 3, step: 1800, loss: 0.701241,  lr: 0.000048
[2023-09-20T20:54:45.095] INFO: epoch: 3, step: 1900, loss: 0.713135,  lr: 0.000049
[2023-09-20T20:55:31.607] INFO: epoch: 3, step: 2000, loss: 0.657496,  lr: 0.000049
[2023-09-20T20:56:18.223] INFO: epoch: 3, step: 2100, loss: 0.628995,  lr: 0.000050
[2023-09-20T20:57:04.763] INFO: epoch: 3, step: 2200, loss: 0.726784,  lr: 0.000050
[2023-09-20T20:57:51.346] INFO: epoch: 3, step: 2300, loss: 0.714132,  lr: 0.000051
[2023-09-20T20:58:37.887] INFO: epoch: 3, step: 2400, loss: 0.713294,  lr: 0.000052
[2023-09-20T20:59:24.350] INFO: epoch: 3, step: 2500, loss: 0.679350,  lr: 0.000052
[2023-09-20T21:12:15.767] INFO: Eval epoch time: 770891.760 ms, per step time: 154.178 ms
[2023-09-20T21:12:58.410] INFO: Result metrics for epoch 3: {'bbox_mAP': 0.36359686543038144, 'loss': 0.7088835831612349}
[2023-09-20T21:12:58.418] INFO: Train epoch time: 1974892.116 ms, per step time: 789.957 ms
[2023-09-20T21:13:44.372] INFO: epoch: 4, step: 100, loss: 0.670755,  lr: 0.000053
[2023-09-20T21:14:30.437] INFO: epoch: 4, step: 200, loss: 0.746679,  lr: 0.000053
[2023-09-20T21:15:16.632] INFO: epoch: 4, step: 300, loss: 0.765545,  lr: 0.000054
[2023-09-20T21:16:02.861] INFO: epoch: 4, step: 400, loss: 0.719370,  lr: 0.000054
[2023-09-20T21:16:49.116] INFO: epoch: 4, step: 500, loss: 0.631251,  lr: 0.000055
[2023-09-20T21:17:35.428] INFO: epoch: 4, step: 600, loss: 0.633735,  lr: 0.000056
[2023-09-20T21:18:21.794] INFO: epoch: 4, step: 700, loss: 0.642321,  lr: 0.000056
[2023-09-20T21:19:08.027] INFO: epoch: 4, step: 800, loss: 0.731721,  lr: 0.000057
[2023-09-20T21:19:54.326] INFO: epoch: 4, step: 900, loss: 0.751165,  lr: 0.000057
[2023-09-20T21:20:40.657] INFO: epoch: 4, step: 1000, loss: 0.647908,  lr: 0.000058
[2023-09-20T21:21:26.988] INFO: epoch: 4, step: 1100, loss: 0.699023,  lr: 0.000058
[2023-09-20T21:22:13.335] INFO: epoch: 4, step: 1200, loss: 0.766589,  lr: 0.000059
[2023-09-20T21:22:59.690] INFO: epoch: 4, step: 1300, loss: 0.610706,  lr: 0.000059
[2023-09-20T21:23:46.004] INFO: epoch: 4, step: 1400, loss: 0.713266,  lr: 0.000060
[2023-09-20T21:24:32.332] INFO: epoch: 4, step: 1500, loss: 0.672221,  lr: 0.000061
[2023-09-20T21:25:18.668] INFO: epoch: 4, step: 1600, loss: 0.650037,  lr: 0.000061
[2023-09-20T21:26:05.042] INFO: epoch: 4, step: 1700, loss: 0.669658,  lr: 0.000062
[2023-09-20T21:26:51.407] INFO: epoch: 4, step: 1800, loss: 0.634266,  lr: 0.000062
[2023-09-20T21:27:37.753] INFO: epoch: 4, step: 1900, loss: 0.657199,  lr: 0.000063
[2023-09-20T21:28:24.149] INFO: epoch: 4, step: 2000, loss: 0.602266,  lr: 0.000063
[2023-09-20T21:29:10.536] INFO: epoch: 4, step: 2100, loss: 0.637381,  lr: 0.000064
[2023-09-20T21:29:56.927] INFO: epoch: 4, step: 2200, loss: 0.630775,  lr: 0.000065
[2023-09-20T21:30:43.275] INFO: epoch: 4, step: 2300, loss: 0.689695,  lr: 0.000065
[2023-09-20T21:31:29.690] INFO: epoch: 4, step: 2400, loss: 0.721765,  lr: 0.000066
[2023-09-20T21:32:16.019] INFO: epoch: 4, step: 2500, loss: 0.700184,  lr: 0.000066
[2023-09-20T21:45:00.194] INFO: Eval epoch time: 763624.569 ms, per step time: 152.725 ms
[2023-09-20T21:45:41.358] INFO: Result metrics for epoch 4: {'bbox_mAP': 0.36431394369567166, 'loss': 0.6798192100167274}
[2023-09-20T21:45:41.366] INFO: Train epoch time: 1962946.485 ms, per step time: 785.179 ms
[2023-09-20T21:46:27.288] INFO: epoch: 5, step: 100, loss: 0.702861,  lr: 0.000067
[2023-09-20T21:47:13.313] INFO: epoch: 5, step: 200, loss: 0.616629,  lr: 0.000067
[2023-09-20T21:48:01.243] INFO: epoch: 5, step: 300, loss: 0.734315,  lr: 0.000068
[2023-09-20T21:48:48.765] INFO: epoch: 5, step: 400, loss: 0.680448,  lr: 0.000068
[2023-09-20T21:49:36.411] INFO: epoch: 5, step: 500, loss: 0.631517,  lr: 0.000069
[2023-09-20T21:50:24.026] INFO: epoch: 5, step: 600, loss: 0.653994,  lr: 0.000070
[2023-09-20T21:51:11.575] INFO: epoch: 5, step: 700, loss: 0.706293,  lr: 0.000070
[2023-09-20T21:51:59.022] INFO: epoch: 5, step: 800, loss: 0.670356,  lr: 0.000071
[2023-09-20T21:52:46.697] INFO: epoch: 5, step: 900, loss: 0.706946,  lr: 0.000071
[2023-09-20T21:53:34.777] INFO: epoch: 5, step: 1000, loss: 0.695594,  lr: 0.000072
[2023-09-20T21:54:22.661] INFO: epoch: 5, step: 1100, loss: 0.691149,  lr: 0.000072
[2023-09-20T21:55:10.368] INFO: epoch: 5, step: 1200, loss: 0.725784,  lr: 0.000073
[2023-09-20T21:55:58.265] INFO: epoch: 5, step: 1300, loss: 0.688137,  lr: 0.000074
[2023-09-20T21:56:46.062] INFO: epoch: 5, step: 1400, loss: 0.610064,  lr: 0.000074
[2023-09-20T21:57:33.911] INFO: epoch: 5, step: 1500, loss: 0.716337,  lr: 0.000075
[2023-09-20T21:58:22.038] INFO: epoch: 5, step: 1600, loss: 0.642340,  lr: 0.000075
[2023-09-20T21:59:09.725] INFO: epoch: 5, step: 1700, loss: 0.630346,  lr: 0.000076
[2023-09-20T21:59:57.516] INFO: epoch: 5, step: 1800, loss: 0.681286,  lr: 0.000076
[2023-09-20T22:00:44.737] INFO: epoch: 5, step: 1900, loss: 0.660155,  lr: 0.000077
[2023-09-20T22:01:32.335] INFO: epoch: 5, step: 2000, loss: 0.617997,  lr: 0.000078
[2023-09-20T22:02:20.008] INFO: epoch: 5, step: 2100, loss: 0.682558,  lr: 0.000078
[2023-09-20T22:03:07.495] INFO: epoch: 5, step: 2200, loss: 0.629569,  lr: 0.000079
[2023-09-20T22:03:55.257] INFO: epoch: 5, step: 2300, loss: 0.744592,  lr: 0.000079
[2023-09-20T22:04:42.931] INFO: epoch: 5, step: 2400, loss: 0.605105,  lr: 0.000080
[2023-09-20T22:05:30.202] INFO: epoch: 5, step: 2500, loss: 0.666540,  lr: 0.000080
[2023-09-20T22:18:35.484] INFO: Eval epoch time: 784728.543 ms, per step time: 156.946 ms
[2023-09-20T22:19:17.468] INFO: Result metrics for epoch 5: {'bbox_mAP': 0.3669068630581188, 'loss': 0.6716364663243294}
[2023-09-20T22:19:17.477] INFO: Train epoch time: 2016109.120 ms, per step time: 806.444 ms
```

### [Transfer Training](#contents)

You can train your own model based on either pretrained classification model
or pretrained detection model. You can perform transfer training by following
steps.

1. Prepare your dataset.
2. Change configuration YAML file according to your own dataset, especially
 the change `num_classes` value and `coco_classes` list.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by
 `pretrained` argument. Transfer training means a new training job, so just set
 `finetune` 1.
4. Run training.

### [Distribute training](#contents)

Distribute training mode (OpenMPI must be installed):

```shell
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for these scripts:

* `CONFIG_PATH`: path to config file.
* `DEVICE_NUM`: number of devices.
* `TRAIN_DATA`: path to train dataset.
* `VAL_DATA`: path to validation dataset.
* `TRAIN_OUT`: path to folder with training experiments.
* `BRIEF`: short experiment name.
* `PRETRAINED_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

## [Evaluation](#contents)

### [Evaluation process](#contents)

#### [Evaluation on GPU](#contents)

```shell
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (Optional)[PREDICTION_PATH]
```

We need four parameters for this script.

* `CONFIG_PATH`: path to config file.
* `VAL_DATA`: the absolute path for dataset subset (validation).
* `CHECKPOINT_PATH`: path to checkpoint.
* `PREDICTION_PATH`: path to file with predictions JSON file (predictions may
 be saved to this file and loaded after).

> checkpoint can be produced in training process.

### [Evaluation result](#contents)

Result for GPU:

```log
total images num:  5000
Processing, please wait a moment.
100%|██████████| 5000/5000 [12:56<00:00,  6.44it/s]
Loading and preparing results...
DONE (t=0.98s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=34.59s).
Accumulating evaluation results...
DONE (t=5.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.208
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
Bbox eval result: 0.39013745180978804

Evaluation done!

Done!
Time taken: 826 seconds
```

## [Inference](#contents)

### [Inference process](#contents)

#### [Inference on GPU](#contents)

Run model inference from retinanet directory:

```bash
bash scripts/run_infer_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH] [PRED_INPUT] [PRED_OUTPUT]
```

We need 4 parameters for these scripts:

* `CONFIG_FILE`： path to config file.
* `CHECKPOINT_PATH`: path to saved checkpoint.
* `PRED_INPUT`: path to input folder or image.
* `PRED_OUTPUT`: path to output JSON file.

### [Inference result](#contents)

Predictions will be saved in JSON file. File content is list of predictions
for each image. It's supported predictions for folder of images (png, jpeg
file in folder root) and single image.

Typical outputs of such script for single image:

```log
{
 "/data/coco-2017/validation/data/000000110042.jpg": {
  "height": 640,
  "width": 425,
  "predictions": [
   {
    "bbox": {
     "x_min": 135.43885803222656,
     "y_min": 440.6851806640625,
     "width": 124.15153503417969,
     "height": 171.2314453125
    },
    "class": {
     "label": 61,
     "category_id": "unknown",
     "name": "toilet"
    },
    "score": 0.7635552287101746
   },
   {
    "bbox": {
     "x_min": 81.5470199584961,
     "y_min": 107.1919174194336,
     "width": 30.140884399414062,
     "height": 56.633323669433594
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "score": 0.5444254875183105
   },
   {
    "bbox": {
     "x_min": 290.54443359375,
     "y_min": 170.9741668701172,
     "width": 54.2479248046875,
     "height": 59.4246826171875
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "score": 0.4894554018974304
   },
   ...
   {
    "bbox": {
     "x_min": 0.0,
     "y_min": 57.74102783203125,
     "width": 59.74127960205078,
     "height": 543.6371459960938
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "score": 0.12632331252098083
   },
   {
    "bbox": {
     "x_min": 278.521728515625,
     "y_min": 224.903564453125,
     "width": 32.63568115234375,
     "height": 25.937530517578125
    },
    "class": {
     "label": 60,
     "category_id": "unknown",
     "name": "dining table"
    },
    "score": 0.12562936544418335
   },
   {
    "bbox": {
     "x_min": 74.56842041015625,
     "y_min": 377.7273864746094,
     "width": 31.203826904296875,
     "height": 30.24652099609375
    },
    "class": {
     "label": 39,
     "category_id": "unknown",
     "name": "bottle"
    },
    "score": 0.1245490238070488
   }
  ]
 }
}
```

Typical outputs for folder with images:

```log
{
 "/data/coco-2017/validation/data/000000194832.jpg": {
  "height": 425,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 282.0591735839844,
     "y_min": 91.51692962646484,
     "width": 57.5623779296875,
     "height": 42.47502899169922
    },
    "class": {
     "label": 62,
     "category_id": "unknown",
     "name": "tv"
    },
    "score": 0.6084915399551392
   },
   {
    "bbox": {
     "x_min": 427.37567138671875,
     "y_min": 201.2731170654297,
     "width": 211.71630859375,
     "height": 223.7268829345703
    },
    "class": {
     "label": 13,
     "category_id": "unknown",
     "name": "bench"
    },
    "score": 0.5938668251037598
   },
   {
    "bbox": {
     "x_min": 0.0,
     "y_min": 203.22372436523438,
     "width": 217.7593231201172,
     "height": 221.77627563476562
    },
    "class": {
     "label": 13,
     "category_id": "unknown",
     "name": "bench"
    },
    "score": 0.5468060970306396
   },
   ...
   {
    "bbox": {
     "x_min": 85.53697967529297,
     "y_min": 90.60108947753906,
     "width": 41.06538391113281,
     "height": 26.322219848632812
    },
    "class": {
     "label": 62,
     "category_id": "unknown",
     "name": "tv"
    },
    "score": 0.1436769813299179
   }
  ]
 },
 "/data/coco-2017/validation/data/000000104572.jpg": {
  "height": 419,
  "width": 640,
  "predictions": [
   ...
  ]
 },
 "/data/coco-2017/validation/data/000000463647.jpg": {
  "height": 480,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 294.2091064453125,
     "y_min": 57.312034606933594,
     "width": 208.81170654296875,
     "height": 88.27271270751953
    },
    "class": {
     "label": 7,
     "category_id": "unknown",
     "name": "truck"
    },
    "score": 0.5929898619651794
   },
   {
    "bbox": {
     "x_min": 458.8509216308594,
     "y_min": 59.58843994140625,
     "width": 180.55007934570312,
     "height": 159.40838623046875
    },
    "class": {
     "label": 7,
     "category_id": "unknown",
     "name": "truck"
    },
    "score": 0.5492860674858093
   },
   ...
   {
    "bbox": {
     "x_min": 526.8856811523438,
     "y_min": 55.33305740356445,
     "width": 10.030517578125,
     "height": 7.845390319824219
    },
    "class": {
     "label": 2,
     "category_id": "unknown",
     "name": "car"
    },
    "score": 0.10228123515844345
   },
   {
    "bbox": {
     "x_min": 184.76625061035156,
     "y_min": 50.797733306884766,
     "width": 43.61712646484375,
     "height": 14.54983139038086
    },
    "class": {
     "label": 2,
     "category_id": "unknown",
     "name": "car"
    },
    "score": 0.10163320600986481
   }
  ]
 },
 "/data/coco-2017/validation/data/000000276285.jpg": {
  "height": 640,
  "width": 427,
  "predictions": [
   ...
  ]
 },
 "/data/coco-2017/validation/data/000000190676.jpg": {
  "height": 264,
  "width": 640,
  "predictions": [
   ...
  ]
 },
 "/data/coco-2017/validation/data/000000492284.jpg": {
  "height": 480,
  "width": 640,
  "predictions": [
   ...
  ]
 },
 ...
}
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | GPU                                                                                                              |
|---------------------|------------------------------------------------------------------------------------------------------------------|
| Model Version       | Mask RCNN SpineNet 49 640x640                                                                                    |
| Resource            | NVIDIA GeForce RTX 3090 (x4)                                                                                     |
| Uploaded Date       | 13/10/2023 (day/month/year)                                                                                      |
| MindSpore Version   | 2.1.0                                                                                                            |
| Dataset             | COCO2017                                                                                                         |
| Pretrained          | noised checkpoint (mAP=41.1%)                                                                                    |
| Training Parameters | epoch = 27, batch_size = 8, accumulate_step = 4 (per device)                                                     |
| Optimizer           | SGD (momentum)                                                                                                   |
| Loss Function       | SigmoidFocalClassificationLoss, SmoothL1Loss                                                                     |
| Speed               | 4pcs: 1276.25 ms/step                                                                                            |
| Total time          | 4pcs: 30h 11m 47s                                                                                                |
| outputs             | mAP                                                                                                              |
| mAP                 | 41.4                                                                                                             |
| Model for inference | 651.5M(.ckpt file)                                                                                               |
| configuration       | retinanet_spinenet_49_640_dist_noised_experiment.yaml                                                            |
| Scripts             |                                                                                                                  |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use
random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
