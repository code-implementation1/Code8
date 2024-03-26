# Contents

* [Contents](#contents)
    * [SpineNet MaskRCNN Description](#maskrcnn-description)
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

## [SpineNet MaskRCNN Description](#contents)

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

**Overview of the pipeline of SpineNet MaskRCNN:**
MaskRCNN is a conceptually simple, flexible, and general framework for object
instance segmentation. The approach efficiently detects objects in an image
while simultaneously generating a high-quality segmentation mask for each
instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a
branch for predicting an object mask in parallel with the existing branch for
bounding box recognition. Mask R-CNN is simple to train and adds only a small
overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to
generalize to other tasks, e.g., allowing to estimate human poses in the same
framework.
It shows top results in all three tracks of the COCO suite of challenges,
including instance segmentation, bounding box object detection, and person
keypoint detection. Without bells and whistles, Mask R-CNN outperforms all
existing, single-model entries on every task, including the COCO 2016
challenge winners.

Region proposals are obtained from RPN and used for RoI feature extraction
from the output feature maps of a CNN backbone. The RoI features are used to
perform classification and localization and mask computation.

MaskRCNN result prediction pipeline:

1. SpineNet backbone.
2. RPN.
3. Proposal generator.
4. ROI extractor (based on ROIAlign operation).
5. Bounding box head.
6. multiclass NMS (reduce number of proposed boxes and omit objects with low
 confidence).
7. ROI extractor (based on ROIAlign operation).
8. Mask head.

MaskRCNN result training pipeline:

1. SpineNet backbone.
2. RPN.
3. RPN Assigner+Sampler.
4. RPN Classification + Localization losses.
5. Proposal generator.
6. RCNN Assigner+Sampler.
7. ROI extractor (based on ROIAlign operation).
8. Bounding box head.
9. RCNN Classification + Localization losses.
10. ROI extractor (based on ROIAlign operation).
11. Mask head.
12. Mask loss.

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
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (OPTIONAL)[PREDICTION_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```log
SpineNet
└── mask_rcnn
    ├── configs
    │   ├── mask_rcnn_143_1280.yaml                                 ## SpineNet143 configuration.
    │   ├── mask_rcnn_49_640.yaml                                   ## SpineNet49 configuration.
    │   ├── mask_rcnn_49S_640.yaml                                  ## SpineNet49S configuration.
    │   └── mask_rcnn_96_1024.yaml                                  ## SpineNet96 configuration.
    ├── scripts
    │   ├── run_distribute_train_gpu.sh                             ## Bash script for distributed training on gpu.
    │   ├── run_eval_gpu.sh                                         ## Bash script for eval on gpu.
    │   ├── run_infer_gpu.sh                                        ## Bash script for gpu model inference.
    │   └── run_standalone_train_gpu.sh                             ## Bash script for training on gpu.
    ├── src
    │   ├── blocks
    │   │   ├── anchor_generator
    │   │   │   ├── anchor_generator.py                         ## Anchor generator.
    │   │   │   └── __init__.py
    │   │   ├── assigners_samplers
    │   │   │   ├── assigner_sampler.py                         ## Wrapper for assigner and sampler.
    │   │   │   ├── __init__.py
    │   │   │   ├── mask_assigner_sampler.py                    ## Wrapper for assigner and sampler working with masks too.
    │   │   │   ├── max_iou_assigner.py                         ## MaxIOUAssigner.
    │   │   │   └── random_sampler.py                           ## Random Sampler.
    │   │   ├── backbones
    │   │   │   ├── __init__.py
    │   │   │   └── spinenet.py                                 ## Implemented SpineNet backbone.
    │   │   ├── bbox_coders
    │   │   │   ├── bbox_coder.py                               ## Bounding box coder.
    │   │   │   └── __init__.py
    │   │   ├── bbox_heads
    │   │   │   ├── convfc_bbox_head.py                         ## Bounding box head.
    │   │   │   └── __init__.py
    │   │   ├── dense_heads
    │   │   │   ├── __init__.py
    │   │   │   ├── proposal_generator.py                       ## Proposal generator (part of RPN).
    │   │   │   └── rpn.py                                      ## Region Proposal Network.
    │   │   ├── initialization
    │   │   │   ├── initialization.py                           ## Weight initialization functional.
    │   │   │   └── __init__.py
    │   │   ├── layers
    │   │   │   ├── conv_module.py                              ## Convolution module.
    │   │   │   └── __init__.py
    │   │   ├── mask_heads
    │   │   │   ├── fcn_mask_head.py                            ## Mask head.
    │   │   │   └── __init__.py
    │   │   ├── roi_extractors
    │   │   │   ├── __init__.py
    │   │   │   └── single_layer_roi_extractor.py               ## Single ROI Extractor.
    │   │   ├── mask_rcnn.py                                      ## MaskRCNN model.
    │   │   └── __init__.py
    │   ├── model_utils
    │   │   ├── config.py                                     ## Configuration file parsing utils.
    │   │   ├── device_adapter.py                             ## File to adapt current used devices.
    │   │   ├── __init__.py
    │   │   ├── local_adapter.py                              ## File to work with local devices.
    │   │   └── moxing_adapter.py                             ## File to work with model arts devices.
    │   ├── callback.py                                             ## Callbacks.
    │   ├── common.py                                               ## Common functional with common setting.
    │   ├── dataset.py                                              ## Images loading and preprocessing.
    │   ├── detecteval.py                                           ## DetectEval class to analyze predictions
    │   ├── eval_utils.py                                           ## Evaluation metrics utilities.
    │   ├── __init__.py
    │   ├── lr_schedule.py                                          ## Optimizer settings.
    │   ├── mlflow_funcs.py                                         ## mlflow utilities.
    │   └── network_define.py                                       ## Model wrappers for training.
    ├── draw_predictions.py                                           ## Draw results of infer.py script on image.
    ├── eval.py                                                       ## Run models evaluation.
    ├── infer.py                                                      ## Make predictions for models.
    ├── __init__.py
    ├── README.md                                                     ## MaskRCNN definition.
    ├── requirements.txt                                              ## Dependencies.
    └── train.py                                                      ## Train script.
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```shell
train_outputs: '/data/mask_rcnn_models'
brief: 'gpu-1'
device_target: GPU
mode: 'graph'
# ==============================================================================
detector: 'mask_rcnn'

# backbone
backbone:
  type: 'spinenet'                                                ## Backbone type
  arch: '49'                                                      ## Backbone architecture
  norm_cfg:                                                       ## Batch normalization parameter
    type: 'BN'
    momentum: 0.01
    eps: 0.001

# rpn
rpn:
  in_channels: 256                                              ## Input channels
  feat_channels: 256                                            ## Number of channels in intermediate feature map in RPN
  num_classes: 1                                                ## Output classes number
  bbox_coder:
    target_means: [0., 0., 0., 0.]                              ## Parameter for bbox encoding (RPN targets generation)
    target_stds: [1.0, 1.0, 1.0, 1.0]                           ## Parameter for bbox encoding (RPN targets generation)
  loss_cls:
    loss_weight: 1.0                                            ## RPN classification loss weight
  loss_bbox:
    loss_weight: 1.0                                            ## RPN localization loss weight
    beta: 0.111111111111                                        ## RPN localization loss parameter (smooth loss)
  anchor_generator:
    scales: [3]                                                 ## Anchor scales
    strides: [8, 16, 32, 64, 128]                               ## Anchor ratios
    ratios: [0.5, 1.0, 2.0]                                     ## Anchor strides for each feature map


bbox_head:
  num_shared_convs: 4                                           ## Number of shared convolution layers
  num_shared_fcs: 1                                             ## Number of shared dense layers
  in_channels: 256                                              ## Number of input channels
  conv_out_channels: 256                                        ## Number of convolution layers' output channels
  fc_out_channels: 1024                                         ## Number of intermediate channels before classification
  roi_feat_size: 7                                              ## Input feature map side length
  reg_class_agnostic: False
  bbox_coder:
    target_means: [0.0, 0.0, 0.0, 0.0]                          ## Bounding box coder parameter
    target_stds: [0.1, 0.1, 0.2, 0.2]                           ## Bounding box coder parameter
  loss_cls:
    loss_weight: 1.0                                            ## Classification loss weight
  loss_bbox:
    beta: 1.0
    loss_weight: 1.0                                            ## Localization loss weight
  norm_cfg:
    type: 'BN'
    momentum: 0.01
    eps: 0.001

mask_head:
  num_convs: 4                                                 ## Number of convolution layers
  in_channels: 256                                             ## Number of input channels
  conv_out_channels: 256                                       ## Number of intermediate layers output channels
  norm_cfg:
    type: 'BN'
    momentum: 0.01
    eps: 0.001
  loss_mask:
    loss_weight: 1.0                                           ## Mask loss weight

# roi_align
roi:                                                                    ## RoI extractor parameters
  roi_layer: {type: 'RoIAlign', out_size: 7, sample_num: 2}             ## RoI configuration
  out_channels: 256                                                     ## out roi channels
  featmap_strides: [8, 16, 32, 64]                                      ## strides for RoIAlign layer
  finest_scale: 56                                                      ## parameter that define roi level
  sample_num: 640

mask_roi:                                                               ## RoI extractor parameters for masks head
  roi_layer: {type: 'RoIAlign', out_size: 14, sample_num: 2}            ## RoI configuration
  out_channels: 256                                                     ## out roi channels
  finest_scale: 56                                                      ## parameter that define roi level
  featmap_strides: [8, 16, 32, 64]                                      ## strides for RoIAlign layer
  sample_num: 128

train_cfg:
  rpn:
    assigner:
      pos_iou_thr: 0.7                                            ## IoU threshold for negative bboxes
      neg_iou_thr: 0.3                                            ## IoU threshold for positive bboxes
      min_pos_iou: 0.3                                            ## Minimum iou for a bbox to be considered as a positive bbox
      match_low_quality: True                                     ## Allow low quality matches
    sampler:
      num: 256                                                    ## Number of chosen samples
      pos_fraction: 0.5                                           ## Fraction of positive samples
      neg_pos_ub: -1                                              ## Max positive-negative samples ratio
      add_gt_as_proposals: False                                  ## Allow low quality matches
  rpn_proposal:
      nms_pre: 2000                                              ## max number of samples per level
      max_per_img: 2000                                          ## max number of output samples
      iou_threshold: 0.7                                         ## NMS threshold for proposal generator
      min_bbox_size: 0                                           ## min bboxes size
  rcnn:
      assigner:
        pos_iou_thr: 0.5                                            ## IoU threshold for negative bboxes
        neg_iou_thr: 0.5                                            ## IoU threshold for positive bboxes
        min_pos_iou: 0.5                                            ## Minimum iou for a bbox to be considered as a positive bbox
        match_low_quality: True                                     ## Allow low quality matches
      sampler:
        num: 512                                                    ## Number of chosen samples
        pos_fraction: 0.25                                          ## Fraction of positive samples
        neg_pos_ub: -1                                              ## Max positive-negative samples ratio
        add_gt_as_proposals: True                                   ## Allow low quality matches
      mask_size: 28                                                 ## Output mask size

test_cfg:
  rpn:
    nms_pre: 1000                                                 ## max number of samples per level
    max_per_img: 1000                                             ## max number of output samples
    iou_threshold: 0.7                                            ## NMS threshold for proposal generator
    min_bbox_size: 0                                              ## min bboxes size
  rcnn:
    score_thr: 0.05                                               ## Confidence threshold
    iou_threshold: 0.5                                            ## IOU threshold
    max_per_img: 100                                              ## Max number of output bboxes
    mask_thr_binary: 0.5                                          ## mask threshold for masks

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
[2023-08-16T06:57:11.629] INFO: Creating network...
[2023-08-16T06:57:18.088] INFO: Number of parameters: 40785333
[2023-08-16T06:57:18.089] INFO: Device type: GPU
[2023-08-16T06:57:18.089] INFO: Creating criterion, lr and opt objects...
[2023-08-16T06:57:19.003] INFO:     Done!

[2023-08-16T06:57:19.895] INFO: Directory created: /experiments/models/230816_MaskRCNN_gpu-1_1024x1024/best_ckpt
[2023-08-16T06:57:19.895] WARNING: Directory already exists: /experiments/models/230816_MaskRCNN_gpu-1_1024x1024/best_ckpt
[2023-08-16T07:01:06.830] INFO: epoch: 1, step: 100, loss: 1.735162,  lr: 0.017150
[2023-08-16T07:02:23.590] INFO: epoch: 1, step: 200, loss: 1.600231,  lr: 0.020300
[2023-08-16T07:03:40.426] INFO: epoch: 1, step: 300, loss: 1.553944,  lr: 0.023450
...
[2023-08-16T10:59:30.337] INFO: epoch: 1, step: 19300, loss: 1.099769,  lr: 0.140000
[2023-08-16T11:00:43.357] INFO: epoch: 1, step: 19400, loss: 1.104567,  lr: 0.140000
[2023-08-16T11:01:56.291] INFO: epoch: 1, step: 19500, loss: 1.101969,  lr: 0.140000
[2023-08-16T11:03:09.049] INFO: epoch: 1, step: 19600, loss: 1.122950,  lr: 0.140000
[2023-08-16T11:04:22.000] INFO: epoch: 1, step: 19700, loss: 1.115548,  lr: 0.140000
[2023-08-16T11:18:07.766] INFO: Eval epoch time: 815743.776 ms, per step time: 163.149 ms
[2023-08-16T11:20:12.022] INFO: Result metrics for epoch 1: {'bbox_mAP': 0.05560288702265213, 'loss': 1.2341959740268515, 'seg_mAP': 0.0420292550358416}
[2023-08-16T11:20:12.031] INFO: Train epoch time: 15771365.491 ms, per step time: 800.008 ms
[2023-08-16T11:21:25.292] INFO: epoch: 2, step: 100, loss: 1.104041,  lr: 0.140000
...
[2023-08-17T08:32:10.299] INFO: Eval epoch time: 753522.395 ms, per step time: 150.704 ms
[2023-08-17T08:34:00.630] INFO: Result metrics for epoch 6: {'bbox_mAP': 0.12116487544031797, 'loss': 0.9886155926583806, 'seg_mAP': 0.09276879073783478}
[2023-08-17T08:34:00.640] INFO: Train epoch time: 15233386.439 ms, per step time: 772.719 ms
[2023-08-17T08:35:12.954] INFO: epoch: 7, step: 100, loss: 0.995060,  lr: 0.140000
...
[2023-08-17T12:33:01.045] INFO: epoch: 7, step: 19700, loss: 0.983506,  lr: 0.140000
[2023-08-17T12:45:24.618] INFO: Eval epoch time: 730170.531 ms, per step time: 146.034 ms
[2023-08-17T12:47:10.485] INFO: Result metrics for epoch 7: {'bbox_mAP': 0.11821124612903977, 'loss': 0.9834115390054616, 'seg_mAP': 0.09215658278533113}
```

### [Transfer Training](#contents)

You can train your own model based on either pretrained classification model
or pretrained detection model. You can perform transfer training by following
steps.

1. Prepare your dataset.
2. Change configuraino YAML file according to your own dataset, especially the
 change `num_classes` value and `coco_classes` list.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by
 `pretrained` argument. Transfer training means a new training job, so just set
 `finetune` 1.
4. Run training.

### [Distribute training](#contents)

Distribute training mode (OpenMPI must be installed):

```shell
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for this script:

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
100%|██████████| 5000/5000 [16:08<00:00,  5.16it/s]

Loading and preparing results...
DONE (t=0.53s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=23.45s).
Accumulating evaluation results...
DONE (t=3.86s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.600
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
Bbox eval result: 0.3951599986193606
Loading and preparing results...
DONE (t=1.36s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=27.01s).
Accumulating evaluation results...
DONE (t=3.69s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.137
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.538
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.251
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.662
Segmentation eval result: 0.3398017528937522

Evaluation done!

Done!
Time taken: 1039 seconds
```

## [Inference](#contents)

### [Inference process](#contents)

#### [Inference on GPU](#contents)

Run model inference from mask_rcnn directory:

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
     "x_min": 291.701171875,
     "y_min": 171.0816650390625,
     "width": 62.83697509765625,
     "height": 94.95254516601562
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "mask": {
     "size": [
      640,
      425
     ],
     "counts": "jZg56hc02N3N1N2N2O1O1O1O1N2N2N2N3M3N2N2N3Mo0QO5K1N2M0O3O00K6O1010O0O2000000002N001O001]N\\^Oh0ea0SOa^Ok0aa0POc^OP1Wb0O1O2N3M=C1O2N0O2O1N2N2N1N3M6Hhd^1"
    },
    "score": 0.994330883026123
   },
   {
    "bbox": {
     "x_min": 134.09532165527344,
     "y_min": 443.8565673828125,
     "width": 120.94125366210938,
     "height": 167.806640625
    },
    "class": {
     "label": 61,
     "category_id": "unknown",
     "name": "toilet"
    },
    "mask": {
     "size": [
      640,
      425
     ],
     "counts": "RVd25kc00O2M6H:J8H6J5K7I5K2N2N2N3M1O1O1O2N1O2N002N1O1O2N2N:F?A:F;E8H7I4L3M4L2N2N1O1O2N1O001O1O001O004L000000000000000000000000000000000000000003M1O1O001O1O002N1O1O1O001O001O1O1O1O0000001O0000000ZMVA6j>H_A1a>NdAN\\>1jAJW>5QBCo=<ZBYOi=e0]BVOd=j0_BROb=m0dBmN]=R1lBdNW=[1g2O2O1N1N3N1O1O2N1N2N3M3L5L2000Y]Z3"
    },
    "score": 0.9927830696105957
   },
   ...
   {
    "bbox": {
     "x_min": 317.4268493652344,
     "y_min": 244.74166870117188,
     "width": 29.49517822265625,
     "height": 20.6441650390625
    },
    "class": {
     "label": 56,
     "category_id": "unknown",
     "name": "chair"
    },
    "mask": {
     "size": [
      640,
      425
     ],
     "counts": "moV67fc03O1O10O0N3N200O1000000001O00001O001O1O001OO1O2N1O1OYha1"
    },
    "score": 0.05061636120080948
   },
   {
    "bbox": {
     "x_min": 0.0,
     "y_min": 115.16654205322266,
     "width": 18.798320770263672,
     "height": 102.94390106201172
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "mask": {
     "size": [
      640,
      425
     ],
     "counts": "m4l1Ub0O2N3M5G<F6L2M4L4@>N3N2M[d05V]o7"
    },
    "score": 0.05020664632320404
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
     "x_min": 281.8936767578125,
     "y_min": 91.69677734375,
     "width": 59.76153564453125,
     "height": 37.133941650390625
    },
    "class": {
     "label": 62,
     "category_id": "unknown",
     "name": "tv"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "aae3P1X<2O1N100000000001O0000000000000000000000000000000000000000000000001O000000000000000000000000000000000000001N1AnC@Y<<i]l3"
    },
    "score": 0.9923324584960938
   },
   {
    "bbox": {
     "x_min": 3.894012451171875,
     "y_min": 0.2272796630859375,
     "width": 626.9788818359375,
     "height": 425.0
    },
    "class": {
     "label": 5,
     "category_id": "unknown",
     "name": "bus"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "kW1RjS8Sb4"
    },
    "score": 0.8347064852714539
   },
   {
    "bbox": {
     "x_min": 0.26325225830078125,
     "y_min": 203.27545166015625,
     "width": 214.13894653320312,
     "height": 226.0927734375
    },
    "class": {
     "label": 56,
     "category_id": "unknown",
     "name": "chair"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "l6]6l60000000O10000O100O1000000O10000O100O100O10000O10000O1000000000000O1001O000000000000001O001O001O001O1O1O1O2N1O1O1O1O1O2N2N2N1O1O1O1O1O1O2N1O2N1O1O1O2N3M2N2N2N2N1O2N2N2N2N2N2N1O1O2N1O3M1O1O1O1O1O002N1O1O001O001O0000001O0000000000000000000000001O0000000000000000000000000000001O00001O00000000000000O1O1O1O100O100O10000000000O10000000L4O1O1N2O1O0O2O1N2N2N2M3N2N2N2O1N3M2N2L4L4N2O1N3M2N2O1N2O2M2O1N3N1N2N3M2M4M2M4G8M4L4J6J6N100000a``5"
    },
    "score": 0.8291604518890381
   },
   {
    "bbox": {
     "x_min": 56.899818420410156,
     "y_min": 217.57296752929688,
     "width": 232.74508666992188,
     "height": 121.5859375
    },
    "class": {
     "label": 13,
     "category_id": "unknown",
     "name": "bench"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "_Vk07Q=2N2O1O011N2O0O2N001O01O000O101N1O100O11N3N2M2O1N101O0O1000000000000000000O01O100O0O2N2O10000000000000000000000000000000RE_OT9a0jFAV9?jFAV9?jFAV9?jFAV9?jFAV9?jFAV9?jFAV9>kFBU9>kFBU9>kFBU9?kF@U9`0i1100000000000000000001O00000000O10000000000000000000000001O00001O000000001O000000001O0000000000001O001O000]DXOe:h0[EXOe:i0ZEWOf:i0YEXOg:h0YEYOf:g0ZEYOf:g0ZEYOf:g0ZEYOe:h0ZEYOf:g0ZEZOe:f0[EZOe:f0[EZOe:f0TEXO@4OMV;g0ZEIf:7SEWOIc0S;7TEWOGd0T;5UEXOFd0T;4VE2h:OXE2g:NYE3f:MZE5d:K\\E7b:I^E:_:FaE>[:BeEi0P:WOPFm0l9TOSFo0j9QOVFQ1h9oNWFR1i9nNWFR1i9nNWFR1i9oNUFR1k9nNUFR1k9nNUFR1k9oNSFR1m9POQFP1P:oNoER1Q:oNnEQ1R:oNnEQ1R:oNnER1Q:nNoER1R:R10000001O001O001O1O001O1O1O001O1N2O1O1O1O1O1O2N4L3M4L1O000000000000O100001O000000000QNPEe1P;ZNREe1n:ZNSEf1m:YNVEe1k:XNXEf1Y;M2M4K:cNUD000M;ked4"
    },
    "score": 0.7281296849250793
   },
   ...
   {
    "bbox": {
     "x_min": 8.95785140991211,
     "y_min": 273.6311950683594,
     "width": 88.79264831542969,
     "height": 118.5775146484375
    },
    "class": {
     "label": 58,
     "category_id": "unknown",
     "name": "potted plant"
    },
    "mask": {
     "size": [
      419,
      640
     ],
     "counts": "Qk65l<4M6I4QD_OA3];c0jDIS;:eDMY;o0O1O1O1N3N2N1O1O2O0NN201O2M22O1fEQN^9P2aFPN^9R2aFnM_9S2_FnMa9S2_FlMa9T2_FlM`9T2aFlM_9U2SFmMX:W2aEnM]:]2N10001O01O00010O01eE[MW:i2O00001O01O010O1O2N2N10O0101N1O0010O00ZNcEf0^:VOgEh0Z:SOkEm0U:POoEn0R:POoEP1R:kNRFU1o:O00OO1M5L:F8ZOgC6g<LbSQ7"
    },
    "score": 0.06213695555925369
   },
   {
    "bbox": {
     "x_min": 366.47119140625,
     "y_min": 178.13882446289062,
     "width": 78.2413330078125,
     "height": 67.73672485351562
    },
    "class": {
     "label": 59,
     "category_id": "unknown",
     "name": "bed"
    },
    "mask": {
     "size": [
      419,
      640
     ],
     "counts": "]Ph44^<MoC:k;ITD9j;HoC?o;;01O0ISOXDm0h;ROYDn0g;ROYDn0g;ROYDn0g;SOXDm0g;TOXDm0g;TOXDm0f;UO[Dj0e;VO[Dj0e;VO\\Di0d;SOYDO5m0b;SObDn0];ROcDn0];QOdDo0\\;QOdDo0i;O001N11O0O1O10000000000000O101OO0100N2O1O1O100O10000O1000000O101O000O10000O100O100O2N1O1O1N3N2M3L_j`2"
    },
    "score": 0.05898278206586838
   },
   {
    "bbox": {
     "x_min": 0.0,
     "y_min": 357.5879211425781,
     "width": 89.16617584228516,
     "height": 60.26873779296875
    },
    "class": {
     "label": 61,
     "category_id": "unknown",
     "name": "toilet"
    },
    "mask": {
     "size": [
      419,
      640
     ],
     "counts": "Y;h1[;000000000001O0000001O000000001O0000001O00001O0000001O0000001O00000000001O00001O0000000000001O0000001O000000001O001O0000000000001O0000001O0000001O0000001O00001O00001O1O2N=C5K00TRQ7"
    },
    "score": 0.05034303665161133
   }
  ]
 },
 "/data/coco-2017/validation/data/000000463647.jpg": {
  "height": 480,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 292.1845703125,
     "y_min": 56.50839614868164,
     "width": 170.41796875,
     "height": 95.09635925292969
    },
    "class": {
     "label": 7,
     "category_id": "unknown",
     "name": "truck"
    },
    "mask": {
     "size": [
      480,
      640
     ],
     "counts": "knX4a0_>000O2M6K6J5J4N1N2O1N2O1N2O0O2O00001O0O101OO1000O10O1000O0100O001O10O01O00000001O1O1O1O1O002N1O101N1O100O2O000O101O00001O000000001O0O100000001O00000000001O0000001O00000001O000000000000000000000000000001O001N3N1O1O1O1O2N2N1M3O1O001O1O001O00001O0020N0000000000001M201N101N101O1N2M3M5L3L4N1N2O0O2O000O101O0000000O101O000O101N2M3M3J6H8H8M2000\\[c2"
    },
    "score": 0.86879962682724
   },
   {
    "bbox": {
     "x_min": 461.06640625,
     "y_min": 59.38587951660156,
     "width": 176.8099365234375,
     "height": 152.72943115234375
    },
    "class": {
     "label": 2,
     "category_id": "unknown",
     "name": "car"
    },
    "mask": {
     "size": [
      480,
      640
     ],
     "counts": "VVh6e1[=00001N2N3N3L3M3N001O0O2O001O00001O00001O00000O0100O10000O1O10O01O1O1O00001O1O001000O010O100O1O1O1O1O2N100O1O2N1O1O2N1O1O2N100O2N1O1O100O101N1000001N1000001O0O101O000O10001O000O100000001O000000000000001O00000000000000001O0000001O0000000000001O0O100000001O000000000000000000001O000000001O001O001O1O1O2N3L8I4L2N2N2N3M1O1O001O1O3M00000001O00000000000000P[1"
    },
    "score": 0.7612753510475159
   },
   {
    "bbox": {
     "x_min": 392.5167236328125,
     "y_min": 156.33848571777344,
     "width": 79.0218505859375,
     "height": 167.2237091064453
    },
    "class": {
     "label": 10,
     "category_id": "unknown",
     "name": "fire hydrant"
    },
    "mask": {
     "size": [
      480,
      640
     ],
     "counts": "SQh5<d>1M2N3K5G:C<I7D;G9I7A`0E;I6M3N2N2N2O2M2O1O101N100O1O1O2N1O100O2N10000O1N2O2N1N2K5F:000001O00000000ZKUFW4]:M101N1O1O001O00001O0O2O1N2N1O2\\O]EdLh:W3d0L4L4L5K5K5Kb1[NeZa2"
    },
    "score": 0.6850777268409729
   },
   ...
   {
    "bbox": {
     "x_min": 563.5179443359375,
     "y_min": 104.06021118164062,
     "width": 22.0667724609375,
     "height": 51.866912841796875
    },
    "class": {
     "label": 26,
     "category_id": "unknown",
     "name": "handbag"
    },
    "mask": {
     "size": [
      264,
      640
     ],
     "counts": "]aa4:j78H4L4K50N3NN210O0O11O1N2N1NH]OmHd0U762N3N1N3N]R>"
    },
    "score": 0.05407293513417244
   },
   {
    "bbox": {
     "x_min": 307.1141662597656,
     "y_min": 110.2352523803711,
     "width": 7.2305908203125,
     "height": 22.190330505371094
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "mask": {
     "size": [
      264,
      640
     ],
     "counts": "kh_2?i70O2M12A]nc2"
    },
    "score": 0.05292808264493942
   }
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
| Pretrained          | noised checkpoint (bbox_mAP=42.0%, segm_mAP=24.3%)                                                               |
| Training Parameters | epoch = 13, batch_size = 7, gradient_accumulation_step=4 (per device)                                            |
| Optimizer           | SGD (momentum)                                                                                                   |
| Loss Function       | Sigmoid Cross Entropy, SoftMax Cross Entropy, SmoothL1Loss                                                       |
| Speed               | 4pcs: 1348 ms/step                                                                                               |
| Total time          | 4pcs: 22h 56m 4s                                                                                                 |
| outputs             | mAP(bbox), mAP(segm)                                                                                             |
| mAP(bbox)           | 42.3                                                                                                             |
| mAP(segm)           | 32.3                                                                                                             |
| Model for inference | 778.5M(.ckpt file)                                                                                               |
| configuration       | mask_rcnn_49_640_dist_noised_experiment.yaml                                                                     |
| Scripts             |                                                                                                                  |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use
random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
