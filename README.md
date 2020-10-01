# Relation Networks for Object Detection based on Pytorch and Torchvision

This is a Faster-RCNN with FPN and relation network based on Torchvision.

This folder contains the training scripts for object detection, Faster-RCNN with FPN model, Relation Network and evaluation scripts.

### Progress
- [x] add Relation Network between two fully connected layers in RoI head
- [x] end to end train this model on COCO 2014
- [x] evaluate mAP based on COCO evaluation tools
- [ ] replace NMS by Relation Network
- [ ] improve the performance (Current mAP: 36)


### Requirements

1. Python 3.6
2. Pytorch 1.4.1 and Torchvision
3. The Python packages:
```
cython
pycocotools
matplotlib
```

### Dataset

Please download the [COCO Dataset](https://cocodataset.org/#download).

### Before Training

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Data-path is the path of dataset.

nproc_per_node is the number of GPUs. 

### Training

Train script is below, the backbone is Resnet 101:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet101_fpn --epochs 30\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

This model contains {1, 1} relation network between the fully connected layers in RoI head.


### Reference

- [Torchvision](https://github.com/pytorch/vision)
- [Relation Networks for object Detection Code Review Blog](https://blog.csdn.net/u014380165/article/details/80779712)
- [Relation Networks for object Detection](https://arxiv.org/abs/1711.11575)
- [Relation Netwroks for Object Detection MXNet Code](https://github.com/msracver/Relation-Networks-for-Object-Detection)
