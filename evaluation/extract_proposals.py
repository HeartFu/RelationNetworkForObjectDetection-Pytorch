import json
import os

import h5py
import torch
import torchvision
import numpy as np
from torch.utils import data
from tqdm import tqdm

from .COCODataset import COCODataset
from object_detection.model.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision import transforms


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, img):
        return self.transform(img)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model
    con_thresh = 0.5
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    print(model)
    model.to(device)
    model.eval()
    # load the dataset
    transform = Transform()
    train_images_path = '/home/fanfu/newdisk/dataset/coco/2014/train2014'
    train_ann_path = '/home/fanfu/newdisk/dataset/coco/2014/annotations/instances_train2014.json'
    train_dataset = COCODataset(train_images_path, train_ann_path, split='val', transform=transform)
    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    images_path = '/home/fanfu/newdisk/dataset/coco/2014/val2014'
    ann_path = '/home/fanfu/newdisk/dataset/coco/2014/annotations/instances_val2014.json'
    val_dataset = COCODataset(images_path, ann_path, split='val', transform=transform)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    proposals = {}

    progress_bar_train = tqdm(train_dataloader, desc='|Train Extract Proposals', leave=False)
    for i, data in enumerate(progress_bar_train):
        img, target, img_id = data
        # print(target)
        # print(img_id)
        prediction = model(img.cuda())
        bbox = prediction[0]['boxes'].to('cpu')
        scores = prediction[0]['scores'].to('cpu')
        classes = prediction[0]['labels'].to('cpu').float()
        # selected_idx = np.where(scores > con_thresh)
        nms_num = 0
        for score in scores:
            if score <= con_thresh:
                break
            nms_num += 1

        dets_labels = torch.cat((bbox, torch.unsqueeze(classes, 1), torch.unsqueeze(scores, 1)), 1)
        # import pdb
        # pdb.set_trace()
        num_dets = bbox.size(0)
        proposal_obj = {
            'dets_num': [num_dets],
            'nms_num': [nms_num],
            'dets_labels': dets_labels.detach().numpy().tolist()
        }
        # import pdb
        # pdb.set_trace()
        proposals[img_id.item()] = proposal_obj
        # proposals.append(proposal_obj)

    progress_bar = tqdm(val_dataloader, desc='|Val Extract Proposals', leave=False)
    for i, data in enumerate(progress_bar):
        img, target, img_id = data
        # print(target)
        # print(img_id)
        prediction = model(img.cuda())
        bbox = prediction[0]['boxes'].to('cpu')
        scores = prediction[0]['scores'].to('cpu')
        classes = prediction[0]['labels'].to('cpu').float()
        # selected_idx = np.where(scores > con_thresh)
        nms_num = 0
        for score in scores:
            if score <= con_thresh:
                break
            nms_num += 1

        dets_labels = torch.cat((bbox, torch.unsqueeze(classes, 1), torch.unsqueeze(scores, 1)), 1)
        # import pdb
        # pdb.set_trace()
        num_dets = bbox.size(0)

        proposal_obj = {
            'dets_num': [num_dets],
            'nms_num': [nms_num],
            'dets_labels': dets_labels.detach().numpy().tolist()
        }
        # import pdb
        # pdb.set_trace()
        proposals[img_id.item()] = proposal_obj
        # proposals.append(proposal_obj)

    with open('detector.json', 'w') as f:
        json.dump(proposals, f)
