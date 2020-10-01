import argparse
import json
from copy import deepcopy

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import os

from misc.dataloader_hdf import HDFSingleDataset
from misc.dataloader_json import JSONSingleDataset

def convert_to_map(imgs):
    image_ids_map = {}
    for i in range(len(imgs)):
        info = imgs[i]
        image_ids_map[info['id']] = info
    return image_ids_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--data_dir', default='../data/coco/',
                        help='result dir path')
    parser.add_argument('--val_dir', default='annotations/instances_val2014.json',
                        help='validation set annotations path')
    parser.add_argument('--train_dir', default='annotations/instances_train2014.json',
                        help='train set annotations path')
    parser.add_argument('--det_res_dir', default='detector.json',
                        help='The result of Object Detection path')
    parser.add_argument('--detection_caption_dir', default='dataset_coco.json',
                        help='The split of coco dataset path from Li FeiFei for original Detection.')
    # parser.add_argument('--save_train_name', default='')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    # print('parsed input parameters:')
    # print(json.dumps(params, indent=2))

    valAnnDir = os.path.join(params['data_dir'], params['val_dir'])
    trainAnnDir = os.path.join(params['data_dir'], params['train_dir'])

    trainCOCO = COCO(trainAnnDir)
    # train_img_id_map = convert_to_map(trainCOCO.dataset['images'])
    print('Original COCO train dataset train length: {}'.format(len(trainCOCO.getImgIds())))
    valCOCO = COCO(valAnnDir)
    # val_img_id_map = convert_to_map(valCOCO.dataset['images'])
    print('Original COCO val dataset train length: {}'.format(len(valCOCO.getImgIds())))
    train_result = []
    val_result = []
    proposals_file = os.path.join(params['data_dir'], params['det_res_dir'])
    # for debug
    count_train = 0
    count_val = 0
    count = 0
    last_img_id = 0
    if proposals_file.endswith('json'):
        dataloader = JSONSingleDataset(proposals_file)
        print('Dataset length: {}'.format(dataloader.__len__()))
        proposals_json = dataloader.getAll()
        print('Proposals length: {}'.format(len(proposals_json)))
        print('Start to format the result.')
        for key in proposals_json.keys():
            # for debug
            value = proposals_json[key]
            dets_labels = np.array(value['dets_labels'])
            if len(dets_labels) == 0:
                ann_one = {
                    'image_id': int(key),
                    'category_id': -1,
                    'bbox': [0,0,0,0],
                    'score': -1
                }
                # print("key = {}, no detetions.".format(key))
                if int(key) in trainCOCO.imgs:
                    train_result.append(ann_one)
                    if last_img_id != int(key):
                        count_train += 1
                        last_img_id = int(key)
                else:
                    val_result.append(ann_one)
                    if last_img_id != int(key):
                        count_val += 1
                        last_img_id = int(key)
            for i in range(len(dets_labels)):
                detection = dets_labels[i]
                bbox = [detection[0], detection[1], detection[2] - detection[0], detection[3] - detection[1]]
                score = detection[5]
                label = int(detection[4])
                ann_one = {
                    'image_id': int(key),
                    'category_id': label,
                    'bbox': bbox,
                    'score': score
                }
                # check the image exsited in validation or train set.
                if int(key) in trainCOCO.imgs:
                    train_result.append(ann_one)
                    if last_img_id != int(key):
                        count_train += 1
                        last_img_id = int(key)
                else:
                    val_result.append(ann_one)
                    if last_img_id != int(key):
                        count_val += 1
                        last_img_id = int(key)
            if count % 10000 == 0:
                print("Processing... Count = {}".format(count))
            count += 1
    else:
        print('original detection format start!')
        dataloader = HDFSingleDataset(proposals_file)
        dataset_len = dataloader.__len__()
        print('Dataset length: {}'.format(dataset_len))
        imgs = json.load(open(os.path.join(params['data_dir'], params['detection_caption_dir']), 'r'))
        imgs = imgs['images']
        for i in range(dataset_len):
            proposal_item = deepcopy(dataloader[i])
            dets_labels = np.array(proposal_item['dets_labels'][0])
            if len(dets_labels) == 0:
                ann_one = {
                    'image_id': imgs[i]['cocoid'],
                    'category_id': -1,
                    'bbox': [0,0,0,0],
                    'score': -1
                }
                # print("key = {}, no detetions.".format(key))
                if int(imgs[i]['cocoid']) in trainCOCO.imgs:
                    train_result.append(ann_one)
                    if last_img_id != int(imgs[i]['cocoid']):
                        count_train += 1
                        last_img_id = int(imgs[i]['cocoid'])
                else:
                    val_result.append(ann_one)
                    if last_img_id != int(imgs[i]['cocoid']):
                        count_val += 1
                        last_img_id = int(imgs[i]['cocoid'])
            for j in range(len(dets_labels)):
                detection = dets_labels[j]
                bbox = [detection[0], detection[1], detection[2] - detection[0], detection[3] - detection[1]]
                score = detection[5]
                label = int(detection[4]) # This label is not real label on COCO, so we should change it
                label = valCOCO.getCatIds()[label - 1]
                if score == 0:
                    continue
                ann_one = {
                    'image_id': imgs[i]['cocoid'],
                    'category_id': label,
                    'bbox': bbox,
                    'score': score
                }
                # check the image exsited in validation or train set.
                if int(imgs[i]['cocoid']) in trainCOCO.imgs:
                    train_result.append(ann_one)
                    if last_img_id != int(imgs[i]['cocoid']):
                        count_train += 1
                        last_img_id = int(imgs[i]['cocoid'])
                else:
                    val_result.append(ann_one)
                    if last_img_id != int(imgs[i]['cocoid']):
                        count_val += 1
                        last_img_id = int(imgs[i]['cocoid'])
            if count % 10000 == 0:
                print("Processing... Count = {}".format(count))
            count += 1

    print('Total count: {}'.format(count))
    print('count_train: {}'.format(count_train))
    print('count_val: {}'.format(count_val))
    print("Saving the formatted result.")

    with open(os.path.join(params['data_dir'], 'nbt_obj_det_result_train.json'), 'w') as f:
        json.dump(train_result, f)
    with open(os.path.join(params['data_dir'], 'nbt_obj_det_result_val.json'), 'w') as f:
        json.dump(val_result, f)
    print("End saving the formatted result.")

