import argparse
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--data_dir', default='../data/coco/',
                        help='result root dir path')
    parser.add_argument('--anntations_dir', default='annotations/instances_val2014.json',
                        help='dataset annotations path')
    parser.add_argument('--result_name', default='obj_det_result_val.json',
                        help='object detection result json file name')
    # parser.add_argument('--train_dir', default='annotations/instances_train2014.json',
    #                     help='train set annotations path')
    # parser.add_argument('--det_res_dir', default='detector.json',
    #                     help='The result of Object Detection path')
    # parser.add_argument('--detection_caption_dir', default='dataset_coco.json',
    #                     help='The split of coco dataset path from Li FeiFei for original Detection.')
    # parser.add_argument('--save_train_name', default='')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    # trainCOCO = COCO(os.path.join(params['data_dir'], params['train_dir']))
    COCO = COCO(os.path.join(params['data_dir'], params['anntations_dir']))

    # trainCOCO_Dt = trainCOCO.loadRes(os.path.join(params['data_dir'], 'obj_det_result_train.json'))
    COCO_Dt = COCO.loadRes(os.path.join(params['data_dir'], params['result_name']))

    # # train set
    # cocoEval = COCOeval(trainCOCO, trainCOCO_Dt, 'bbox')
    # # imgIds = sorted(trainCOCO.getImgIds())
    # # imgIds = imgIds[0:100]
    # # cocoEval.params.imgIds = imgIds
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

    # validation set
    cocoEval = COCOeval(COCO, COCO_Dt, 'bbox')
    imgIds = sorted(COCO.getImgIds())
    imgIds = imgIds[2:3]
    print(imgIds)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()