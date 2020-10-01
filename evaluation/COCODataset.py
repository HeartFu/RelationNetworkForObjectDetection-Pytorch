import os

from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data


class COCODataset(data.Dataset):
    def __init__(self, images_path, ann_path, split='train', transform=None):
        self.coco = COCO(ann_path)
        self.image_path = images_path
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.split = split

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # print(ann_ids)
        target = self.coco.loadAnns(ann_ids)
        # print(target)
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.image_path, file_name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_id

    def __len__(self):
        return len(self.ids)
