import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append('../picodet')
from picodet.coco import COCODataset
from torch.utils.data import Dataset,DataLoader

"""
def collate_fn(batch):
    n = len(batch)
    output = {}
    for k in batch[0].keys():
        if 'path' in k:
            continue
        tmp = []
        for i in range(n):
            tmp.append(batch[i][k])
        if (not 'gt_' in k) and (not 'is_crowd' in k):
            tmp = torch.stack(tmp, dim=0)
        output[k] = tmp
    return output
"""


def collate_fn(batch):
    return tuple(zip(*batch))


def create_train_dataset(train_root, train_anno_path, to_size,
                         device, train_tf=None):
    train_dataset = COCODataset(train_root,
                                train_anno_path,
                                to_size,
                                device,
                                train_tf)
    return train_dataset


def create_val_dataset(val_root,val_anno_path, to_size,
                       device, val_tf=None):
    val_dataset = COCODataset(val_root, val_anno_path, to_size, device,val_tf)
    return val_dataset


def create_train_loader(train_dataset,batch_size,num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle = True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_val_loader(val_dataset,batch_size=1, num_workers=0):
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return val_loader

def create_train_transform(to_size):
    height, width = to_size
    #img_trans = A.Compose([
    #    A.GridDropout(),
    #])
    det_trans = A.Compose([
        A.HueSaturationValue(hue_shift_limit=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomSizedBBoxSafeCrop(height,width,erosion_rate=0.4, p=0.5),
        A.Resize(height,width),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format= 'pascal_voc',
        label_fields=['classes']
    ))
    return det_trans


def create_val_transform(to_size):
    height, width = to_size
    return A.Compose([
        A.Resize(height,width),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format= 'pascal_voc',
        label_fields=['classes']
    ))



# visualize single image bounding boxes
def visualize_sample(record):
    image, targets = record
    #print(targets.keys())
    image = image.permute([1,2,0]).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box_num in range(len(targets['gt_class'])):
        bbox = targets['gt_bbox'][box_num]
        label = targets['gt_class'][box_num]
        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            image, str(label.item()), (int(bbox[0]), int(bbox[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
    cv2.imshow('image', image)
    cv2.waitKey(0)


def test_dataset():
    im_root = 'D:/data/shixinqiu/val'
    anno_path = 'D:/data/shixinqiu/annotations/val_bbox.json'
    transform = create_train_transform((288, 512))
    dataset = COCODataset(im_root, anno_path, (288, 512),device=torch.cpu,
                          transform=transform)
    print(f"Number of training images: {len(dataset)}")
    NUM_SAMPLES_TO_VISUALIZE = 100
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        record = dataset[i]
        visualize_sample(record)


def test_loader():
    train_root = 'D:/data/shixinqiu/train'
    train_anno_path = 'D:/data/shixinqiu/annotations/train_bbox.json'
    val_root = 'D:/data/shixinqiu/val'
    val_anno_path = 'D:/data/shixinqiu/annotations/val_bbox.json'
    to_size=(288,512)
    train_ds = create_train_dataset(train_root,train_anno_path,to_size)
    val_ds = create_val_dataset(val_root, val_anno_path, to_size)
    train_loader = create_train_loader(train_ds,batch_size=2)
    val_loader = create_val_loader(val_ds,batch_size=2)
    BATCH_TO_TEST = 2
    for i, train_batch in enumerate(train_loader):
        if i >= BATCH_TO_TEST:
            break
        print(f'---train batch {i+1}---')
        print(train_batch)
        print(f'-----------------------')
        print(train_batch['image'].shape)
    for i, val_batch in enumerate(val_loader):
        if i >= BATCH_TO_TEST:
            break
        print(f'---val batch {i+1}---')
        print(val_batch)
        print(f'-----------------------')
        print(val_batch['image'].shape)


if __name__ == '__main__':
    test_dataset()