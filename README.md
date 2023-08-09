# A Pytorch implementation of PP-PicoDet 
Original paper:<br>
[PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices](https://arxiv.org/abs/2111.00902)<br>

### Introduction
This vanilla Pytorch object-dection implementation of PP-PicoDet is simplified and modified based on the [Official PaddleDetection Code](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/picodet/README_en.md). The network is pretrained on COCO2017 and the weights can be download from [Google Drive](https://drive.google.com/u/0/uc?id=15aeB6sEVKzyB20tISF38qegZ1bdJm9ZL&export=download). Inference, training and validation demos are provided. If you have any questions about the code, please feel free to ask in issues.

## Requirements
- Python 3.8
- cudatoolkit 11.3
- torchvision 0.13.1
- pytorch 1.12.1
- opencv-python 4.8.0.74
- pycocotools 2.0.6
- albumentations 1.3.1
- tqdm 4.65.1

## Dataset
Please download COCO2017 under the *./data* directory with the following structure:
```
data/
|-- coco/
|---- train2017/
|---- val2017/
|---- annotations/
...
```

## Inference
Please download the pretrained weights ([Google Drive](https://drive.google.com/u/0/uc?id=15aeB6sEVKzyB20tISF38qegZ1bdJm9ZL&export=download)) and put it under the *./checkpoints* directory.
The demo images (picked from COCO2017) have already been put into *./inference/*, Simply run infer.py:
```shell
python infer.py --images inference/0.jpg inference/1.jpg 
```
Result images will be generated in the same *./inference/* folder.

## Training
The training configuration of hyper parameters is in *./configs/coco2017.yml*, you can modify it according to the machines.
Run this line below: 
```shell
python train.py --device cuda --config configs/coco2017.yml --total_epoch 100
```
Please note that to use the pretrained weights all the input images are scale to the size of **288x512**. This is hard-coded in the train.py file and I am too lazy to make it flexible. The same for inference and validation. Please change this and rerun trainning according to your specific needs.

## Validation
For validation, please run:
```shell
python eval.py --device cuda --config configs/coco2017.yml
```
A map(0.5:0.95) of 26.1 is reached by the pretrained weight, still lower than the acclaimed 27.1 published in the paper. However, I did the whole trainning on my personal computer with a single RTX3080 and I can't bear the time-cost to saturate the training. Feel free to train it more sufficiently on your machines. 
