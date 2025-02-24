# [Dacon] 저해상도 조류 이미지 분류 AI 경진대회

## Members
18기 강동헌, 20기 강민정, 21기 김연주, 송상현

## Introduction
저해상도의 조류 이미지로부터 종을 정확히 분류할 수 있는 AI 알고리즘 개발  

## Datasets
- train, upscale train (256 x 256) 15,834장
- test (64 x 64) 6,786장

## Tasks
### 1. Preprocessing
   - Data Augmentation (Resize, Horizontal Flip, Colorjitter, Cutmix)  
     ![Image](https://github.com/user-attachments/assets/d5edc9b0-8c19-448e-ab0a-4581737efd48)
     
   - Detection (Yolov8 fine-tuning) : 고해상도 사진에서 새를 탐지해 bounding box, contour line의 좌표를 각각 학습시킨 뒤, 저해상도 사진과 해당 좌표 label를 yolo에 fine-tuning
     
     ![Image](https://github.com/user-attachments/assets/9ac33206-4941-4fdc-9276-b0367c7b8b27)
     
   - Super Resolution (Real-ESRGAN) : test data upscaling
     ![Image](https://github.com/user-attachments/assets/a8ccf951-f01a-4758-9c81-2536e547f201)
  
### 2. Classification
   - ViT (Vision Transformer) 기반 모델에 저해상도/고해상도 train data를 학습시킨 뒤, test data에 대한 분류 진행
   - BEiT v2
   - SwinV2

## Results


