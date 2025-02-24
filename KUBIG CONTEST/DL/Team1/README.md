# [Dacon] 저해상도 조류 이미지 분류 AI 경진대회

## Members
18기 강동헌, 20기 강민정, 21기 김연주, 송상현

## Introduction
저해상도의 조류 이미지로부터 종을 정확히 분류할 수 있는 AI 알고리즘 개발  

## Datasets
- train, upscale train (256 x 256) 15,834
- test (64 x 64) 6,786

## Tasks
1. Preprocessing
   - Data Augmentation (Resize, Horizontal Flip, Colorjitter, Cutmix)
     <img src="https://github.com/user-attachments/assets/a97c1016-1a4c-49a3-94f1-6de26d7be54f.png  width="200" height="400"/>
   - Detection (Yolov8 fine-tuning)
   - Super Resolution (Real-ESRGAN)
  
2. Classification
   - ViT (Vision Transformer) 기반 모델에 train/upscale train 데이터를 학습시킨 뒤, test 데이터에 대한 분류 진행
   - BEiT v2
   - SwinV2

## Results


