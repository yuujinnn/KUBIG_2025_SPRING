멀티모달을 이용한 알츠하이머 예측
=========================
# 1. Motivation
* MRI 영상을 이용한, 단일 modality 기반의 진단 모델은 제한된 정보만을 사용하기 때문에, 알츠하이머와 같은 복합적인 질환의 진단 및 예측에 한계를 보일 것이라 예상
* 텍스트 형태의 환자 데이터를 추가하여 풍부한 임상 정보를 통합적으로 활용하고, 분류 정확도를 높이는 것을 목표로 함
* 비슷한 MRI 사진이어도, text 형태의 환자 데이터에 따라 알츠하이머 예측을 다르게 할 것이라 가정하고 실험을 진행
# 2. Dataset
## OASIS-1 Dataset
1. 데이터 규모 및 피험자 연령대
* 총 416명의, 18세부터 96세까지의 피험자를 대상으로 한 T1 weighted MRI 영상을 포함
* 모두 오른손잡이이며, 남성과 여성을 모두 포함하고 있음
* 정상 인지 기능부터, 경도(mild)에서 중등도(moderate)의 알츠하이머 질환을 가진 피험자를 포함
2. 개발 목적
* 치매 조기 진단 알고리즘 개발
* 분류(classification), 혹은 회귀(regression) 모델의 학습용
* segmentation 모델의 학습에도 많이 사용됨
3. Data
* 1.5T Siemens Vision MRI 스캐너에서 T1 weighted MPRAGE 시퀀스를 이용해 촬영
![022421371941352](https://github.com/user-attachments/assets/2333ca60-cd07-4774-a0f3-420364dc1f7e)

* 촬영된 사람의 성별, 나이, 치매 정도 등 다양한 정보를 알 수 있는 보고서 포함
<img width="868" alt="Screenshot 2025-02-24 at 9 46 21 PM" src="https://github.com/user-attachments/assets/be01087a-16d8-46ed-93db-3391da363109" />

https://sites.wustl.edu/oasisbrains/

    Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults. Marcus, DS, Wang, TH, Parker, J, Csernansky, JG, Morris, JC, Buckner, RL. Journal of Cognitive Neuroscience, 19, 1498-1507. doi: 10.1162/jocn.2007.19.9.1498
# 3. Method
## 1. Text data
### Text 데이터 정제
#### 기존 text 데이터
* MRI 영상을 촬영한 사람의 성별, 나이, 치매 등 다양한 임상 정보를 포함
#### 정제된 text 데이터
* 촬영한 사람의 성별, 나이를 text prompt로 활용할 수 있게 정제
<img width="871" alt="Screenshot 2025-02-24 at 10 00 22 PM" src="https://github.com/user-attachments/assets/7f9bb489-e780-407b-9728-800079fe4153" />

### Text 데이터 embedding
* CLIP pretrained text encoder 사용

## 2. MRI data
### Image 데이터 정제
#### 기존 image 데이터
* 헤더 파일 + img 파일
#### 정제된 image 데이터
* (256, 256, 128) 형태의 Nifti 파일
* nibabel python library를 이용하여 확장자 변환
### Image 데이터 embedding
![logo](https://github.com/user-attachments/assets/f1a1d028-c234-43a2-b49d-629185019134)

* ResNet encoder를 기반으로, 여러 모달리티의 메디컬 데이터셋으로 pretrained된 MedicalNet 사용

https://github.com/Tencent/MedicalNet

        @article{chen2019med3d,
            title={Med3D: Transfer Learning for 3D Medical Image Analysis},
            author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
            journal={arXiv preprint arXiv:1904.00625},
            year={2019}
        }

## 3. Multimodal embedding
* Concatenate
* Contrastive learning

## 4. Framework
<img width="888" alt="Screenshot 2025-02-24 at 10 12 58 PM" src="https://github.com/user-attachments/assets/00e061cf-9357-4499-b963-6957c8b89d44" />

# 4. Results
## Train/Val/Test
* 0.8 / 0.1 / 0.1
## Task
* 단순 classification
* NonDemented / VeryMildDementia / MildDementia / ModerateDementia
## Results
<img width="200" alt="Screenshot 2025-02-24 at 10 26 52 PM" src="https://github.com/user-attachments/assets/e9f67ff6-d443-4ac2-a4c4-76e71b6a57d7" />

* MRI image data만 가지고 학습헸을 때보다, text data를 함께 사용했을 때, 더 좋은 결과를 보임

## Limitation
* 3d MRI image에 대한 다양한 visualization 결과를 보여주지 못함
* 성별, 나이 뿐만 아니라 여러 임상적 데이터에 대한 다양한 correlation이 존재할 것
* 가장 간단한 형태의 4 class classification task이기 때문에 좋은 성능을 보임

## Future work
* 3d MRI image에 GradCAM과 같은 시각화 방법론을 적용시킬 수 있는 방안 모색
* 여러 임상 정보 및 음성, 영상과 같은 더 다양한 modality의 correlation에 대한 연구 가능
* classification에서 발전하여 알츠하이머가 심하게 진행되는 부위 등에 대한 segmentation task도 수행 가능








