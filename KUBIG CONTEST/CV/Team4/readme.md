# 멀티모달 AI 기반 옷 리뷰 작성 자동화

- 사용자가 옷을 착용한 사진을 업로드하면, 옷에 대한 리뷰를 자동으로 생성합니다.
- 만약, 사용자가 옷을 착용한 사진 외의 다른 사진을 업로드하더라도, **가상 피팅 모델**을 활용하여 리뷰를 작성할 옷을 입은 사진 생성합니다.
- 사용자가 입력할 값은 다음과 같습니다.

(1) 옷 착용 사진 or 본인 사진(가상 피팅 모델 사용)

(2) 리뷰를 작성할 옷 카테고리 선택 (ex. 상의-니트웨어, 아우터-코드, 하의-청바지)

(3) 원하는 리뷰 스타일 선택 (ex. 리뷰 말투, 핏, 만족도)

## 1. Architecture

![Image](https://github.com/user-attachments/assets/ea0644a5-5cb1-4676-955d-39e09765aded)

- **ACGPN (Adaptive Content Generating and Preserving Network)**
    - 가상 피팅 모델
    - 사용자가 리뷰를 작성할 옷을 착용한 사진을 올리지 않았을 경우, 사용자의 다른 사진과 리뷰 작성 옷을 합성한 가상 피팅 이미지 생성 -> 생성한 이미지를 바탕으로 리뷰 작성

- **segmentation**

  - YOLO-segmentation 사용
  - **train dataset = 12,000 장", "epoch = 30** 으로 학습한 모델 사용
  - 사용자가 입력한 옷 사진에 대해 segmentation을 수행 -> 옷 영역 탐지
  - 여러 개의 옷이 탐지되었을 경우, 사용자가 선택한 **리뷰를 작성할 옷 카테고리**에 속하는 클래스에 해당하는 것만 반환

- **cropping**
    - segmentation으로 예측된 폴리곤 좌표를 토대로 **옷 영역**만 cropping
    - 옷 특징을 자연어로 변환하기 위한 LLaVA 모델에서 더 정교한 image captioning 수행을 위해 옷 영역을 제외한 나머지 부분은 투명하게 RGBA 이미지로 처리
 
- **LLaVA**
    - 리뷰를 작성할 옷에 대한 특징을 GPT-4 모델로 넘기기 위해 자연어로 변환
    - **핏, 색감, 디자인(패턴, 스타일 등), 재질, 계절감**에 대한 특징 추출 (프롬프트 이용)
    - **input** : copped image, 프롬프트

- **GPT-4**
    - LLaVA를 통해 자연어로 변환된 옷에 대한 특징을 바탕으로 1차 리뷰 생성 -> 사용자 피드백 -> 최종 리뷰 생성
    - **input** : 사용자의 선택값(리뷰 말투, 핏, 만족도), image captioning 결과 (프롬프트 이용)



## 2. Dataset

![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/186.jpg)

![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202025-02-23%20170609.png)

- AI-Hub의 **K-패션 이미지**
- 23가지 스타일 (ex. 페미닌, 스트리트, 모던 등)으로 분류되어 있음
- 총 120만 장의 이미지 -> 23개의 스타일 중에서 6개의 스타일을 선정한 후, 각각 2000장씩 추출해서 총 12,000장을 training data로 사용
- 기존의 라벨링 데이터는 json 파일이었음. -> YOLO segmentation을 위해 필요한 정보 (ex. 클래스(아우터-코드), 폴리곤 좌표)만 추출해서 txt 파일로 변환
- YOLO segmentation으로 학습하기 위한 yaml 파일 생성

## 3. Result

### (1) test image
  ![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/test_image_%EC%9B%90%EB%B3%B8.png)

### (2) segmentation
  ![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/test_image_seg.png)

### (3) cropping
  ![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/test_image_cropped.png
)

### (4) LLaVA
  > ['The clothing in the image is described as having a cropped fit and a short length. This means that the clothing is not oversized or slim, but rather a more fitted and shorter version of the original design. The woman in the image is wearing a cropped shirt, which is a shorter version of a regular sh',
 'The color of the clothing is pink.',
 'The clothing in the image features a plaid pattern, which gives it a casual and stylish appearance. The woman is wearing a plaid shirt, and the design complements her overall look. The plaid pattern adds a touch of sophistication and character to the outfit, making it a fashionable choice for various',
 'The clothing in the image is made of plaid fabric.',
 'The clothing in the image, which includes a plaid shirt and a checkered shirt, is suitable for the fall and winter seasons. These patterns are often associated with colder weather and can provide warmth and comfort during the colder months.']

### (5) GPT-4
- 1차 리뷰 생성
  - 프롬프트
    > user_data = {
    "tone": "애교 있음",
    "fit": "잘 맞아요",
    "satisfaction": 90
} 
    >> Generated Review:
 이 분홍 체크무늬 크롭 셔츠, 정말 매력적인 스타일이네요! 👗🌸 봄이랑 여름에 입기 딱 좋아요. 가볍고 간편해서 덥지 않은 날씨에 입기에 최적입니당. 패턴도 너무 귀엽고, 탄탄한 느낌의 소재라서 자주 입어도 문제없을 것 같아요! 🤗그리고 이 크롭 길이 덕분에 핏도 너무 예쁘고 스타일리쉬하게 입을 수 있어서, 어떤 하의랑 매치해도 딱딱 맞는 느낌이에요. 블랙 팬츠랑 같이 코디했는데, 시크하면서도 포인트가 되는 것 같아 만족도 90%~~ 완전 대만족입니당! 다음에는 다른 색상도 한 번 도전해보고 싶어졌어요.😆💖

- 피드백
  - 블랙 팬츠랑 같이 코디했다는 내용을 제외해줘.

- 최종 리뷰 생성
   > Revised Review:
 이 분홍 체크무늬 크롭 셔츠, 정말 매력적인 스타일이네요! 👗🌸 봄이랑 여름에 입기 딱 좋아요. 가볍고 간편해서 덥지 않은 날씨에 입기에 최적입니당. 패턴도 너무 귀엽고, 탄탄한 느낌의 소재라서 자주 입어도 문제없을 것 같아요! 🤗그리고 이 크롭 길이 덕분에 핏도 너무 예쁘고 스타일리쉬하게 입을 수 있어서, 어떤 하의랑 매치해도 딱딱 맞는 느낌이에요. 완전 대만족입니당! 다음에는 다른 색상도 한 번 도전해보고 싶어졌어요.😆💖

### (6) ACGPN
#### Before
![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/before.png)

#### After
![image](https://github.com/KU-BIG/KUBIG_2025_SPRING/blob/main/KUBIG%20CONTEST/CV/Team4/images/after.png)
