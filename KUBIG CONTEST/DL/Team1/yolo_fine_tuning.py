
# YOLOv8 설치 (Ultralytics)
!pip install ultralytics
!pip install torch torchvision torchaudio

import torch
from ultralytics import YOLO

# GPU 확인
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("사용 가능한 GPU 개수:", torch.cuda.device_count())
print("현재 사용 중인 GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

pip install ultralytics opencv-python matplotlib tqdm

from google.colab import drive
drive.mount('/content/drive')

'''upscaled data에서 조류 탐지 후 json 파일로 박스, 외곽선 위치 저장'''

# 바운딩 박스 한개만 허용
import cv2
import os
import glob
import shutil
import numpy as np
import json
import torch
from tqdm import tqdm
from ultralytics import YOLO

# YOLO v8 모델 로드
model = YOLO("yolov8x-seg.pt")

# 입력 및 출력 폴더 설정
input_folder = "/content/drive/MyDrive/yolo_dataset/images"
output_folder = "/content/drive/MyDrive/yolo_dataset/bounded_images"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# COCO 형식의 어노테이션 저장 폴더
annotations_file = os.path.join(output_folder, "coco_annotations.json")
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": []
}

# 탐지 실패 로그 파일 생성
log_file = os.path.join(output_folder, "failed_images.txt")
failed_log = open(log_file, "w")

# 이미지 파일 목록 가져오기
image_paths = glob.glob(os.path.join(input_folder, "*.png"))
image_paths += glob.glob(os.path.join(input_folder, "*.jpg"))
image_paths += glob.glob(os.path.join(input_folder, "*.jpeg"))

print(f"총 {len(image_paths)}개의 이미지를 처리합니다.")

# 이미지 처리 루프
annotation_id = 1  # 어노테이션 ID
category_id = 1  # 단일 클래스(새) 가정

for img_id, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    results = model(img)

    # 탐지된 객체가 없으면 원본 저장
    if len(results[0].boxes) == 0:
        shutil.copy(img_path, os.path.join(output_folder, os.path.basename(img_path)))
        failed_log.write(f"{img_path}\n")
        continue

    output_img = img.copy()
    img_height, img_width = img.shape[:2]

    # COCO 스타일 이미지 정보 저장
    coco_annotations["images"].append({
        "id": img_id,
        "file_name": os.path.basename(img_path),
        "width": img_width,
        "height": img_height
    })

    # 바운딩 박스 및 세그멘테이션 정보 추출
    for box, mask in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].masks.data.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        bbox_width, bbox_height = x2 - x1, y2 - y1

        # 바운딩 박스 그리기 (초록색)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 마스크를 원본 크기로 변환 후 적용
        mask_resized = cv2.resize(mask, (img_width, img_height))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold 적용

        # 컨투어(객체 외곽선) 찾기
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # COCO Segmentation 저장 형식 (폴리곤 좌표)
        segmentation = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # 너무 작은 컨투어는 제외
                flattened = cnt.flatten().tolist()
                segmentation.append(flattened)

        # COCO 어노테이션 저장
        coco_annotations["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x1, y1, bbox_width, bbox_height],
            "segmentation": segmentation,
            "iscrowd": 0,
            "area": bbox_width * bbox_height
        })
        annotation_id += 1

        # 컨투어를 파란색으로 그리기 (세그멘테이션 시각화)
        cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2)  # 파란색 컨투어

    # 최종 결과 저장
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, output_img)

# COCO 어노테이션 JSON 저장
with open(annotations_file, "w") as f:
    json.dump(coco_annotations, f, indent=4)

# 탐지 실패한 로그 파일 닫기
failed_log.close()

print(f"✅ 모든 이미지 처리 완료! 결과는 {output_folder}에 저장되었습니다.")
print(f"📝 COCO 스타일 어노테이션이 {annotations_file}에 저장되었습니다.")


# 바운딩 박스 한개만 허용
import cv2
import os
import glob
import shutil
import numpy as np
import json
import torch
from tqdm import tqdm
from ultralytics import YOLO

# YOLO v8 모델 로드
model = YOLO("yolov8x-seg.pt")

# 입력 및 출력 폴더 설정
input_folder = "/content/drive/MyDrive/yolo_dataset/images"
output_folder = "/content/drive/MyDrive/yolo_dataset/bounded_images"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# COCO 형식의 어노테이션 저장 폴더
annotations_file = os.path.join(output_folder, "coco_annotations.json")
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "bird"}]  # '새' 카테고리 추가
}

# 탐지 실패 로그 파일 생성
log_file = os.path.join(output_folder, "failed_images.txt")
failed_log = open(log_file, "w")

# 이미지 파일 목록 가져오기
image_paths = glob.glob(os.path.join(input_folder, "*.png"))
image_paths += glob.glob(os.path.join(input_folder, "*.jpg"))
image_paths += glob.glob(os.path.join(input_folder, "*.jpeg"))

print(f"총 {len(image_paths)}개의 이미지를 처리합니다.")

# 이미지 처리 루프
annotation_id = 1  # 어노테이션 ID
category_id = 1  # 단일 클래스(새) 가정

for img_id, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    results = model(img)

    # 탐지된 객체가 없으면 원본 저장
    if len(results[0].boxes) == 0:
        shutil.copy(img_path, os.path.join(output_folder, os.path.basename(img_path)))
        failed_log.write(f"{img_path}\n")
        continue

    output_img = img.copy()
    img_height, img_width = img.shape[:2]

    # COCO 스타일 이미지 정보 저장
    coco_annotations["images"].append({
        "id": img_id,
        "file_name": os.path.basename(img_path),
        "width": img_width,
        "height": img_height
    })

    # 바운딩 박스 및 신뢰도 정렬 (신뢰도가 가장 높은 한 개만 선택)
    detections = list(zip(results[0].boxes.xyxy.cpu().numpy(), results[0].masks.data.cpu().numpy(), results[0].boxes.conf.cpu().numpy()))
    detections.sort(key=lambda x: x[2], reverse=True)  # 신뢰도(conf) 기준 정렬

    if detections:  # 감지된 객체가 있는 경우
        box, mask, conf = detections[0]  # 가장 신뢰도 높은 한 개만 선택
        x1, y1, x2, y2 = map(int, box)
        bbox_width, bbox_height = x2 - x1, y2 - y1

        # 바운딩 박스 그리기 (초록색)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 마스크를 원본 크기로 변환 후 적용
        mask_resized = cv2.resize(mask, (img_width, img_height))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold 적용

        # 컨투어(객체 외곽선) 찾기
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # COCO Segmentation 저장 형식 (폴리곤 좌표)
        segmentation = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # 너무 작은 컨투어는 제외
                flattened = cnt.flatten().tolist()
                segmentation.append(flattened)

        # COCO 어노테이션 저장 (가장 신뢰도 높은 객체만 추가)
        coco_annotations["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x1, y1, bbox_width, bbox_height],
            "segmentation": segmentation,
            "iscrowd": 0,
            "area": bbox_width * bbox_height
        })
        annotation_id += 1

        # 컨투어를 파란색으로 그리기 (세그멘테이션 시각화)
        cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2)  # 파란색 컨투어

    # 최종 결과 저장
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, output_img)

# COCO 어노테이션 JSON 저장
with open(annotations_file, "w") as f:
    json.dump(coco_annotations, f, indent=4)

# 탐지 실패한 로그 파일 닫기
failed_log.close()

print(f"✅ 모든 이미지 처리 완료! 결과는 {output_folder}에 저장되었습니다.")
print(f"📝 COCO 스타일 어노테이션이 {annotations_file}에 저장되었습니다.")

'''label 생성'''

import json
import os

# 파일 경로 설정
annotations_path = "/content/drive/MyDrive/yolo_dataset/coco_annotations.json"
labels_folder = "/content/drive/MyDrive/yolo_dataset/labels_seg"

# 저장 폴더 생성
os.makedirs(labels_folder, exist_ok=True)

# COCO 어노테이션 로드
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# 이미지 정보 딕셔너리 생성
image_info = {img["id"]: img for img in coco_data["images"]}

# YOLO 형식으로 변환
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]

    if image_id not in image_info:
        print(f"⚠ 경고: image_id {image_id}가 COCO 데이터에 없음, 스킵합니다.")
        continue

    file_name = image_info[image_id]["file_name"]
    img_width = image_info[image_id]["width"]
    img_height = image_info[image_id]["height"]

    # YOLO 형식 바운딩 박스 변환
    x1, y1, w, h = ann["bbox"]
    x_center = (x1 + w / 2) / img_width
    y_center = (y1 + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height

    # YOLO 세그멘테이션 변환 (컨투어 폴리곤 좌표 정규화)
    segmentation = []
    for segment in ann["segmentation"]:
        normalized_segment = [segment[i] / img_width if i % 2 == 0 else segment[i] / img_height for i in range(len(segment))]
        segmentation.extend(normalized_segment)

    # 최대 16개 좌표(x,y) 제한 (YOLO는 최대 16개 점 사용 가능)
    if len(segmentation) > 32:  # 16개 쌍(32개 값) 초과 시 균등한 간격으로 16개 샘플링
      indices = np.linspace(0, len(segmentation) // 2 - 1, 16, dtype=int)  # 16개 인덱스 균등 샘플링
      segmentation = [segmentation[i * 2] for i in indices] + [segmentation[i * 2 + 1] for i in indices]  # x, y 좌표 분리 후 재조합

    # YOLO 라벨 파일명
    label_file = os.path.join(labels_folder, os.path.splitext(file_name)[0] + ".txt")

    # YOLO Segmentation 라벨 파일 저장
    with open(label_file, "w") as f:
        category_id = ann["category_id"] - 1  # YOLO는 클래스 ID를 0부터 시작
        f.write(f"{category_id} " + " ".join(map(str, segmentation)) + "\n")

print(f"✅ YOLO 세그멘테이션 포맷 변환 완료! 라벨 파일이 {labels_folder}에 저장되었습니다.")

'''모델 학습'''

import os
import yaml

# 📌 데이터 경로 설정
dataset_path = "/content/drive/MyDrive/yolo_dataset/model_building"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels_seg")  # 세그멘테이션 라벨 사용

# 📌 data.yaml 파일 생성 (train과 val을 동일하게 설정)
data_yaml = {
    "train": images_path,  # 모든 데이터를 train으로 사용
    "val": images_path,    # 동일한 데이터를 val로 사용
    "nc": 1,               # 클래스 개수 (새 1종)
    "names": ["bird"],     # 클래스 이름
    "task": "segment"      # 세그멘테이션 모델 학습
}

yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"✅ data.yaml 파일 생성 완료: {yaml_path}")

!yolo task=segment mode=train model=yolov8m-seg.pt data="/content/drive/MyDrive/yolo_dataset/data.yaml" epochs=50 imgsz=512 device=0 workers=16 batch=64