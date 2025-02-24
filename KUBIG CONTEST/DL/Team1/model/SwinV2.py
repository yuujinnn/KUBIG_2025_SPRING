!pip install pytorch-lightning

import os
import random
import math
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torchvision.transforms import v2  # torchvision.transforms.v2 에서 CutMix 사용


import warnings
warnings.filterwarnings(action='ignore')

!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

'''Randomseed 고정'''

CFG = {
    'EPOCHS': 20,
    'IMG_SIZE': 224,
    'LEARNING_RATE': 5e-5,
    'BATCH_SIZE': 16,
    'SEED': 41
}

def set_seed(seed=CFG['SEED']):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
set_seed()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''Data load & Preprocessing''''

# img path 수정
folder_path = '/content/drive/MyDrive'
train = pd.read_csv(f'{folder_path}/train.csv')

train['img_path'] = train['img_path'].apply(lambda x: folder_path + x[1:])
train['upscale_img_path'] = train['upscale_img_path'].apply(lambda x: folder_path + x[1:])

# train-validation split
train_df, val_df = train_test_split(train, test_size=0.2, stratify=train['label'], random_state=CFG['SEED'])

# Label Encoding
le = preprocessing.LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
val_df['label'] = le.transform(val_df['label'])

# upscaled 데이터 추가하여 train_df 확장
train_expanded_df = pd.concat([
    train_df,  # 원본
    train_df.assign(img_path=train_df['upscale_img_path'])  # 업스케일링
], ignore_index=True)

print("train_expanded_df:", len(train_expanded_df))

'''CustomDataset'''

class CustomDataset(Dataset):
    def __init__(self, df, transforms, processor):
        self.df = df
        self.transforms = transforms
        self.processor = processor

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row['img_path']
        label = row['label']

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image'] #augmentation

        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (1, C, H, W) -> (C, H, W)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(int(label), dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)

'''Augmentation'''

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ToTensorV2()
])

'''Cutmix'''

num_classes = len(le.classes_)

# CutMix 객체 생성 (num_classes 적용)
cutmix = v2.CutMix(num_classes=num_classes)

# CutMix 사용 여부 설정 (True: 사용, False: 사용 안 함)
USE_CUTMIX = True

# cutmix 적용을 위한 collate_fn (학습용)
def train_collate_fn(batch):
    images = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    # 학습 시에만 CutMix 적용 (USE_CUTMIX가 True일 경우)
    if USE_CUTMIX and cutmix is not None:
        images, labels = cutmix(images, labels)
    return {"pixel_values": images, "labels": labels}

# 검증용 collate_fn (CutMix 미적용)
def val_collate_fn(batch):
    images = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"pixel_values": images, "labels": labels}

'''Model - SwinV2'''

model_name = "microsoft/swinv2-base-patch4-window16-256"
processor = AutoImageProcessor.from_pretrained(model_name, do_normalize=False)

train_dataset = CustomDataset(train_expanded_df, train_transform, processor)
val_dataset = CustomDataset(val_df, train_transform, processor)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, collate_fn=train_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, collate_fn=val_collate_fn)

class SwinClassifier(pl.LightningModule):
    def __init__(self, num_classes, model_name=model_name, learning_rate=CFG['LEARNING_RATE']):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.backbone = AutoModel.from_pretrained(model_name)
        self.latent_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.latent_dim, num_classes)

        self.validation_step_outputs = []

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self.forward(pixel_values)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self.forward(pixel_values)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # 배치 단위에서는 F1 점수를 개별로 계산하지 않고 저장
        self.validation_step_outputs.append({"preds": preds, "labels": labels})

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # 전체 validation 데이터셋에 대해 F1 Score 계산
        preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0).cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0).cpu().numpy()
        val_f1 = f1_score(labels, preds, average="macro")

        self.log("val_f1", val_f1, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.clear()  # 다음 epoch를 위해 초기화

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2
        )

        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=2,
            min_lr=1e-7,
            verbose=True,
        )

        return [optimizer], [
            {
                "scheduler": plateau_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "name": "plateau_scheduler"
            }
        ]

from pytorch_lightning.callbacks import ModelCheckpoint

# ModelCheckpoint 콜백 정의 (val_f1 지표가 최대일 때를 기준으로)
checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",           
    mode="max",
    save_top_k=1,               
    verbose=True,
    dirpath="/content/drive/MyDrive/results/swin_torch",
    filename="best-checkpoint"
)

# 조기종료
from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor="val_f1",      
    min_delta=0.001,        
    patience=7,             
    verbose=True,
    mode="max"
)

'''Train'''

model = SwinClassifier(num_classes=num_classes)

# Trainer 생성 (accelerator="auto"로 GPU 사용 가능 시 자동 선택)
trainer = pl.Trainer(
    max_epochs=CFG['EPOCHS'],
    accelerator="auto",
    devices="auto",
    precision=16,
    accumulate_grad_batches=2,
    callbacks=[checkpoint_callback, early_stop_callback]
)

trainer.fit(model, train_loader, val_loader)

'''Inference'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])

class TestDataset(Dataset):
    def __init__(self, df, transforms, processor):
        self.df = df
        self.transforms = transforms
        self.processor = processor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        image = cv2.imread(img_path)
        # 이미지 로드 실패 시 경고 출력 후 None 반환하여 건너뜁니다.
        if image is None:
            print(f"Warning: 이미지 로드 실패 - {img_path}. 건너뜁니다.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (1, C, H, W) -> (C, H, W)
        return {"pixel_values": pixel_values}

    def __len__(self):
        return len(self.df)

def test_collate_fn(batch):
    # None인 항목 제거
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("모든 이미지 로드에 실패했습니다.")
    images = [item["pixel_values"] for item in batch]
    images = torch.stack(images)
    return {"pixel_values": images}


#테스트 데이터 로드
folder_path = "/content/drive/MyDrive"
test_csv_path = "/content/drive/MyDrive/test.csv"

test_df = pd.read_csv(test_csv_path)
test_df['img_path'] = test_df["img_path"].apply(lambda x: os.path.join("/content/drive/MyDrive/test_upscaled", os.path.basename(x).replace('test', ''))).tolist()

# 모델 로드
model_name = "microsoft/swinv2-base-patch4-window16-256"
processor = AutoImageProcessor.from_pretrained(model_name, do_normalize=False)

test_dataset = TestDataset(test_df, transforms=test_transforms, processor=processor)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4, collate_fn=test_collate_fn, drop_last=False)

# 체크포인트 경로로
best_checkpoint_path = "/content/drive/MyDrive/results/swin_upscale/best-checkpoint-v1.ckpt" 

model = SwinClassifier.load_from_checkpoint(best_checkpoint_path, num_classes = num_classes)
model.to(device)
model.eval()

predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        # 배치 딕셔너리에서 pixel_values만 추출하여 device로 이동
        pixel_values = batch["pixel_values"].to(device)  # [B, C, H, W]
        logits = model(pixel_values)  # forward() 호출; logits shape: [B, num_classes]
        preds = torch.argmax(logits, dim=1)  # [B]
        predictions.extend(preds.cpu().numpy())

train_csv_path = "/content/drive/MyDrive/train.csv"
train_df = pd.read_csv(train_csv_path)
le = preprocessing.LabelEncoder()
le.fit(train_df["label"])

# 예측 결과(숫자)를 원래 클래스명으로 역변환
final_labels = le.inverse_transform(np.array(predictions))

# sample_submission.csv 파일을 불러와 예측 결과 적용
submission_csv_path = "/content/drive/MyDrive/sample_submission.csv"
submission_df = pd.read_csv(submission_csv_path)
submission_df["label"] = final_labels
submission_df.to_csv("submission33.csv", index=False)
