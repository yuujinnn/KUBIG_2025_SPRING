import os
import sys
import glob
import torch
import nibabel as nib
import numpy as np
import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import random

# 랜덤 시드 설정
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(1)  # 실행 시 항상 같은 결과 보장

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MRI 데이터 폴더 및 CSV 파일 경로
mri_folder = "./nifti/"
csv_path = "./oasis_cross_sectional.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# MRI 파일 리스트 가져오기
mri_files = glob.glob(os.path.join(mri_folder, "**/*.nii"))
print(f"총 {len(mri_files)}개의 MRI 파일을 찾았습니다.")

# CSV 파일 ID 정리
def extract_id_from_filename(filename):
    return os.path.basename(filename).split('_')[1]

def extract_id_from_csv(original_id):
    return original_id.split('_')[1]

df['ID'] = df['ID'].apply(extract_id_from_csv)

# MRI 파일과 CSV 매칭
matched_data = []
for filename in mri_files:
    mri_id = extract_id_from_filename(filename)
    row = df[df['ID'] == mri_id]
    if not row.empty:
        # CSV에서 CLIP_Text 전체를 읽어온 다음
        full_text = row.iloc[0]["CLIP_Text"]

        # 여기서 원하는 방식으로 필요한 부분만 추출
        # 예시) "This is an MRI scan of a 74-year-old female" 까지 잘라내는 간단한 방법

        # 1) 단순히 키워드로 잘라내는 경우
        # 특정 구문 직후로만 잘라내고 싶다면 (키워드 이후 문장 제거 등)
        keyword = "male"
        # 만약 CSV 내용 중에 해당 keyword가 있다면 그 부분까지만 추출
        if keyword in full_text:
            idx = full_text.index(keyword) + len(keyword)
            clip_text = full_text[:idx]
        else:
            # keyword가 없는 경우, 기본적으로 full_text를 그대로 쓰거나, 빈 문자열 처리
            clip_text = full_text
        #print(clip_text)

        # 2) 문장 단위로 잘라내는 경우
        # 어떤 문장부호(.) 기준으로 자르고 첫 문장만 쓰고 싶다면 예시:
        # sentences = full_text.split('.')
        # clip_text = sentences[0] + '.' if len(sentences) > 0 else ''

        # 이후 matched_data에 저장
        matched_data.append({
            "mri_path": filename,
            "clip_text": clip_text,
            "label": row.iloc[0]["CDR"]
        })

print(f"총 {len(matched_data)}개의 MRI 데이터가 CSV와 매칭되었습니다.")

# 데이터셋 클래스 정의
class OASISDataset(Dataset):
    def __init__(self, matched_data):
        self.data = matched_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        nii_img = nib.load(sample['mri_path'])
        img_data = nii_img.get_fdata()

        # MRI 데이터 변환
        if len(img_data.shape) == 3:
            img_data = np.expand_dims(img_data, axis=0)
        elif len(img_data.shape) == 4 and img_data.shape[-1] == 1:
            img_data = img_data[..., 0]
            img_data = np.expand_dims(img_data, axis=0)
        else:
            raise ValueError(f"❌ MRI 데이터 차원이 예상과 다름: {img_data.shape}")

        # 정규화
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)
        img_data = F.interpolate(img_data, size=(16, 112, 112), mode='trilinear', align_corners=False)
        img_data = img_data.squeeze(0)

        return img_data, sample['clip_text'], torch.tensor(sample['label'], dtype=torch.long)

# 데이터 분할
def split_data(matched_data, seed=1):
    set_random_seed(seed)
    random.shuffle(matched_data)
    train_size = int(0.8 * len(matched_data))
    val_size = int(0.1 * len(matched_data))
    test_size = len(matched_data) - train_size - val_size

    return (
        OASISDataset(matched_data[:train_size]),
        OASISDataset(matched_data[train_size:train_size+val_size]),
        OASISDataset(matched_data[train_size+val_size:])
    )

# MedicalNet MRI 인코더
sys.path.append("./MedicalNet")
from MedicalNet.model import generate_model

class MedicalNetEncoder(nn.Module):
    def __init__(self, feature_dim=512, use_pretrained=False):
        super(MedicalNetEncoder, self).__init__()
        opt = argparse.Namespace(
            model='resnet', model_depth=10, input_W=112, input_H=112, input_D=16,
            resnet_shortcut='B', no_cuda=False, n_seg_classes=2, gpu_id=[0],
            phase='train', pretrain_path="./MedicalNet/resnet_10.pth" if use_pretrained else None,
            new_layer_names=[]
        )
        self.model, _ = generate_model(opt)
        self.model.fc = nn.Identity()
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(TextEncoder, self).__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, text):
        if isinstance(text, list):
            text = text[0]  # CLIP_Text가 리스트 형태로 들어오는 경우 대비
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        return text_features

# 멀티모달 분류 모델 정의
class MultimodalClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=4):
        super(MultimodalClassifier, self).__init__()
        # 6272는 mri_encoder의 출력 shape에 따라 달라질 수 있음
        self.mri_fc = nn.Linear(6272, feature_dim)
        self.text_fc = nn.Linear(512, feature_dim)
        self.relu = nn.ReLU()
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, mri_features, text_features):
        mri_features = mri_features.view(mri_features.size(0), -1)
        mri_features = self.mri_fc(mri_features)
        text_features = self.text_fc(text_features)
        fused_features = torch.cat((mri_features, text_features), dim=1)
        fused_features = self.relu(self.fusion_layer(fused_features))
        return self.classifier(fused_features)

# ---------------------------
# (1) MRI-Only 학습 및 테스트
# ---------------------------
def train_mri_only():
    print("\n🚀 Training MRI-Only Model (ResNet-10 + Classifier) 🚀\n")
    train_dataset, val_dataset, test_dataset = split_data(matched_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mri_encoder = MedicalNetEncoder(feature_dim=512, use_pretrained=True).to(device)

    # mri_encoder 출력 차원을 자동계산하여 classifier 크기 맞춤
    sample_input = torch.randn(1, 1, 16, 112, 112).to(device)
    sample_output = mri_encoder(sample_input)
    feature_dim = sample_output.view(sample_output.size(0), -1).shape[1]  # Flatten 후 크기

    #classifier = nn.Linear(feature_dim, 4).to(device)
    classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        ).to(device)


    optimizer = torch.optim.Adam(list(mri_encoder.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        mri_encoder.train()
        classifier.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (MRI-Only)", unit="batch"):
            mri_tensor, _, labels = batch
            mri_tensor, labels = mri_tensor.to(device), labels.to(device)

            mri_features = mri_encoder(mri_tensor)
            mri_features = mri_features.view(mri_features.size(0), -1)
            output = classifier(mri_features)

            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    print("\n✅ MRI-Only Model Training Complete!\n")
    # 학습이 끝난 모델과 test_loader를 리턴
    return mri_encoder, classifier, test_loader

def test_mri_only(mri_encoder, classifier, test_loader):
    print("\n🔎 Testing MRI-Only Model...\n")
    mri_encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for mri_tensor, _, labels in test_loader:
            mri_tensor, labels = mri_tensor.to(device), labels.to(device)

            # 추론
            mri_features = mri_encoder(mri_tensor)
            mri_features = mri_features.view(mri_features.size(0), -1)
            output = classifier(mri_features)

            # 예측 및 정확도 계산
            _, preds = torch.max(output, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f"✅ MRI-Only Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


# ---------------------------
# (2) Multimodal 학습 및 테스트
# ---------------------------
def train_multimodal():
    print("\n🚀 Training Multimodal Model (MRI + CLIP Text) 🚀\n")

    train_dataset, val_dataset, test_dataset = split_data(matched_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mri_encoder = MedicalNetEncoder(feature_dim=512, use_pretrained=True).to(device)
    text_encoder = TextEncoder().to(device)
    # multimodal 분류모델
    # mri_encoder 출력 채널(6272 가정) 자동 계산
    sample_input = torch.randn(1, 1, 16, 112, 112).to(device)
    sample_mri_features = mri_encoder(sample_input)
    mri_feature_dim = sample_mri_features.view(sample_mri_features.size(0), -1).shape[1]

    # text_encoder 출력 채널(512) + mri_feature_dim -> fusion
    # 여기서는 MultimodalClassifier 사용 시
    # 내부적으로 mri_fc(in_features=6272)로 계산.
    # mri_feature_dim이 6272가 아니라면 해당 값으로 수정 필요.
    # 일단 아래처럼 실제 계산된 차원을 바로 집어넣도록 합니다.
    multimodal_model = MultimodalClassifier(feature_dim=512, num_classes=4).to(device)
    multimodal_model.mri_fc = nn.Linear(mri_feature_dim, 512).to(device)

    optimizer = torch.optim.Adam(
        list(mri_encoder.parameters()) + list(text_encoder.parameters()) + list(multimodal_model.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        mri_encoder.train()
        text_encoder.model.train()
        multimodal_model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Multimodal)", unit="batch"):
            mri_tensor, text_inputs, labels = batch
            mri_tensor, labels = mri_tensor.to(device), labels.to(device)

            # 텍스트 임베딩 추출
            text_inputs_proc = text_encoder.processor(text=text_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = text_encoder.model.get_text_features(**text_inputs_proc)

            # MRI 임베딩 추출
            mri_features = mri_encoder(mri_tensor)

            # 멀티모달 분류
            output = multimodal_model(mri_features, text_features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed!")

    print("\n✅ Multimodal Model Training Complete!\n")
    return mri_encoder, text_encoder, multimodal_model, test_loader

def test_multimodal(mri_encoder, text_encoder, multimodal_model, test_loader):
    print("\n🔎 Testing Multimodal Model...\n")
    mri_encoder.eval()
    text_encoder.model.eval()
    multimodal_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for mri_tensor, text_inputs, labels in test_loader:
            mri_tensor, labels = mri_tensor.to(device), labels.to(device)

            # 텍스트 임베딩 추출
            text_inputs_proc = text_encoder.processor(text=text_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = text_encoder.model.get_text_features(**text_inputs_proc)

            # MRI 임베딩 추출
            mri_features = mri_encoder(mri_tensor)

            # 멀티모달 분류 추론
            output = multimodal_model(mri_features, text_features)

            # 예측 및 정확도 계산
            _, preds = torch.max(output, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f"✅ Multimodal Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    # 1) MRI-Only 학습 후 테스트
    mri_encoder, classifier, test_loader_mri = train_mri_only()
    test_mri_only(mri_encoder, classifier, test_loader_mri)

    # 2) Multimodal (MRI + CLIP Text) 학습 후 테스트
    mri_encoder_multi, text_encoder, multimodal_model, test_loader_multi = train_multimodal()
    test_multimodal(mri_encoder_multi, text_encoder, multimodal_model, test_loader_multi)
