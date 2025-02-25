from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import mean_squared_error

# KLUE STS 데이터셋 로드
dataset = load_dataset('klue', 'sts')

# 토크나이저 및 모델 불러오기
MODEL_NAME = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

# 문장을 토큰화하는 전처리 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'],
                     truncation=True, padding='max_length', max_length=128)

# 데이터셋 토큰화
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 배치 내 label 변환 함수 정의
def convert_label(batch_labels):
    """
    각 label을 float 값으로 변환하여 리스트로 반환
    """
    new_labels = []
    for lab in batch_labels:
        if isinstance(lab, dict):
            if "label" in lab:
                new_labels.append(float(lab["label"]))
            elif "score" in lab:
                new_labels.append(float(lab["score"]))
            else:
                new_labels.append(0.0)
        elif isinstance(lab, list) and len(lab) > 0:
            new_labels.append(float(lab[0]))
        elif isinstance(lab, (int, float)):
            new_labels.append(float(lab))
        else:
            new_labels.append(0.0)
    return new_labels

# 배치 모드로 label 변환 적용
tokenized_datasets = tokenized_datasets.map(
    lambda x: {"labels": convert_label(x["labels"])},
    batched=True
)

# 존재하는 컬럼만 삭제
columns_to_remove = ['guid', 'sentence1', 'sentence2', 'source']
current_columns = tokenized_datasets.column_names['train']
columns_to_remove = [col for col in columns_to_remove if col in current_columns]

if columns_to_remove:
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

# PyTorch tensor 형식으로 설정
tokenized_datasets.set_format("torch")

# 훈련 및 평가 데이터 분리
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# MSE 손실 함수
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.flatten()
    mse = mean_squared_error(labels, preds)
    return {"mse": mse}

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 모델 학습
trainer.train()


model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")