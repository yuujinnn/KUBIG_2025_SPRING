from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
from dotenv import load_dotenv

# OpenAI API 설정
OPENAI_API_KEY = "key"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def extract_medical_keywords(current_query):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical AI that specializes in extracting concise and meaningful keywords from medical-related user input. "
                    "Your goal is to identify 4 core keywords that represent the most important medical terms, topics, or concepts within the input. "
                    "These keywords should be specific and medically relevant, such as diseases, symptoms, definitions, or biological processes. "
                    "Avoid overly broad or generic words like '아프다', '질병', '질문', '정보', '설명', or other non-medical terms."
                )
            },
            {
                "role": "user",
                "content": (
                    f"The user is asking: '{current_query}'.\n"
                    "Please extract exactly 4 keywords that are the most important medical terms or phrases representing the user's query. "
                    "Focus on key medical topics such as diseases, conditions, symptoms, treatments, definitions, or diagnostic terms. "
                    "Output the keywords only as a comma-separated list. Do not include any explanations, extra text, or labels like 'Keywords:'."
                )
            }
        ]
    )

    raw_keywords = response.choices[0].message.content.strip().splitlines()  # type: ignore
    keywords = [phrase.strip() for phrase in raw_keywords if phrase.strip()]
    return keywords

def generate_embedding_average(keywords):
    # 각 키워드별로 임베딩 생성
    embs = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=keywords
    )

    # 각 키워드의 임베딩 벡터 추출
    embeddings = [emb.data.embedding for emb in embs.data]

    # 평균 벡터 계산
    average_embedding = np.mean(embeddings, axis=0)

    return average_embedding

def remove_stopwords(sentence, stopword_file_path):
    # 1. Stopword 리스트 불러오기
    with open(stopword_file_path, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().strip().split('\n'))

    # 2. 단어 단위로 split하고 stopwords 제거
    words = re.findall(r'\b\w+\b', sentence)  # 한글 단어 추출
    filtered_words = [word for word in words if word not in stopwords]

    # 3. Stopword가 제거된 문장 생성
    cleaned_sentence = ' '.join(filtered_words)
    return cleaned_sentence

def embed_sentence_without_stopwords(sentence, stopword_file_path):
    # Stopword 제거
    cleaned_sentence = remove_stopwords(sentence, stopword_file_path)

    # KoSentenceBERT 모델 로드 (한국어 특화된 문장 임베딩 모델)
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    # 문장 임베딩 생성
    embedding = model.encode(cleaned_sentence)
    return embedding