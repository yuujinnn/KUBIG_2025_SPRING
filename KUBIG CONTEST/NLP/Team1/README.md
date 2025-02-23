
# 난 진짜 라이어 아님.

NLP Team1 장어구이

## Members
20기 김민재, 김재훈 21기 강서연, 남동연


## Introduction
사용자가 혼자서도 즐길 수 있는 라이어 게임을 구현했습니다. LLM과 플레이하는 라이어 게임으로, NLP 모델을 활용하여 라이어의 단어 예측, 설명 생성, 문장 유사도 기반 투표 시스템을 기반으로 합니다.

**🎭 웹 플레이 하러 가기**

https://kubig-nlpteam1-liargame.streamlit.app/

![alt text](<스크린샷 2025-02-23 175237.png>)


## 기능 개요
게임 진행 방식

1. 역할 배정: 랜덤하게 라이어와 진실 플레이어를 배정
2. 비밀 단어 선택: 장소, 음식, 직업, 사물, 캐릭터 등의 주제에서 랜덤 단어 선정
3. 설명 생성:
- 진실 플레이어: GPT API를 이용해 단어를 설명
- 라이어: 다른 플레이어의 설명을 NLP 기반 모델(Bert, Klue, Koe5 시도)을 활용해 분석하고, 가장 유사한 단어를 예측한 후 거짓된 설명 생성
4. 투표:
- 모든 플레이어는 라이어라고 생각되는 사람에게 투표
- AI는 KLUE-RoBERTa 기반 문장 유사도 분석을 통해 의심스러운 사람에게 투표

## Architecture
- GPT-4
    - GPT 기반 LLM을 활용하여 AI 플레이어들의 설명 생성 
    - 프롬프트 설계를 통해 단어에 대한 간접적인 설명 유사도
- 문장 임베딩 및 단어 예측 모델 (Bert, Klue, Koe5)
    - XLM-RoBERTa: 100가지 언어를 지원하는 다국적 BERT 기반 모델
    - KLUE-RoBERTa: 한국어 특화 문장 임베딩 모델
    - KoE5: BERT 기반의 E5 모델을 한국어에 맞게 Fine-tuning
    - 단어 후보들 위키피디아 문서 크롤링, 단어 전처리, 문장 단위 분할 후 임베딩 사전 학습
    ![alt text](image-2.png)
- 모델 평가 지표
    - Recall@K: 정답이 상위 K개의 예측 내 포함될 확률 측정
    - MRR (Mean Reciprocal Rank): 정답이 처음 등장하는 순위의 역수 평균
    - NDCG (Normalized Discounted Cumulative Gain): 정답이 상위에 위치할수록 높은 점수 부여
    - 평가 결과
    ![alt text](image-1.png)
- 문장 유사도 기반 라이어 탐지 및 투표 시스템
    - KLUE-STS 데이터셋을 활용한 KLUE-RoBERTa 추가 학습
    - (1 - 평균 유사도) 값을 사용해 의심도 점수 부여 후 확률 기반 투표 진행
- Frontend (Streamlit)
    - 게임 UI 제공: 역할 배정, 설명 입력, 투표 기능 구현



## Structure
```
📂 Team1
|── 📂 app                      # 메인 애플리케이션 폴더
|    │── README.md               # 앱 프로젝트 설명 및 실행 방법
|    │── requirements.txt        # 프로젝트 실행에 필요한 패키지 목록
|    │── app.py                  # Streamlit 기반 메인 웹 애플리케이션 실행 파일
|    |── (앱 프로젝트 관련 중복 파일들)
│
│── 📂 dev                # 개발용 코드 폴더
│    ├── main.py           # 개발 환경에서 실행할 메인 스크립트
│    ├── liar_game.py      # 게임 로직 테스트용 스크립트
│    ├── player.py         # 플레이어 동작 테스트 코드
│    ├── evaluation.py     # 모델 평가 테스트 코드
│    ├── ai_utils_bert.py  # BERT 기반 NLP 유틸리티 함수들
│    ├── evaluator_bert.py # BERT 기반 문장 유사도 평가 모델 (klue, koe5 따로 존재)
│
│── 📂 model_training        # 모델 학습 및 평가 관련 폴더
|    |──(임베딩 및 문장 유사도 모델 훈련 코드)
```
