#evaluator.py

import random
import numpy as np
import torch
from ai_utils_bert import gpt_generate_response, compute_secret_embeddings, util
from evaluation import recall_k, MRR, NDCG

class LiarGameEvaluator:
    def __init__(self, topics):
        self.topics = topics
        self.secret_words = {topic: list(words) for topic, words in topics.items()}
        self.embeddings = compute_secret_embeddings([word for words in topics.values() for word in words])
    
    def generate_ai_truth_description(self, secret_word):
        """
        진실 플레이어 AI의 설명을 생성합니다.
        """
        system_prompt = (
            "You are a truthful player in the Liar Game. The rules of the game are as follows:\n"
            "1. All truthful players share the same secret word, but the liar does not know it.\n"
            "2. Each player must describe the secret word in one sentence without directly revealing it.\n"
            "3. Descriptions should be subtle and not too obvious, so the liar cannot easily guess the word.\n"
            "4. After all players give their descriptions, they will discuss and vote to identify the liar.\n"
            "5. If the liar is correctly identified, they will have one last chance to guess the secret word.\n"
            "6. If the liar guesses correctly, they win. If they fail, the truthful players win.\n\n"
            f"Your secret word for this round is '{secret_word}'. "
            "Write a one-sentence description of this word using abstract and metaphorical expressions. "
            "Do not mention the secret word directly. "
            "Do not give any clear hints that would help the liar. "
            "Make your description sound natural and logical. "
            "Respond in Korean."
        )
        return gpt_generate_response(system_prompt)
    
    def predict_top_5_words(self, descriptions, topic):
        """
        플레이어들이 제공한 설명을 기반으로 예측된 단어 5개를 반환
        """
        description_embeddings = compute_secret_embeddings(descriptions)
        description_embedding = np.mean(list(description_embeddings.values()), axis=0)
        prediction_scores = {}
        for word in self.secret_words[topic]:
            word_embedding = self.embeddings[word]
            similarity = util.cos_sim(description_embedding, word_embedding).item()
            prediction_scores[word] = similarity
        
        sorted_predictions = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_predictions[:5]]
    
    def evaluate_model(self, num_trials=30):
        """
        모델의 예측 성능을 평가하는 함수
        - 랜덤한 30개의 단어를 선택
        - 각 단어에 대해 3개의 힌트를 생성
        - 예측된 단어 5개 선정 및 평가
        """
        recall_k_scores = []
        MRR_scores = []
        NDCG_scores = []
        
        for trial in range(num_trials):
            print(f'===Trial {trial+1}/{num_trials} ===')
            topic = random.choice(list(self.topics.keys()))
            secret_word = random.choice(self.topics[topic])
            descriptions = [self.generate_ai_truth_description(secret_word) for _ in range(3)]
            print(f'Secret word: {secret_word}')
            print('Generated descriptions:', descriptions)
            top_5_predicted = self.predict_top_5_words(descriptions, topic)
            print('Predicted top 5 words:', top_5_predicted)
            
            prediction_scores = {}
            for word in top_5_predicted:
                avg_score = 0
                hints = [self.generate_ai_truth_description(word) for _ in range(3)]
                for hint in hints:
                    hint_embedding = compute_secret_embeddings([hint])[hint]
                    word_embedding = self.embeddings[word]
                    similarity = util.cos_sim(hint_embedding, word_embedding).item()
                    avg_score += similarity
                avg_score /= len(hints)
                prediction_scores[word] = avg_score
            
            sorted_predictions = dict(sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True))
            
            # 평가 수행
            recall_k_scores.append(recall_k(sorted_predictions, secret_word, 3))
            MRR_scores.append(MRR(sorted_predictions, secret_word))
            NDCG_scores.append(NDCG(sorted_predictions, secret_word))
        
        # 평균 성능 출력
        print("\n=== 모델 성능 평가 결과 ===")
        print(f"Recall@K 평균: {np.mean(recall_k_scores):.4f}")
        print(f"MRR 평균: {np.mean(MRR_scores):.4f}")
        print(f"NDCG 평균: {np.mean(NDCG_scores):.4f}")
        return np.mean(recall_k_scores), np.mean(MRR_scores), np.mean(NDCG_scores)

# 기존 secret words 데이터 유지
topics = {
            "place": [
                "공원", "도서관", "해변", "산", "도시", "마을", "강", "호수", "광장", "카페",
                "식당", "학교", "병원", "극장", "박물관", "시장", "공항", "체육관", "지하철역", "호텔"
            ],
            "food": [
                "초콜릿", "피자", "라면", "스시", "햄버거", "김치", "비빔밥", "떡볶이", "파스타", "스테이크",
                "샐러드", "치킨", "감자튀김", "샌드위치", "토스트", "오믈렛", "초밥", "케이크", "아이스크림", "컵라면"
            ],
            "job": [
                "의사", "변호사", "요리사", "교사", "프로그래머", "디자이너", "엔지니어", "간호사", "회계사", "군인",
                "경찰", "소방관", "조종사", "비서", "관리자", "연구원", "작가", "예술가", "음악가", "배우"
            ],
            "object": [
                "컴퓨터", "휴대폰", "책상", "의자", "시계", "텔레비전", "냉장고", "전자레인지", "세탁기", "전구",
                "수도꼭지", "마우스", "책상", "프린터", "카메라", "스피커", "이어폰", "헤드폰", "책", "노트"
            ],
            "character": [
                "해리포터", "슈퍼맨", "아이언맨", "스파이더맨", "신데렐라", "닥터 스트레인지", "가모라", "타노스", "배트맨", "원더우먼",
                "플래시", "캡틴 아메리카", "토르", "헐크", "로켓 라쿤", "데드풀", "엑스맨", "스파이더우먼", "레드후드", "닉 퓨리"
            ]
}

evaluator = LiarGameEvaluator(topics)

# 모델 평가 실행
evaluator.evaluate_model()


