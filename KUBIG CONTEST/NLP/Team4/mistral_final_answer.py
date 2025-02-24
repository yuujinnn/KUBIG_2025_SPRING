import json
import requests
import fast_main  # fast_main.py에서 user_input 및 검색된 답변 데이터 가져오기

# Mistral API Key (테스트용 - 운영 환경에서는 환경 변수 사용 권장)
MISTRAL_API_KEY = "key"

# Mistral API 설정
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# API 요청 함수
def get_mistral_response(prompt):
    """
    Mistral API를 호출하여 최종 답변을 생성하는 함수
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-medium",  
        "messages": [
            {"role": "system", "content": "You are a professional medical chatbot. You must provide reliable and trustworthy information. The final response must be written in Korean."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"API Error: {response.status_code} - {response.json()}"

# 최종 프롬프트 생성 함수
def generate_prompt(user_question, documents):
    """
    유저 질문과 유사한 답변 데이터를 결합하여 Mistral 프롬프트 생성
    """
    prompt = f"""
    Context:
    You are an AI-powered healthcare assistant responsible for providing clear, accurate, and coherent medical information. Below are multiple relevant answers retrieved from a medical database that relate to the user's question.

    Your task is to **synthesize** these answers into **a single well-structured, natural, and cohesive response** in **Korean**. The response should read fluently and logically, as if written by a professional healthcare provider.

    **Guidelines:**
    1. **Avoid simply listing answers separately.** Instead, integrate them smoothly into a unified, well-organized response.
    2. **Ensure logical flow and coherence.** Organize the response so that it feels structured, with a natural progression of ideas.
    3. **Prioritize clarity and completeness.** Ensure the final response fully answers the user's question in an easy-to-understand yet professional manner.
    4. **Do not fabricate information.** Only use the provided medical answers to construct your response.

    ---

    User Question:
    {user_question}

    Relevant Answers:
    """
    for idx, doc in enumerate(documents, 1):
        metadata = doc.metadata
        prompt += f"\nAnswer {idx}:\nDepartment: {metadata.get('department_text', '없음')}\nDisease: {metadata.get('disease_kor_text', '없음')}\nIntention: {metadata.get('intention_text', '없음')}\nContent:\n"
        prompt += f"- {metadata.get('answer_text', '없음')}\n"

    prompt += "\nProvide a final response in Korean based on the above information."
    return prompt

# fast_main.py에서 사용자 입력 및 top_k 답변 가져오기
user_question = fast_main.user_input  # fast_main.py에서 입력된 질문 가져오기
documents = fast_main.custom_search_just_query(user_question, top_k=3)  # 검색된 Top 3 답변 데이터 가져오기

# 프롬프트 생성
prompt = generate_prompt(user_question, documents)

# Mistral API 호출 및 최종 답변 생성
final_answer = get_mistral_response(prompt)

# 최종 답변 출력
print("\n최종 답변:\n", final_answer)
