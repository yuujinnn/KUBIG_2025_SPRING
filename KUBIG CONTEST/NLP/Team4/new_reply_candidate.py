import os
import json
import time
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

# ✅ 경로 및 설정
PERSIST_DIR = "/Users/yarlu/Desktop/KUBIG/NLP_PJ"
embedding_model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# ✅ 임베딩 모델 로드
print("🚀 임베딩 모델 로딩 중...")
embedding_fn = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ 임베딩 모델 로딩 완료!")

# ✅ 퍼시스턴스 파일(chroma.sqlite3)로부터 벡터스토어 로드
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_fn
)
print("✅ 퍼시스턴스 파일에서 벡터스토어 로드 완료!")

def custom_search_just_query(user_question, top_k=3):
    start_time = time.time()

    fixed_user_question = ", ".join(user_question[:4]) if isinstance(user_question, list) else user_question

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    results = retriever.get_relevant_documents(fixed_user_question)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n=== 검색 결과 ===")
    for rank, doc in enumerate(results, start=1):
        metadata = doc.metadata
        print(f"[{rank}위] ID: {metadata.get('fileName', 'unknown')} / 유사도: N/A (자동 계산)")
        print(f"department: {metadata.get('department_text', '없음')}")
        print(f"disease: {metadata.get('disease_kor_text', '없음')}")
        print(f"intention: {metadata.get('intention_text', '없음')}")
        print(f"     → answer_text: {metadata.get('answer_text', '없음')}...\n")

    print(f"🚀 소요 시간: {elapsed_time:.2f}초")
    return results

def custom_search_all(user_question, user_keyword, top_k=3):
    start_time = time.time()

    fixed_user_keyword = ", ".join(user_keyword[:4]) if isinstance(user_keyword, list) else user_keyword
    all_in_one = f"{user_question},{fixed_user_keyword}"

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    results = retriever.get_relevant_documents(all_in_one)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n=== 검색 결과 ===")
    for rank, doc in enumerate(results, start=1):
        metadata = doc.metadata
        print(f"[{rank}위] ID: {metadata.get('fileName', 'unknown')} / 유사도: N/A (자동 계산)")
        print(f"department: {metadata.get('department_text', '없음')}")
        print(f"disease: {metadata.get('disease_kor_text', '없음')}")
        print(f"intention: {metadata.get('intention_text', '없음')}")
        print(f"     → answer_text: {metadata.get('answer_text', '없음')}...\n")

    print(f"🚀 소요 시간: {elapsed_time:.2f}초")
    return results
