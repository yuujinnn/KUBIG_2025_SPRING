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


# ✅ 코사인 유사도 함수
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def custom_search(user_disease_name_kor, user_department, user_intention, user_question, top_k=3):
    start_time = time.time()
    # ✅ 사용자 입력 임베딩 생성
    emb_user_disease = embedding_fn.embed_query(user_disease_name_kor)
    emb_user_department = embedding_fn.embed_query(user_department)
    emb_user_intention = embedding_fn.embed_query(user_intention)
    emb_user_question = embedding_fn.embed_query(
        ", ".join(user_question[:4]) if isinstance(user_question, list) else user_question
    )

    # ✅ 컬렉션 메타데이터 및 문서 불러오기
    results = vectorstore.get(include=["metadatas", "documents"])
    metadatas = results["metadatas"]
    documents = results["documents"]

    scored_docs = []

    # ✅ 각 문서와 사용자 입력 간의 평균 유사도 계산
    for md, doc in zip(metadatas, documents):
        doc_id = md.get("fileName", "unknown")

        # 메타데이터에서 임베딩 불러오기
        emb_doc_disease = json.loads(md["disease_name_kor_vec"])
        emb_doc_dept = json.loads(md["department_vec"])
        emb_doc_intent = json.loads(md["intention_vec"])
        emb_doc_answer = json.loads(md["answer_vec"])

        # 평균 코사인 유사도 계산
        avg_sim = (
            cosine_sim(emb_user_disease, emb_doc_disease) +
            cosine_sim(emb_user_department, emb_doc_dept) +
            cosine_sim(emb_user_intention, emb_doc_intent) +
            cosine_sim(emb_user_question, emb_doc_answer)
        ) / 4.0

        scored_docs.append({
            "id": doc_id,
            "answer_text": md["answer_text"],
            "similarity": avg_sim
        })

    # ✅ 유사도 순으로 정렬 및 상위 top_k 반환
    scored_docs.sort(key=lambda x: x["similarity"], reverse=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n=== 검색 결과 ===")
    for rank, doc in enumerate(scored_docs[:top_k], start=1):
        print(f"[{rank}위] ID: {doc['id']} / 유사도: {doc['similarity']:.4f}")
        print(f"     → answer_text: {doc['answer_text']}...\n")
    print(f"소요시간:{elapsed_time}")
    return scored_docs[:top_k]

def custom_search_only_query(user_disease_name_kor, user_department, user_intention, user_question, top_k=3):
    start_time = time.time()
    emb_user_question = embedding_fn.embed_query(", ".join(user_question[:4]) if isinstance(user_question, list) else user_question)

    results = vectorstore.get(include=["metadatas", "documents"])
    metadatas = results["metadatas"]
    documents = results["documents"]

    scored_docs = []
    for md in metadatas:
        doc_id = md.get("fileName", "unknown")
        emb_doc_answer = json.loads(md["answer_vec"])

        sim_quest = cosine_sim(emb_user_question, emb_doc_answer)

        scored_docs.append({
            "id": doc_id,
            "answer_text": md["answer_text"],
            "similarity": sim_quest
        })

    scored_docs.sort(key=lambda x: x["similarity"], reverse=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("=== 검색 결과 ===")
    for rank, doc in enumerate(scored_docs[:top_k], start=1):
        print(f"[{rank}위] ID: {doc['id']} / 유사도: {doc['similarity']:.4f}")
        print(f"     → answer_text: {doc['answer_text']}...")
        print()
    print(f"소요시간:{elapsed_time}")
    
    return scored_docs[:top_k]

def custom_search_all(user_disease_name_kor, user_department, user_intention, user_question, user_keyword, top_k=3):
    start_time = time.time()
    emb_user_disease = embedding_fn.embed_query(user_disease_name_kor)
    emb_user_department = embedding_fn.embed_query(user_department)
    emb_user_intention = embedding_fn.embed_query(user_intention)
    emb_user_question = embedding_fn.embed_query(", ".join(user_question[:4]) if isinstance(user_question, list) else user_question)
    emb_user_keyword = embedding_fn.embed_query(", ".join(user_keyword[:4]) if isinstance(user_keyword, list) else user_keyword)

    results = vectorstore.get(include=["metadatas", "documents"])
    metadatas = results["metadatas"]
    documents = results["documents"]

    scored_docs = []

    for md in metadatas:
        doc_id = md.get("fileName", "unknown")

        emb_doc_disease = json.loads(md["disease_name_kor_vec"])
        emb_doc_dept = json.loads(md["department_vec"])
        emb_doc_intent = json.loads(md["intention_vec"])
        emb_doc_answer = json.loads(md["answer_vec"])

        avg_sim = (cosine_sim(emb_user_disease, emb_doc_disease) +
                   cosine_sim(emb_user_department, emb_doc_dept) +
                   cosine_sim(emb_user_intention, emb_doc_intent) +
                   cosine_sim(emb_user_question, emb_doc_answer) +
                   cosine_sim(emb_user_keyword, emb_doc_answer)
                   ) / 5.0

        scored_docs.append({
            "id": doc_id,
            "answer_text": md["answer_text"],
            "similarity": avg_sim
        })

    scored_docs.sort(key=lambda x: x["similarity"], reverse=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n=== 검색 결과 ===")
    for rank, doc in enumerate(scored_docs[:top_k], start=1):
        print(f"[{rank}위] ID: {doc['id']} / 유사도: {doc['similarity']:.4f}")
        print(f"     → answer_text: {doc['answer_text']}...\n")
    print(f"소요시간:{elapsed_time}")

    return scored_docs[:top_k]