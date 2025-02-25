import os
import glob
import json
import pickle
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ 경로 및 설정
PERSIST_DIR = "/Users/yarlu/Desktop/KUBIG/NLP_PJ"
COLLECTION_NAME = "my_collection"
PICKLE_PATH = "collection_backup_final.pkl"

# ✅ 임베딩 모델 로드
embedding_model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
print("🚀 임베딩 모델 로딩 중...")
embedding_fn = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    encode_kwargs={"normalize_embeddings": True},
)
print("✅ 임베딩 모델 로딩 완료!")

# ✅ 벡터스토어 로드 또는 새로 생성
try:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_fn
    )
    print("✅ 기존 벡터스토어 로드 완료!")
except Exception as e:
    print(f"⚠️ 기존 벡터스토어 로드 실패: {e}")
    vectorstore = Chroma.from_documents(
        documents=[],  # 빈 문서로 생성
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR
    )
    print("🆕 새로운 벡터스토어 생성!")

# ✅ 임베딩 생성 및 벡터스토어 추가 함수
def build_or_load_embeddings(json_dir="2.답변"):
    """DB에 임베딩이 없으면 생성하고, 있으면 스킵하며 진행 상황을 tqdm으로 표시"""
    metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
    existing_ids = {md["fileName"] for md in metadatas} if metadatas else set()

    json_paths = glob.glob(f"{json_dir}/**/*.json", recursive=True)
    print(f"🔍 총 {len(json_paths)}개의 JSON 파일 발견!")

    added_count = 0

    for path in tqdm(json_paths, desc="임베딩 처리 진행 중", unit="파일"):
        file_id = os.path.basename(path)
        if file_id in existing_ids:
            continue  # 이미 처리된 파일은 스킵

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        disease_kor_text = data.get("disease_name", {}).get("kor", "")
        department_text = ", ".join(data.get("department", [])) if isinstance(data.get("department", []), list) else data.get("department", "")
        intention_text = data.get("intention", "")
        answer_text = " ".join(data.get("answer", {}).values()) if isinstance(data.get("answer", {}), dict) else str(data.get("answer", ""))

        # ✅ 임베딩 생성
        emb_disease = embedding_fn.embed_query(disease_kor_text)
        emb_department = embedding_fn.embed_query(department_text)
        emb_intention = embedding_fn.embed_query(intention_text)
        emb_answer = embedding_fn.embed_query(answer_text)

        metadata = {
            "fileName": file_id,
            "disease_name_kor_vec": json.dumps(emb_disease),
            "department_vec": json.dumps(emb_department),
            "intention_vec": json.dumps(emb_intention),
            "answer_vec": json.dumps(emb_answer),
            "disease_name_kor_text": disease_kor_text,
            "department_text": department_text,
            "intention_text": intention_text,
            "answer_text": answer_text
        }

        # ✅ 벡터스토어에 추가
        vectorstore.add_texts(
            texts=[answer_text],
            metadatas=[metadata],
            ids=[file_id]
        )
        added_count += 1

    # ✅ 퍼시스턴스 저장
    vectorstore.persist()
    print(f"💾 벡터스토어가 '{PERSIST_DIR}'에 저장되었습니다.")

    if added_count:
        print(f"✅ {added_count}개의 새 임베딩 추가 완료.")
    else:
        print("✅ 기존 임베딩 사용 (새로 추가된 파일 없음).")

    # ✅ Pickle로 백업 저장
    data = vectorstore.get(include=["metadatas", "documents"])
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Pickle 파일로 '{PICKLE_PATH}'에 백업 저장 완료!")

# ✅ 임베딩 빌드 실행
build_or_load_embeddings()