# 🏥 RAG 기반 의료 Q&A 시스템 (ChatUpstage 기반)

## 🔍 프로젝트 개요
이 프로젝트는 **RAG (Retrieval-Augmented Generation) 기반의 의료 Q&A 시스템**을 구축하는 것을 목표로 합니다.  
사용자가 의료 질문을 입력하면, **키워드 및 의도 분석 → 의료 데이터베이스 검색 → LLM을 통한 자연스러운 답변 생성** 과정을 거쳐 **정확하고 신뢰할 수 있는 의료 정보**를 제공합니다.  

본 프로젝트는 **ChatUpstage API**를 활용하여 최종 응답을 생성합니다.

---
## 주요 기능

**사용자 질문 분석 및 키워드 추출**  
- `class_extraction.py`: 질문에서 **질병 카테고리, 질병명, 의도(Intention) 추출**  
- `keyword_extraction.py`: 질문에서 **의학적 핵심 키워드 4개 추출**  

**유사 의료 답변 검색 (RAG 기반)**  
- `new_reply_candidate.py`:  
  - `custom_search_just_query()` → **문장 임베딩 기반 유사도 검색**
- `ChromaDB` 벡터스토어 활용하여 **사전 저장된 의료 데이터**에서 관련 문서 검색  

**ChatUpstage API를 활용한 자연어 답변 생성**  
- 검색된 **Top 3 관련 문서**를 조합하여 `ChatUpstage API`를 활용한 최종 답변 생성  
- `generate_prompt()` 함수를 통해 **프롬프트 최적화 및 일관성 있는 응답 제공**  
