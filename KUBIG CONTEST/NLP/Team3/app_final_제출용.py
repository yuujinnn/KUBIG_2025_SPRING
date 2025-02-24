### python -m streamlit run app.py
### conda activate /Users/happykuma/miniconda3

import os
import requests
from bs4 import BeautifulSoup
import chromadb  # type: ignore
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # type: ignore
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv
import streamlit as st

# 한국 시간 라이브러리
from datetime import datetime
import pytz

# naver api 검색 라이브러리
import urllib.request
import json
import uuid
import re
import ssl
import urllib.parse
import numpy as np
import pandas as pd

# selenium 라이브러리리
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


##### 마지막에 변경하기 #####

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")

# 네이버 api 설정
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

# 오늘 날짜 설정
kst = pytz.timezone("Asia/Seoul")
today_kst = datetime.now(kst).strftime("%a, %d %b %Y")  # 한국 시간 기준 오늘 날짜 포맷

# 모델 설정
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o-mini"

# OpenAI 및 ChromaDB 초기화
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL
)

###################################
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
corpus_collection = chroma_client.get_or_create_collection(
    name='NEWS',
    embedding_function=embedding_function
)

# RSS XML에서 뉴스 제목과 링크 추출
def fetch_news_titles(xml_url):
    response = requests.get(xml_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'lxml-xml')

    articles = []
    for item in soup.find_all('item', limit=5):  # 최신 뉴스 5개 가져오기
        title = item.find('title').text
        link = item.find('link').text
        articles.append((title, link))
    return articles

# 텍스트 chunk 분할 함수 
def smart_chunk_splitter(texts, titles, dates, max_chunk_size=1500):
    chunks = []

    if isinstance(texts, str):
        texts = [texts]
        titles = [titles]
        dates = [dates]

    for text, title, date in zip(texts, titles, dates):
        current_chunk = ""
        sentences = text.split('.')

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue  # 빈 문장은 스킵

            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += sentence + '. '

            else:
                current_chunk = f"title : {title}, date : {date}, content : {current_chunk}"
                chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        if current_chunk:
            chunks.append(f"title: {title}, date: {date}, content: {current_chunk.strip()}")

    return chunks

# # 텍스트 chunk 분할 함수 2 (원본)
# def smart_chunk_splitter2(text, max_chunk_size=100):
#     sentences = text.split('. ')
#     chunks = []
#     current_chunk = ""

#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
#             current_chunk += sentence + '. '
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + '. '

#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# keyword 추출 프롬프트 - 1
def create_chat_prompt1(user_query):
    system_prompt = """ 당신은 사용자가 입력한 문장에서 웹 검색에 최적화된 키워드를 추출하는 AI입니다.

            📌 **지침**
            1. **가장 중요한 정보**를 포착한 키워드를 **최대 2개까지** 추출하세요. 반드시 하나 이상 추출하세요.   
            2. **복합명사**는 가능한 한 **하나로 묶어서** 추출하세요.
            3. 한글로 답변해 주세요. 반드시 **문장 안의 단어**로만 답변해 주세요. 
            4. **불필요한 단어(예: "관련", "뉴스", "요약", "정보", "소식")는 제외하세요.**
            5. 출력 형식: **키워드만 공백으로 구분**해 출력하세요. 추가적인 설명이나 접두어 없이 순수한 키워드만 포함해야 합니다.
               예) `손흥민 경기`

            #Sentence: 손흥민 선수가 최근 경기에서 어떤 결과를 냈는지 궁금해.
            #Keyword: 손흥민 경기

            #Sentence: 캡틴아메리카 복장을 한 사람이 국회에 난입했다고 하는데, 그 이유가 뭐야?
            #Keyword: 캡틴아메리카 난입
         
            #Sentence: 삼성, LG 관련 뉴스 알려줘
            #Keyword: 삼성 LG

            #Sentence: 오늘의 뉴스 알려줘
            #Keyword: 오늘
            """
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"""
            #Sentence: {user_query}
            #Keyword:
        """
        }
    ]

# 쿼리에서 keyword 추출
def generate_keywords(user_query):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=create_chat_prompt1(user_query),
        max_tokens=1500
    )
    return response.choices[0].message.content.replace("#Keyword:", "").strip() # type: ignore

# naver api로 뉴스 검색
def find_document(keyword, num_doc):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    def fetch_news(keyword, client_id, client_secret, num_doc):
        encText = urllib.parse.quote(keyword)
        url = f"https://openapi.naver.com/v1/search/news?query={encText}&display={num_doc}&sort=sim" # display로 오타 수정

        context = ssl._create_unverified_context()

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        response = urllib.request.urlopen(request, context=context)
        if response.getcode() != 200:
            print("Error Code:", response.getcode())
            return []

        response_body = response.read().decode('utf-8')
        news_data = json.loads(response_body)

        return news_data.get('items', [])

    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")

    news_items = fetch_news(keyword, client_id, client_secret, num_doc)

    contents = []
    titles = []
    dates = []
    url_pattern = re.compile(r"https://(m|n)\.([a-z]+\.)?naver\.com")

    # Selenium
    def get_mobile_news_content(driver, link):
        try:
            driver.get(link)
            title = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "NewsEndMain_article_head_title__ztaL4"))
            ).text
            content = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "_article_content"))
            ).text
            return title, content

        except Exception as e:
            print(f"Error loading mobile page: {link}, {e}")
            return "Failed to load content"

    # BeautifulSoup
    def get_desktop_news_content(link):
        try:
            page = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
            if page.status_code != 200:
                print(f"Failed to fetch page: {link} (Status Code: {page.status_code})")
                return "Failed to load content"

            soup = BeautifulSoup(page.content, "html.parser")
            title = soup.find(class_="media_end_head_headline").text.strip() # type: ignore
            article_body = soup.find("div", class_="newsct_article _article_body")
            content = article_body.get_text(strip=True) if article_body else "No content available"
            return title, content

        except Exception as e:
            print(f"Error processing link: {link}, {e}")
            return "Failed to load content"

    for item in news_items:
        link = item.get('link', 'No Link')
        date = item.get('pubDate', 'No Date')

        if not url_pattern.match(link):
            continue

        if link.startswith("https://m."):
            title, content= get_mobile_news_content(driver, link)

        else:
            title, content= get_desktop_news_content(link)

        # 제목이 이미 리스트에 있으면 중복이므로 건너뛰기
        if title in titles:
            continue
        
        pattern = re.compile(r'(<.*?>|\(.*?기자.*?\)|\[.*?기자.*?\]|무단 전재.*?금지|ⓒ.*?\s|▶.*?\s|영상|\[출처.*?\]|포토|앵커|▲.*?\s)')
        cleaned_content = pattern.sub(' ', content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        
        contents.append(cleaned_content)
        titles.append(title)
        dates.append(date)

    driver.quit()

    return contents, titles, dates

# QA 프롬프트 생성 함수 -2
def create_chat_prompt2(system_prompt, user_query, context_documents):
    # 중첩 리스트 해제 (리스트 평탄화)
    if isinstance(context_documents[0], list):
        context_documents = [item for sublist in context_documents for item in sublist]

    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"""
            Context:
            {" ".join(context_documents)}

            📌 **지침**
            1. Context를 기반으로 사용자 질문에 정확하고 구체적으로 답변하세요.
            2. 핵심 요점을 먼저 제시한 후, 상세한 설명을 추가하세요.

            📌 **추가 뉴스 요청 처리**
            - "또 없어?", "더 보여줘" 등 추가 정보를 요청하면 이전 질문을 참고해 더 많은 정보를 제공합니다.

            📌 **Context에 관련 내용이 전혀 없을 시에만, 아래 형식을 따르세요:**
            - "관련 최신 뉴스가 없습니다." 문구 출력
            - 당신이 아는 정보를 기반으로 한 문단으로 답변하세요.
            
            📌 **출처 표기**
            출처가 되는 기사의 제목과, 본문 문장을 답변 마지막에 "출처:" 를 남겨주세요. 명확한 문장을 남기면 좋습니다. 
            - 여러 출처가 있는 경우 각각 명시하세요.

            Question:
            {user_query}

            Answer(출처 포함):

        """} ### 이전 질문 어떻게 참고할 것?
    ]

# 뉴스 요약 프롬프트
def create_headline_summary_prompt(headline_texts):
    return [
        {'role': 'system', 'content': "당신은 뛰어난 뉴스 요약 AI 에이전트입니다."},
        {'role': 'user', 'content': f"""
            아래의 뉴스 헤드라인과 내용을 간결하고 핵심적인 정보만 남기도록 요약하세요.
            각 뉴스는 핵심 사건, 주요 배경, 인과 관계를 포함해야 하며, 최대 3줄로 요약되어야 합니다.

            요약 형식:
            1. **뉴스 제목**  
               뉴스 내용 요약
            2. **뉴스 제목**  
               뉴스 내용 요약
            ...

            뉴스들:
            {headline_texts}

            요약:
        """}
    ]

# 헤드라인 뉴스 요약 생성
def generate_headline_summary(headline_texts):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=create_headline_summary_prompt(headline_texts), 
        max_tokens=1500
    )
    return response.choices[0].message.content

    
# OpenAI API를 통한 응답 생성 함수
def generate_response(system_prompt, user_query, context_documents):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=create_chat_prompt2(system_prompt, user_query, context_documents),
        max_tokens=1500
    )
    return response.choices[0].message.content


# RAG 기반 응답 생성 함수
def chat_with_rag(system_prompt, user_query):
    query_result = corpus_collection.query(query_texts=[user_query], n_results=6) # 5개의 chunk를 GPT에게 전달
    # 디버깅용
    # st.write(f"DB 크기 : {corpus_collection.count()}")
    
    # distance 임계값 설정
    distances = query_result['distances'][0] # type: ignore
    threshold = 1.3

    # 디버깅용
    # st.write(f"임계값 : {threshold}") # 디버깅용
    # st.write(f"{distances}") # 디버깅용

    relevant_docs = [d for d in distances if d <= threshold]

    # 관련 문서가 2개 미만이면 웹 검색 수행
    if len(relevant_docs) < 2:
        st.write("관련 최신 뉴스가 없습니다.")
        keywords = generate_keywords(user_query)
        # 디버깅용
        # st.write(f"🔍 **키워드:** {keywords}")  # 키워드 확인 - 디버깅용
        
        with st.spinner("웹에서 검색 중입니다..."):
            news_list, news_title, news_date = find_document(keywords, num_doc=20)

            # naver_news 내용 출력 - 디버깅용
            # if news_list:
            #     st.write(f"📊 **가져온 뉴스 개수:** {len(news_list)}")
            #     for idx, news in enumerate(news_list):
            #         st.write(f"**{idx + 1}.** {news[:100] if news else 'No content available'}...")
            # else:
            #     st.write("검색 결과를 찾을 수 없습니다.")

            chunks = smart_chunk_splitter(news_list, news_title, news_date, max_chunk_size=1500) 
            
            # 현재 저장된 뉴스 문서 가져오기
            existing_docs = set(corpus_collection.get()['documents']) # list -> set으로 변환 # type: ignore
            # 중복 없는 뉴스만 추가
            new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

            # 고유 ID 생성 함수 - 디버깅용
            def generate_unique_id():
                    return str(uuid.uuid4())

            # 새로운 뉴스 DB에 추가
            if new_chunks:
                try:
                    new_ids = [generate_unique_id() for _ in new_chunks]
                    corpus_collection.add(ids=new_ids, documents=new_chunks)

                    # 디버깅용
                    # st.success(f"✅ {len(new_chunks)}개의 새로운 청크가 추가되었습니다.") # 디버깅용
                    # st.write(f"📌 **DB 크기:** {corpus_collection.count()}")
                    
                except Exception as e:
                    st.error(f"❌ 뉴스 추가 중 오류 발생: {e}")
            else:
                st.info("🔍 추가할 새로운 뉴스가 없습니다.")

            # 사용자 쿼리와 DB 내 context를 비교하여 유사도 높은 상위 6개 추출
            query_result = corpus_collection.query(query_texts=[keywords], n_results=6)
        ###################################
    response = generate_response(system_prompt, user_query, query_result['documents'])
    
    # 디버깅용
    # st.write(query_result['documents']) # 디버깅용

    return response


# **XML에서 텍스트 추출 및 뉴스 데이터 추가**
def fetch_news_from_rss(rss_url, category_name="", summary = False):
    """RSS에서 뉴스 데이터를 가져와서 청크 리스트 반환"""
    try:
        response = requests.get(rss_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml-xml')

        items = soup.find_all('item')
        category_prefix = f"[오늘의 {category_name} 뉴스] " if category_name else ""
        
        # summary=True일 때만 texts 생성
        texts = []
        if summary:
            texts = [
                f"오늘의 헤드라인 뉴스 {i+1}: {item.title.text}\n\n{item.find('content:encoded').text if item.find('content:encoded') else item.description.text}"
                for i, item in enumerate(items)
            ]

        # 뉴스 데이터 청크 리스트
        chunks = [
            chunk
            for item in items
            for chunk in smart_chunk_splitter(
                item.find('content:encoded').text if item.find('content:encoded') else item.description.text,
                f"{category_prefix}, {item.title.text}", item.pubDate.text,
                max_chunk_size=1500
            )
        ]
        return (texts, chunks) if summary else chunks  # summary=True면 둘 다 반환, 아니면 chunks만 반환
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ 뉴스 불러오기 실패: {e}")
        return []


###################################

# **📌 Streamlit UI 설정**
st.set_page_config(page_title="NewSeans | AI 뉴스 챗봇", page_icon="📰")
st.markdown('<p style="font-size:20px; margin-bottom: 0;">NewSeans와 함께하는 오늘의 뉴스</p>', unsafe_allow_html=True)
st.subheader("뉴스 요약부터 분석까지! 궁금한 건 모두 나에게 물어봐 🤖", divider=True)


# **세션 스토리지 초기화 (채팅 기록 저장)**
if "messages" not in st.session_state:
    st.session_state.messages = []

# **사이드바: 최신 뉴스 표시**
st.sidebar.subheader("🔥 뉴스 헤드라인")
rss_url = "https://www.yonhapnewstv.co.kr/category/news/headline/feed/"
news_articles = fetch_news_titles(rss_url)

# 전체 최신 뉴스 표시
for title, link in news_articles:
    st.sidebar.markdown(f"[{title}]({link})")

# **카테고리별 뉴스 RSS 피드**
news_categories = {
    "최신": "http://www.yonhapnewstv.co.kr/browse/feed/",
    "정치": "http://www.yonhapnewstv.co.kr/category/news/politics/feed/",
    "경제": "http://www.yonhapnewstv.co.kr/category/news/economy/feed/",
    "사회": "http://www.yonhapnewstv.co.kr/category/news/society/feed/",
    "지역": "http://www.yonhapnewstv.co.kr/category/news/local/feed/",
    "세계": "http://www.yonhapnewstv.co.kr/category/news/international/feed/",
    "문화ㆍ연예": "http://www.yonhapnewstv.co.kr/category/news/culture/feed/",
    "스포츠": "http://www.yonhapnewstv.co.kr/category/news/sports/feed/"
}

# **카테고리 선택**
st.sidebar.subheader("🗂️ 뉴스 카테고리")

# 라디오 버튼을 사용하여 카테고리 선택 (기본값 없음)
selected_category = st.sidebar.radio("카테고리를 선택하세요:", list(news_categories.keys()), index=None)

# **선택된 카테고리의 뉴스 불러오기**
if selected_category:
    selected_url = news_categories[selected_category]
    category_articles = fetch_news_titles(selected_url)

    # 📌 카테고리 제목과 뉴스 3개를 바로 아래에 표시
    with st.sidebar.expander(f"📌 이시각 {selected_category} 뉴스", expanded=True):
        for title, link in category_articles[:3]:
            st.markdown(f"- [{title}]({link})")


# # **💬 이전 대화내용 표시**
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# ✅ 최초 페이지 로드 감지 (세션 유지)
if "page_loaded" not in st.session_state:
    st.session_state.page_loaded = False  # 페이지가 로드되었음을 나타냄
    st.session_state.corpus_collection = corpus_collection  # DB 유지

# ✅ 페이지가 새로고침되었을 때만 기존 뉴스 삭제 & 추가
if not st.session_state.page_loaded:
    existing_ids = st.session_state.corpus_collection.get()["ids"]
    if existing_ids:
        st.session_state.corpus_collection.delete(ids=existing_ids)
        # 디버깅용
        # st.write(f"🗑 기존 데이터 삭제 완료! (삭제된 문서 개수: {len(existing_ids)})")

    # ✅ 뉴스 데이터 추가 (페이지가 새로고침된 경우에만 실행)
    headline_texts, chunks = fetch_news_from_rss(rss_url, "헤드라인", summary=True)

    # **카테고리별 뉴스 데이터 추가**
    for category, category_rss in news_categories.items():
        category_chunks = fetch_news_from_rss(category_rss, category)
        chunks.extend(category_chunks)  # chunk 합치기

    # **중복 제거 (순서 유지 & 빠른 처리)**
    seen = set()
    new_chunks = [x for x in chunks if not (x in seen or seen.add(x))]

    # **ChromaDB에 새 뉴스 데이터 추가**
    ids = [str(uuid.uuid4()) for _ in new_chunks]
    corpus_collection.add(ids=ids, documents=new_chunks)
    
    # 디버깅용
    # st.write(f"DB 크기 : {corpus_collection.count()}")
    # st.write(f"✅ 새 뉴스 {len(new_chunks)}개 추가 완료!")

    # **현재 저장된 문서 개수 확인**
    existing_docs = corpus_collection.get()["documents"]
    
    # 디버깅용
    # st.write(f"📊 현재 저장된 뉴스 문서 개수: {len(existing_docs)}") # type: ignore

    st.session_state.page_loaded = True  # 이제 새로고침이 아님

#####################
# 뉴스 요약 생성
# 처음 들어갔을 때만 실행되도록 설정
# 헤드라인 뉴스 요약 출력

if "headline_summary" not in st.session_state:
    with st.spinner('AI가 헤드라인 뉴스를 요약 중입니다...'):
        st.session_state.headline_summary = generate_headline_summary(headline_texts)

if "headline_title" not in st.session_state:
    st.session_state.headline_title = "📰 헤드라인 뉴스 요약"

if "headline_subtitle" not in st.session_state:
    st.session_state.headline_subtitle = "오늘의 주요 헤드라인 뉴스는 아래와 같습니다:"

# 헤드라인 뉴스 요약 제목 출력
st.header(st.session_state.headline_title)

# 설명 문구 출력
st.write(st.session_state.headline_subtitle)

# 요약된 뉴스 출력
st.write(st.session_state.headline_summary)
#####################

# **💬 이전 대화내용 표시**
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# **사용자 입력 창**
user_query = st.chat_input("뉴스와 관련된 궁금한 점을 질문해 주세요! 🗣️")

if user_query:
    # **사용자 질문을 대화 히스토리에 저장**
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("⏳ AI가 답변을 생성 중입니다..."):
        # 시스템 프롬프트 설정
        system_prompt = """
        최신 뉴스 질문에 사실 기반으로 답변하는 어시스턴트입니다. 당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다.
        당신은 정확한 날짜 정보를 고려하여 질문에 답하는 AI입니다.

        오늘 날짜는 {today_kst} 입니다.
        아래 제공된 문맥(Context)에는 문서의 날짜 정보가 포함되어 있습니다.
        답변을 할 때 다음 원칙을 따르세요:
        1. **오늘(today)**, **내일(tomorrow)**, **어제(yesterday)** 같은 상대적 날짜 표현을 정확히 해석하세요.
        2. 문맥에 포함된 날짜(date)를 현재 날짜({today_kst})와 비교하여 답변하세요.
        3. 날짜 정보가 없을 경우 일반적인 정보만 제공합니다.
        """

        # RAG 기반 응답 생성
        response_text = chat_with_rag(system_prompt, user_query)

        # **AI 응답을 대화 히스토리에 저장**
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.write(response_text)