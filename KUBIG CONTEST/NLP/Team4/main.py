import os
import pickle
from class_extraction import prompt_llm_extraction
from keyword_extraction import extract_medical_keywords
from reply_candidate import custom_search, custom_search_only_query, custom_search_all
from new_reply_candidate import custom_search_just_query

# 사용자 입력 받기
user_input = input("질문을 입력하세요: ")

# Method 1: Query Keyword Embedding with Class Extraction
extracted_class = prompt_llm_extraction(user_input)
extracted_keyword = extract_medical_keywords(user_input)

query_disease_name = extracted_class['disease_name']
query_department = extracted_class['disease_category']
query_intention = extracted_class['intention']

mtd1_top_answers = custom_search(query_disease_name, query_department, query_intention, extracted_keyword, top_k=3)

# Method 2: Query Keyword Embedding w/o Class Extraction
mtd2_top_answers = custom_search_only_query(query_disease_name, query_department, query_intention, extracted_keyword, top_k=3)

# Method 3: Whole Query Sentence Embedding with Class Extraction
mtd3_top_answers = custom_search(query_disease_name, query_department, query_intention, user_input, top_k=3)

# Method 4: Whole Query Sentence Embedding w/o Class Extraction
mtd4_top_answers = custom_search_only_query(query_disease_name, query_department, query_intention, user_input, top_k=3)

# Method 5: Whole Query Sentence & Keyword Embedding with Class Extraction
mtd5_top_answers = custom_search_all(query_disease_name, query_department, query_intention, user_input, extracted_keyword, top_k=3)


'''
# 결과 출력
print(
    f"Method 1 Answers: {mtd1_top_answers}\n\n",
    f"Method 2 Answers: {mtd2_top_answers}\n\n",
    f"Method 3 Answers: {mtd3_top_answers}\n\n",
    f"Method 4 Answers: {mtd4_top_answers}\n\n",
    f"Method 5 Answers: {mtd5_top_answers}\n\n"
)
'''