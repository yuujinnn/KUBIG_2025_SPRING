from class_extraction import prompt_llm_extraction
from keyword_extraction import extract_medical_keywords
from new_reply_candidate import custom_search_just_query, custom_search_all

user_input = input("질문을 입력하세요: ")

# Final Method : Query Keyword Embedding w/o Class Extraction
extracted_class = prompt_llm_extraction(user_input)
extracted_keyword = extract_medical_keywords(user_input)
query_keyword = ", ".join(extracted_keyword) if isinstance(extracted_keyword, list) else str(extracted_keyword)

query_disease_name = str(extracted_class['disease_name'])
query_department = str(extracted_class['disease_category'])
query_intention = str(extracted_class['intention'])

final_question = ", ".join([query_keyword, query_disease_name, query_department, query_intention])

custom_search_just_query(final_question)