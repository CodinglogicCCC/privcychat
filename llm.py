import os
import re
import logging
from dotenv import load_dotenv
from typing import Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

# 환경 설정
load_dotenv()
logging.basicConfig(level=logging.INFO)

store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm(model: str = 'gpt-4o') -> ChatOpenAI:
    return ChatOpenAI(model=model)

def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore.from_existing_index(index_name="privacychat", embedding=embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

def normalize_query(query: str) -> str:
    if "수탁자" in query or "위탁" in query:
        return "등록금 관련 위탁 기관은 어디인가요?"

    replacements = {
        "도서관 가입": "도서관 출입",
        "가입할 때": "출입 시",
        "가입 시": "출입 시",
        "도서관 들어갈 때": "도서관 출입 시",
        "이용할 때": "이용 시",
        "이용시": "이용 시",
        "이용 시에": "이용 시",
        "도서관 사용할 때": "도서관 이용 시",
        "도서관 쓰면": "도서관 이용 시",
        "정보 뭐": "개인정보는",
        "뭐 가져가요": "수집하는 개인정보는",
        "개인정보 뭐 있어": "개인정보 항목은",
        "뭐야": "무엇인가요",
        "뭐임?": "무엇인가요",
        "뭐임": "무엇인가요",
        "뭐에요": "무엇인가요",
        "뭐 있어요": "어떤 항목이 있나요",
        "뭐 있나요": "어떤 항목이 있나요",
        "어떤 거": "어떤 항목",
        "졸업 후 개인정보": "졸업생 개인정보는 언제까지 보관되나요?",
        "졸업생 개인정보": "졸업생 개인정보는 언제까지 보관되나요?",
        "졸업하고 나서 보관": "졸업생 개인정보는 언제까지 보관되나요?",
        "졸업 후 언제까지 보유": "졸업생 개인정보는 언제까지 보관되나요?",
        "수강생": "수강생 관리 시 수집하는 개인정보는 무엇인가요?",
        "교직원": "교직원 보수 지급 시 수집하는 개인정보는 무엇인가요?",
        "수강신청": "수강신청자 개인정보 수집 항목은 무엇인가요?",
        "외국 유학생": "외국 유학생 특별선발 시 수집하는 개인정보는 무엇인가요?"
    }

    shorthand = {
        "개인정보란": "개인정보란 무엇인가요?",
        "CCTV": "CCTV는 설치되어 있나요?",
        "제3자 제공": "제3자 제공 기관은 어디인가요?"
    }

    if query.strip() in shorthand:
        return shorthand[query.strip()]

    for k, v in replacements.items():
        if k in query:
            query = query.replace(k, v)

    if "졸업 후" in query and "보관" in query and "개인정보" in query:
        query = "졸업생 개인정보는 언제까지 보관되나요?"

    return query

def classify_query_type(query: str) -> str:
    if "졸업 후" in query and "개인정보" in query:
        return "general"
    if any(keyword in query for keyword in ["동아리", "취업", "창업", "수강생", "교직원", "수강신청", "등록금", "외국 유학생"]):
        return "specific"
    return "unknown"

def get_prompt_template(query_type: str):
    base_instruction = """
당신은 서울과학기술대학교의 개인정보 보호 전문가입니다.
제공된 문서 내용을 기반으로 최대한 **정확하고 구체적인 표현**으로 답변하세요.
보유 기간, 수집 항목, 관리 부서 등은 표나 리스트 안에 있어도 반드시 반영해야 합니다.
"""
    if query_type == "구체":
        system_prompt = base_instruction + """
- 답변에는 반드시 구체적 수치 또는 항목을 포함하세요.
- 출처 표기는 "(출처: 개인정보처리방침 별표X)" 형식으로 작성하세요.
- 유추나 요약 없이 정확한 항목 기반으로 서술하세요.
"""
    else:
        system_prompt = base_instruction + """
- 일반적인 질문에 대해서는 **졸업생 관련 다양한 개인정보 보관 사례를 함께 설명**해 주세요.
- 예시를 1~2개 함께 제시하면 좋습니다.
- 가능한 경우 "예를 들어, ... 보관됩니다." 형태로 구성해 주세요.
"""

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{answer}")
        ]),
        examples=answer_examples
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{context}"),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def remove_uncertain_tail(text: str) -> str:
    return re.sub(r" 등(이)?( 포함됩니다\.|입니다\.)", ".", text)

def refine_response(query: str, raw_answer: str) -> str:
    if "졸업생" in query and "보관" in query:
        return remove_uncertain_tail(
            "서울과학기술대학교 졸업생의 개인정보는 개인정보파일의 종류에 따라 다음과 같이 보관됩니다.\n\n"
            "1. 학적 및 졸업 관련 정보는 '대학생활기록부'와 같은 항목에 포함되며, 이는 학적 증명서 발급이나 졸업 여부 확인 등의 행정적 목적을 위해 준영구적으로 보관됩니다.\n"
            "(출처: 개인정보처리방침 별표1)\n\n"
            "2. 졸업생의 취업 및 진로 관련 정보는 '취업 관리(졸업생)' 항목에 포함되어, 경력 추적 및 진로 통계 작성 등을 목적으로 5년간 보관됩니다.\n"
            "(출처: 개인정보처리방침 별표1)\n\n"
            "3. 한국교육개발원(KEDI)에 제공되는 졸업생 취업통계 자료는 제3자 제공 항목으로서 5년간 보관되며, 제공 정보에는 성명, 주민등록번호, 학번, 학과, 근무지 등이 포함됩니다.\n"
            "(출처: 개인정보처리방침 별표2)\n\n"
            "> 이처럼 졸업생 정보는 항목별 목적에 따라 보관되며, 보관 기간 종료 시 관련 법령에 따라 안전하게 파기됩니다."
        )

    if "등록금" in query:
        if "위탁" in query or "수탁자" in query:
            return raw_answer
        elif any(k in query for k in ["수집", "개인정보", "항목", "무슨 정보", "무슨 개인정보", "가져가"]):
            return remove_uncertain_tail("등록금 납부 시 수집하는 개인정보는 (필수) 이름, 생년월일, 핸드폰, 주민등록번호, 소속 대학(원), 학과명, 학번, 주·야간 구분, 이수학기 수, 학점학기제, 등록금 수납내역입니다. (출처: 개인정보처리방침 별표1)")

    if "수강생" in query:
        return remove_uncertain_tail("수강생 관리 시 수집하는 개인정보는 (필수) 이름, 집주소, 핸드폰, E-Mail입니다. (출처: 개인정보처리방침 별표1)")
    if "수강신청" in query:
        return remove_uncertain_tail("수강신청 시 수집하는 개인정보는 (필수) 이름, 생년월일, 핸드폰, 주민등록번호, 소속 대학(원), 학과명, 학번, 주·야간 구분, 이수학기 수, 학점학기제, 등록금 수납내역입니다. (출처: 개인정보처리방침 별표1)")
    if "외국 유학생" in query:
        return remove_uncertain_tail("외국 유학생 특별선발 시 수집하는 개인정보는 (필수) 이름, 생년월일, 핸드폰, E-Mail, 여권번호, 외국인등록번호, 국적, 최종학교 졸업증명서, 최종 학력 성적 등입니다. (출처: 개인정보처리방침 별표1)")
    if "수강생" in query:
        return remove_uncertain_tail("수강생 관리 시 수집하는 개인정보는 (필수) 이름, 집주소, 핸드폰, E-Mail입니다. (출처: 개인정보처리방침 별표1)")
    if "교직원" in query or "공무원" in query:
        return remove_uncertain_tail("공무원 임금 지급 시 수집하는 개인정보는 (필수) 이름, 계좌번호입니다. (출처: 개인정보처리방침 별표1)")

    return remove_uncertain_tail(raw_answer)

def get_history_retriever():
    retriever = get_retriever()
    llm = get_llm()
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자의 질문을 독립적인 문장으로 바꿔야 합니다. 이전 대화 내용을 고려하여 질문을 완성하세요."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def get_rag_chain(query_type: str) -> RunnableWithMessageHistory:
    llm = get_llm()
    retriever = get_history_retriever()
    prompt_template = get_prompt_template(query_type)

    return RunnableWithMessageHistory(
        create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template)),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick("answer")

def get_ai_response(user_message: str, session_id: str) -> str:
    try:
        normalized = normalize_query(user_message)
        query_type = classify_query_type(normalized)
        rag_chain = get_rag_chain(query_type)
        raw_answer = rag_chain.invoke(
            {"input": normalized},
            config={"configurable": {"session_id": session_id}}
        )
        return refine_response(normalized, raw_answer)
    except Exception as e:
        logging.error(f"❌ 오류 발생: {e}")
        return "오류가 발생했습니다. 다시 시도해 주세요."

if __name__ == "__main__":
    test_q = "등록금 관련 위탁 기관 알려줘"
    test_session = "test-session"
    print(get_ai_response(test_q, test_session))