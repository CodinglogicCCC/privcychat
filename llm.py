import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import answer_examples

# .env 로드 (파일 경로 명시)
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 세션별 대화 히스토리 저장소
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID별 대화 히스토리 관리"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    """Pinecone 벡터 검색기 생성"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다.")

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    
    index_name = "privacychat"
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    return database.as_retriever(search_kwargs={"k": 5})

def get_llm(model: str = "gpt-4o") -> ChatOpenAI:
    """GPT-4o 모델을 설정"""
    return ChatOpenAI(model=model, streaming=True)

### `get_rag_chain()` 추가 (오류 해결)
def get_rag_chain() -> RunnableWithMessageHistory:
    """RAG 체인 생성"""
    llm = get_llm()
    retriever = get_retriever()

    system_prompt = (
        "당신은 서울과학기술대학교의 개인정보 보호 전문가입니다. "
        "검색된 문서를 반드시 활용하여 답변하세요. "
        "검색된 문서에서 가장 관련성이 높은 정보를 제공하세요. "
        "출처는 '(출처: 문서명: [문서명], 페이지: [페이지])' 형식으로 포함하세요. "
        "문서 내용과 관련 없는 정보를 생성하지 마세요. "
        "만약 검색된 문서가 부족하면 '죄송합니다. 현재 해당 질문과 관련된 정보를 찾을 수 없습니다. 다른 방식으로 질문해 보시겠어요?'라고 답변하세요.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer",
    )

### 질문 정규화 개선
def normalize_query(user_message: str) -> str:
    """구어체 및 다양한 표현을 표준화하여 변환"""
    replacements = {
        "서울과기대": "서울과학기술대학교",
        "과기대": "서울과학기술대학교",
        "학교": "서울과학기술대학교",
        "보유 기간": "보존 기간",
        "저장 기간": "보존 기간",
        "개인정보": "개인정보 수집 항목",
        "내 정보": "개인정보",
        "학교에서 내 개인정보": "서울과학기술대학교 개인정보",
        "졸업하면 내 개인정보": "졸업 후 개인정보",
        "졸업하고 나면 개인정보": "졸업 후 개인정보",
        "학교 졸업하면 내 개인정보": "졸업 후 개인정보",
        "학교 졸업 후 내 정보": "졸업 후 개인정보",
        "학교 졸업 후 내 개인정보는": "졸업 후 개인정보",
        "졸업하면 정보 삭제?": "졸업 후 개인정보 처리",
        "졸업하면 개인 정보는 어떻게 돼?": "졸업 후 개인정보 처리",
        "졸업 후 내 정보 없어지나요?": "졸업 후 개인정보 삭제 여부"
    }
    
    for key, value in replacements.items():
        user_message = user_message.replace(key, value)

    return user_message

### 질질문 유형 분류
def classify_query(user_message: str) -> str:
    """질문을 유형별로 분류"""
    llm = get_llm()
    classification_prompt = f"""
    사용자의 질문을 다음 카테고리 중 하나로 분류하세요: 
    1. 개인정보 수집 항목
    2. 개인정보 보존 기간
    3. 개인정보 운영 목적
    4. 기타

    질문: "{user_message}"
    카테고리 번호만 출력하세요.
    """

    response = llm.invoke(classification_prompt)

    # 🔍 타입 및 응답 직접 확인
    print("응답 타입:", type(response))
    print("응답 내용:", response)

    return "4"  # 일단 기본값 반환해서 에러 피하기

### 검색 실패 시 유사 질문 추천
def find_similar_questions(query, retriever, top_n=3):
    """입력된 질문과 가장 유사한 문서를 찾음"""
    docs = retriever.invoke(query)
    best_docs = sorted(docs, key=lambda d: fuzz.ratio(query, d.page_content), reverse=True)[:top_n]
    return best_docs

### 메인 함수 (오류 수정)
from config import answer_examples

def get_ai_response(user_message: str, session_id: str) -> str:
    """질문의 의도를 분석한 후 적절한 검색 방식을 선택"""
    try:
        # (1) 질문 정규화
        normalized_query = normalize_query(user_message)

        # (2) 질문 유형 분류
        category = classify_query(normalized_query)

        # (3) 문서 검색
        retriever = get_retriever()
        retrieved_docs = retriever.invoke(normalized_query)

        # (4) 검색 결과가 없으면 Few-shot 데이터에서 유사 질문 찾기
        if not retrieved_docs:
            best_match = None
            best_score = 0

            for example in answer_examples:
                similarity = fuzz.ratio(normalized_query, example["input"])
                if similarity > best_score and similarity > 80:  # 80% 이상 유사한 질문 찾기
                    best_score = similarity
                    best_match = example["answer"]

            if best_match:
                return best_match  # 유사한 질문이 있으면 해당 답변 제공

            # (5) 그래도 찾을 수 없으면 유사 질문 추천
            similar_docs = find_similar_questions(normalized_query, retriever)
            if similar_docs:
                suggestions = "\n".join([f"- {doc.page_content[:50]}..." for doc in similar_docs])
                return f"❓ 찾으시는 정보가 없어요. 혹시 이런 질문을 하시려 했나요?\n\n{suggestions}"
            return "죄송합니다. 현재 해당 질문과 관련된 정보를 찾을 수 없습니다."

        # (6) 답변 생성
        rag_chain = get_rag_chain()
        ai_response = rag_chain.invoke(
            {"input": normalized_query},
            config={"configurable": {"session_id": session_id}},
        )

        # 안전하게 응답 내용 추출
        if isinstance(ai_response, dict):
            answer = ai_response.get("answer", "")
        elif hasattr(ai_response, "content"):
            answer = ai_response.content
        else:
            answer = str(ai_response)

        return answer.strip()

    except Exception as e:
        logging.error(f"❌ AI 응답 생성 중 오류 발생: {e}")
        return "오류가 발생했습니다. 다시 시도해 주세요."
