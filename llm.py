import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from fuzzywuzzy import fuzz
from langchain_core.prompts import ChatPromptTemplate
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
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다.")

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    index_name = "privacychat"
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    return database.as_retriever(search_kwargs={"k": 15})

def get_llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model, streaming=True)

def get_rag_chain() -> RunnableWithMessageHistory:
    llm = get_llm()
    retriever = get_retriever()

    system_prompt = (
        "당신은 서울과학기술대학교의 개인정보 보호 전문가입니다. "
        "검색된 문서를 반드시 활용하여 답변하세요. "
        "검색된 문서에서 가장 관련성이 높은 정보를 제공하세요. "
        "가능하다면 제공 항목, 제공 대상 기관, 보유 기간, 법적 근거 등을 구체적으로 포함하세요. "
        "출처는 '(출처: 문서명: [문서명], 페이지: [페이지])' 형식으로 포함하세요. "
        "문서 내용과 관련 없는 정보를 생성하지 마세요. "
        "문서에서 직접적으로 명시되지 않았더라도, 유사한 항목이나 일반적인 원칙이 있다면 활용하여 답변하세요. "
        "정보가 부족하더라도 질문자의 이해를 돕기 위해 최소한의 원칙이나 유사 문맥을 근거로 설명하세요. "
        "단, 전혀 근거를 찾을 수 없는 경우에만 '죄송합니다. 현재 해당 질문과 관련된 정보를 찾을 수 없습니다. 다른 방식으로 질문해 보시겠어요?'라고 답변하세요."
        "\n\n{context}"
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

def get_ai_response(user_message: str, session_id: str) -> str:
    try:
        normalized_query = user_message.strip()
        retriever = get_retriever()
        retrieved_docs = retriever.invoke(normalized_query)

        print("🔍 [검색된 청크 미리보기]")
        if retrieved_docs:
            for doc in retrieved_docs:
                print("-", doc.page_content[:200].replace("\n", " "), "...")

        rag_chain = get_rag_chain()
        ai_response = rag_chain.invoke(
            {"input": normalized_query},
            config={"configurable": {"session_id": session_id}},
        )

        if isinstance(ai_response, dict):
            answer = ai_response.get("answer", "")
        elif hasattr(ai_response, "content"):
            answer = ai_response.content
        else:
            answer = str(ai_response)

        answer = answer.strip()

        fallback_keywords = [
            "죄송합니다. 현재 해당 질문과 관련된 정보를 찾을 수 없습니다.",
            "다른 방식으로 질문해 보시겠어요?",
            "관련 정보를 찾지 못했습니다."
        ]
        for phrase in fallback_keywords:
            if phrase in answer and len(answer.split(phrase)[0].strip()) > 50:
                answer = answer.split(phrase)[0].strip()
                break

        return answer

    except Exception as e:
        logging.error(f"❌ AI 응답 생성 중 오류 발생: {e}")
        return f"AI 응답 생성 중 오류 발생: {e}"
