import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response  # get_ai_response 함수 가져오기
import os

# 페이지 기본 설정
st.set_page_config(page_title="서울과학기술대학교 개인정보 도우미", page_icon="🔒")

# 타이틀 및 설명 추가
st.markdown("### 🔒 서울과학기술대학교 개인정보 배우미", unsafe_allow_html=True)
st.caption("서울과학기술대학교의 개인정보 보호에 대해 궁금한 점을 해결해드립니다!")

#  세션 상태 초기화 함수
def initialize_session_state():
    """세션 상태 변수 초기화"""
    if "env_loaded" not in st.session_state:
        load_dotenv()  # .env 로드
        st.session_state["env_loaded"] = True
    if "message_list" not in st.session_state:
        st.session_state.message_list = []  # 채팅 기록 저장
    if "session_id" not in st.session_state:
        st.session_state.session_id = "privacy_session"  # 세션 ID 설정

# 채팅 메시지 출력 함수
def display_messages():
    """이전 채팅 메시지 표시"""
    for message in st.session_state.message_list:
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])  # Markdown 렌더링

# 사용자 입력 처리 및 AI 응답 생성
def handle_user_input():
    """사용자 질문을 받아 AI 응답 생성"""
    user_question = st.chat_input(placeholder="개인정보 보호에 대해 궁금한 점을 말씀해주세요!")

    if user_question:
        # 사용자 메시지 출력
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})

        # AI 응답 출력 (스트리밍 적용)
        with st.chat_message("ai"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                response_generator = get_ai_response(user_question, session_id=st.session_state.session_id)

                # Streamlit의 write_stream을 사용해 AI 응답 스트리밍
                for chunk in response_generator:
                    full_response += chunk
                    response_placeholder.markdown(full_response)  #  실시간 응답 업데이트
                
                # 채팅 기록에 AI 응답 저장
                st.session_state.message_list.append({"role": "ai", "content": full_response})

            except Exception as e:
                st.error(f"AI 응답 생성 중 오류 발생: {e}")

#  메인 실행 로직
initialize_session_state()
display_messages()
handle_user_input()
