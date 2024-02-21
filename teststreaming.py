import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
import time

# LangChain 설정
model = ChatOpenAI(
    temperature=0,
    max_tokens=2048,
    model_name='gpt-3.5-turbo'
)
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

async def fetch_joke(topic):
    response_text = ""
    async for chunk in chain.astream({"topic": topic}):
        response_text += chunk
        # 비동기 작업 완료 후 응답을 st.session_state.messages에 추가
        st.session_state.messages.append(chunk)
        time.sleep(0.5)

def main():
    st.header("농담 생성기")

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    topic = st.text_input("주제를 입력하세요:", key="topic")

    if st.button("농담 생성하기"):
        # 기존 메시지 초기화
        st.session_state['messages'] = []
        # 비동기 작업 실행
        asyncio.run(fetch_joke(topic))

    for message in st.session_state['messages']:
        st.write(message)

if __name__ == "__main__":
    main()
