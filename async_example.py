import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import asyncio
import textwrap

# 비동기로 대화 출력하는 함수를 정의합니다.
async def async_conversation(chain, topic):
    placeholder = st.empty()  # 출력을 위한 새로운 플레이스홀더를 생성합니다.
    answer = ""
    wrapper = textwrap.TextWrapper(width=53)  # 80글자 너비로 개행 설정
    async for chunk in chain.astream({"topic": topic}):
        answer += chunk
        wrapped_text = wrapper.fill(text=answer)  # 개행된 텍스트를 얻습니다.
        placeholder.code(wrapped_text)  # 비동기로 메시지를 업데이트합니다.
        await asyncio.sleep(0.01)  # 메시지 간에 약간의 딜레이를 줍니다.

# Streamlit 애플리케이션의 메인 함수입니다.
def main():
    st.title("비동기 대화 예제")

    # 예제에서 사용된 체인을 정의합니다.
    model = ChatOpenAI(
                temperature=0,
                max_tokens=2048,
                model_name='gpt-3.5-turbo'
            )
    prompt = ChatPromptTemplate.from_template("{topic}에 대해 100자 내외의 에세이를 작성해줘")
    parser = StrOutputParser()
    chain = prompt | model | parser

    # 사용자로부터 토픽을 입력받습니다.
    topic = st.text_input("토픽을 입력하세요")


    if st.button("대화 시작"):
        # 비동기 작업을 실행합니다.
        asyncio.run(async_conversation(chain, topic))

class AsyncConversationManager:
    def __init__(self):
        # LLM과 RAG 초기화
        self.llm = ChatOpenAI(
            temperature=0,
            max_tokens=2048,
            model_name='gpt-3.5-turbo'
        )
        self.db, self.PROMPT, self.chain_type_kwargs = self.init_rag()
        
        self.placeholder = st.empty()
    def init_rag(self):
        loader = DirectoryLoader('test', glob="**/*.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)

        prompt_template = """
        Question : {context}
        Answer :
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        return db, PROMPT, chain_type_kwargs
if __name__ == "__main__":
    main()