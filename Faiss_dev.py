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

    async def async_conversation(self, topic):
        retriever = self.db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=self.chain_type_kwargs,
        )
        result = qa_chain({"query": topic})
        result = self.process_llm_response(result)

        answer = result
        wrapper = textwrap.TextWrapper(width=53)
        wrapped_text = wrapper.fill(text=answer)
        self.placeholder.code(wrapped_text)

    def process_llm_response(self, llm_response):
        if isinstance(llm_response, dict) and "source_documents" in llm_response:
            sources = "\n".join([source.metadata['source'] for source in llm_response["source_documents"]])
            return f"{llm_response.get('result', '죄송합니다. 답변을 생성할 수 없습니다.')}\n\n관련 소스:\n{sources}"
        else:
            return "죄송합니다. 답변을 생성할 수 없습니다."

    def start_conversation(self, topic):
        asyncio.run(self.async_conversation(topic))

def main():
    st.title("비동기 대화 예제")
    conversation_manager = AsyncConversationManager()

    # 사용자가 텍스트 입력 필드에 토픽을 입력하고 엔터를 누를 때 실행됩니다.
    def on_topic_change():
        topic = st.session_state.get('topic', '')  # session_state를 통해 토픽을 가져옵니다.
        if topic:  # 토픽이 비어있지 않은 경우에만 실행
            conversation_manager.start_conversation(topic)

    # 텍스트 입력 필드에 키를 할당하고, on_change 이벤트를 정의합니다.
    st.text_input("토픽을 입력하세요", key="topic", on_change=on_topic_change)

if __name__ == "__main__":
    main()
