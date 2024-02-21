import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from st_chat_message import message
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import RetrievalQA

class WizBot:
    def __init__(self):
        self.db, self.PROMPT, self.llm, self.chain_type_kwargs = self.init()

    def init(self):
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

        llm = ChatOpenAI(
            temperature=0,
            max_tokens=2048,
            model_name='gpt-3.5-turbo'
        )
        return db, PROMPT, llm, chain_type_kwargs
    
    def process_llm_response(self, llm_response):
        if isinstance(llm_response, dict) and "source_documents" in llm_response:
            sources = "\n".join([source.metadata['source'] for source in llm_response["source_documents"]])
            return f"{llm_response.get('result', '죄송합니다. 답변을 생성할 수 없습니다.')}\n\n관련 소스:\n{sources}"
        else:
            return "죄송합니다. 답변을 생성할 수 없습니다."
        
    def run(self):
        st.header("WizBot 에게 질문해보세요!!")

        if "messages" not in st.session_state:
            st.session_state['messages'] = [AIMessage(content="뭐 물어보고싶어?")]

        user_input = st.text_input("질문을 입력하세요: ", key="user_input")

        if user_input:
            st.session_state['messages'].append(HumanMessage(content=user_input))
            
            with st.spinner("답변 생성 중..."):
                retriever = self.db.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs=self.chain_type_kwargs,
                )
                result = qa_chain({"query": user_input})
                result = self.process_llm_response(result)
            
            st.session_state.messages.append(AIMessage(content=result))

        for i, msg in enumerate(st.session_state.messages):
            if isinstance(msg, HumanMessage):
                message(msg.content, is_user=True, key=f"message_{i}_user")
            elif isinstance(msg, AIMessage):
                message(msg.content, is_user=False, key=f"message_{i}_ai")

def main():
    wiz_bot = WizBot()
    wiz_bot.run()

if __name__ == "__main__":
    main()
