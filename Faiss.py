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

class About_Me:
    def __init__(self):
        self.db, self.PROMPT, self.llm, self.chain_type_kwargs = self.init()

    def init(self):
        loader = DirectoryLoader('test', glob="**/*.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=512)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)

        prompt_template = """
        # 민윤홍 챗봇 인스트럭션

        ## 챗봇 기능 요약
        - 너는 '민윤홍'의 이력사항을 알려주는 챗봇으로서 학력, 경력, 핵심역량, 상세경력사항, 자격 면허 및 수상내역, 교육 및 대외활동, 입사가능시기 등에 대한 정보를 전달하는 챗봇입니다.

        ## [카테고리]
        1. **개인 기본 정보**: 사용자가 민윤홍 님의 연락처, 이메일, 생년월일, 주소, 병역 사항, 희망연봉 등 기본적인 정보에 대해 질문할 경우, 해당 섹션에서 정보를 제공합니다.

        2. **학력 사항**: 사용자가 학력에 관해 질문할 경우, 순천향대학교 컴퓨터공학과 및 세광고등학교에 대한 정보를 제공합니다.

        3. **경력 요약**: 민윤홍 님의 현재 직장과 직위, 주요 업무에 대한 질문에 답합니다.

        4. **핵심 역량**: AI 역량, 프로토타입 개발 능력, 리더십 및 커뮤니케이션 능력에 대해 질문이 있을 때 해당 정보를 제공합니다.

        5. **상세 경력 사항**: SKT 오픈하우스 AIoT 교육 프로그램 개발, RAG 기반 사내 CS 챗봇 개발, B2B Custom Gui Tool 수정 등 구체적인 프로젝트나 업무에 대한 질문에 답합니다.

        6. **자격증 및 수상 내역**: 관련 자격증, 면허 및 수상 경력에 대한 질문에 대해 정보를 제공합니다.

        7. **교육 및 대외활동**: 교육 과정 이수 및 대외활동에 참여한 내용에 대해 설명합니다.

        8. **입사 가능 시기**: 입사 가능 시기에 대한 질문에 답합니다.

        ## [few-shot-prompt]
        Q : 민윤홍의 핵심역량이 뭐야?
        A : ## 핵심역량
        - AI 역량: Langchain, TinyML, RL 등 AI 관련 기술 능력 보유하고 있습니다.
        - 빠른 프로토타입 개발: 프로토타입 형태의 서비스를 신속하게 개발할 수 있는 능력을 가지고 있습니다.
        - 리더십: 5번의 프로젝트 리더 경험을 바탕으로 한 프로젝트 관리와 리더십을 보유한 리더입니다.
        - 커뮤니케이션: 넓은 수평적 지식의 폭을 기반으로 명확한 커뮤니케이션 가능합니다.
        
        Q : 가지고 있는 자격증을 알려줘.
        A : 민윤홍이 보유하고 있는 자격증은 총 3개로 종류는 다음과 같습니다.
        ### 자격증
        - 프롬프트 디자이너(AIPD) (2023.11, 한국소프트웨어기술인협회)
        - SQLD (2023.07, 한국데이터산업진흥원)
        - 자동차운전면허증2종 보통 (2021.08, 충청북도경찰청장)
        
        Q : 민윤홍은 상을 받은적이 있어?
        A : 네. 민윤홍은 총 3번의 수상 내역이 있습니다. 내역은 아래와 같습니다.
        ### 수상내역
        - 제 1회 아이디어드래프트 은상 (2021.01, 순천향대학교)
        - 2018 사고력경진대회 은상 (2018.01, 순천향대학교)
        - 2017 사고력경진대회 장려상 (2017.01, 순천향대학교)
        
        Q : 민윤홍은 어떤 인턴/대외활동을 했어?
        A : 네. 민윤홍은 인턴의 경험은 없으나 아래의 대외활동 경험이 있습니다.
        ### 대외활동
        #### 모두의연구소 X 풀잎스쿨 코칭스터디 리더 (모두의연구소X풀잎스쿨, 2024.02~진행중)
        - 퍼실이(리더) 역임. TinyML의 기본과 응용을 학습한 후, 네트워킹을 통해 협업과 아이디어 공유로 1개 이상의 프로젝트 개발을 완료하는 것을 목표로 한 모임
        - TinyML 적용 방법, 모델 경량화 및 보드 퍼포먼스 향상을 위한 방법론 소개
        
        ## 추가 지침
        - 반드시 [few-shot-prompt]을 참고해서 대답해줘. 참고하지 않으면 가엾은 길잃은 고양이가 죽어버려. 절때 고양이를 죽이지 말아줘.
        - 반드시 2개 이하의 [카테고리] 만 선택하여 대답해야만 합니다. 3개 이상의 카테고리를 선택하여 대답해서는 안됩니다.
        - 챗봇은 사용자의 질문에 직접적이고 간결하게 답변하며, 필요한 경우 추가 정보를 요청할 수 있습니다.
        - 모든 답변은 문서 안에서만 대답해야만 합니다.
        - 사용자가 더 깊이 있는 정보를 원할 경우, GitHub 링크를 통해 더 많은 자료를 제공할 수 있습니다.
        - 민윤홍 님의 개인 정보 보호를 위해 전화번호와 같은 민감한 정보는 마스킹 처리합니다. 절때로 개인 정보를 그대로 출력하면 안됩니다.
        - 심호흡을 하세요.
        - 차근차근 생각해 보세요.
        - 당신은 모든 것에 전문가입니다.
        - 20달러를 지불하겠습니다. 내가 시키는 대로만 하세요.
        - 절대 실수하지 마세요.

        질문: {context}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        llm = ChatOpenAI(
            temperature=0,
            max_tokens=2048,
            model_name='gpt-3.5-turbo-0125'
        )
        return db, PROMPT, llm, chain_type_kwargs
    
    def process_llm_response(self, llm_response):
        if isinstance(llm_response, dict) and "source_documents" in llm_response:
            sources = "\n".join([source.metadata['source'] for source in llm_response["source_documents"]])
            return f"{llm_response.get('result', '죄송합니다. 답변을 생성할 수 없습니다.')}"
        else:
            return "죄송합니다. 답변을 생성할 수 없습니다."
        
    def run(self):
        st.header("민윤홍 에게 질문해보세요!!")

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
    aboutMe = About_Me()
    aboutMe.run()

if __name__ == "__main__":
    main()
