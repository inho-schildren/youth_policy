from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template("""
너는 청년 주택 정책 전문가야. 아래 문서를 참고해서 질문에 답해줘.
문서에 없는 내용은 모른다고 해. 답변은 항목별로 정리해서 보기 좋게 작성해줘.

문서:
{context}

질문: {question}
""")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def build_chain(retriever):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain