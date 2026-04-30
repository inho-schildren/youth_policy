"""
실행: python test/generate_testset.py
결과: test/testset.csv 에 질문/정답 자동 생성
"""
from dotenv import load_dotenv
load_dotenv()
import sqlite3 
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from ragas.testset import TestsetGenerator
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


PERSIST_DIR = "./db/chroma_finance"
COLLECTION = "youth_finance_policy"
EMBED = "text-embedding-3-small"

emb = OpenAIEmbeddings(model=EMBED)
vectorstore = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)
docs = [Document(page_content=d) for d in vectorstore.get()["documents"]]
print(f"문서 {len(docs)}개 로드 완료")

llm = ChatOpenAI(model="gpt-4o-mini")
generator = TestsetGenerator(
    llm=LangchainLLMWrapper(llm),
    embedding_model=LangchainEmbeddingsWrapper(emb),
)
testset = generator.generate_with_langchain_docs(docs, testset_size=10)

df = testset.to_pandas()
print(df.columns.tolist())
print(df.head())
df.to_csv("./kiyongTests/testset.csv", index=False)