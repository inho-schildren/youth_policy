from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import time
from dotenv import load_dotenv
from config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY

load_dotenv()

BATCH_SIZE = 100

embedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
)

def embed_and_save(chunks):
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"  🔄 임베딩 중... [{i+1}/{len(chunks)}]")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding,
                persist_directory=CHROMA_DIR
            )
        else:
            vectorstore.add_documents(batch)
        time.sleep(0.5)
    print(f"✅ 임베딩 완료 → {CHROMA_DIR} ({len(chunks)}개 청크)")
    return vectorstore

def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )