from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os
import time
from dotenv import load_dotenv
from config import (
    EMBEDDING_MODEL, OPENAI_API_KEY, 
    CHROMA_DIR, FINANCE_CHROMA_DIR,
    FAISS_DIR, FINANCE_FAISS_DIR)

load_dotenv()

BATCH_SIZE = 100

def get_openai_embedder_small():
    return OpenAIEmbeddings(
        model = EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

def get_openai_embedder_large():
    return OpenAIEmbeddings(
        model = "text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )

# CHROMA
def embed_and_save_chroma(chunks, embedder,
    persist_dir=CHROMA_DIR, collection_name="youth_housing_policy"):
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"  🔄 Chroma 임베딩 중... [{i+1}/{len(chunks)}]")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedder,
                collection_name=collection_name,
                persist_directory=persist_dir
            )
        else:
            vectorstore.add_documents(batch)
        time.sleep(0.5)
    print(f"✅ Chroma 임베딩 완료 → {persist_dir} ({len(chunks)}개 청크)")
    return vectorstore

def load_chroma(embedder,
                persist_dir=CHROMA_DIR,
                collection_name="youth_housing_policy"):
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedder
    )


# FAISS
def embed_and_save_faiss(chunks, embedder, persist_dir=FAISS_DIR):
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"  🔄 FAISS 임베딩 중... [{i+1}/{len(chunks)}]")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedder)
        else:
            vectorstore.add_documents(batch)
        time.sleep(0.5)
    vectorstore.save_local(persist_dir)
    print(f"✅ FAISS 임베딩 완료 → {persist_dir} ({len(chunks)}개 청크)")
    return vectorstore

def load_faiss(embedder, persist_dir=FAISS_DIR):
    return FAISS.load_local(
        persist_dir, embedder,
        allow_dangerous_deserialization=True
    )

# 하위 호환
def housing_embed_and_save(chunks):
    embedder = get_openai_embedder_small()
    return embed_and_save_chroma(chunks, embedder, persist_dir=CHROMA_DIR, collection_name="youth_housing_policy")

def housing_load_vectorstore():
    embedder = get_openai_embedder_small()
    return load_chroma(embedder, persist_dir=CHROMA_DIR, collection_name="youth_housing_policy")

def finance_embed_and_save(chunks):
    embedder = get_openai_embedder_small()
    return embed_and_save_chroma(chunks, embedder,
                                  persist_dir=FINANCE_CHROMA_DIR,
                                  collection_name="youth_finance_policy")

def finance_load_vectorstore():
    embedder = get_openai_embedder_small()
    return load_chroma(embedder,
                       persist_dir=FINANCE_CHROMA_DIR,
                       collection_name="youth_finance_policy")