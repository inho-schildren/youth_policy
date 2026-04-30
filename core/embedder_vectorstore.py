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

def get_openai_embedder():
    return OpenAIEmbeddings(
        model = EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

def get_huggingface_embedder():
    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}

    )

# CHROMA
def embed_and_save_chroma(chunks, embedder,
    persist_dir=CHROMA_DIR, collection_name="youth_housing_policy"):
    
    # ChromaDB는 메타데이터에 리스트를 허용하지 않으므로 문자열로 변환
    for chunk in chunks:
        chunk.metadata = {
            k: ", ".join(map(str, v)) if isinstance(v, list) else v 
            for k, v in chunk.metadata.items()
        }

    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"  Chroma Embedding... [{i+1}/{len(chunks)}]")
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
    print(f"✅ Chroma Embedding Done -> {persist_dir} ({len(chunks)} chunks)")
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
    embedder = get_openai_embedder()
    return embed_and_save_chroma(chunks, embedder, persist_dir=CHROMA_DIR, collection_name="youth_housing_policy")

def housing_load_vectorstore():
    embedder = get_openai_embedder()
    return load_chroma(embedder, persist_dir=CHROMA_DIR, collection_name="youth_housing_policy")

def finance_embed_and_save(chunks):
    embedder = get_openai_embedder()
    return embed_and_save_chroma(chunks, embedder,
                                  persist_dir=FINANCE_CHROMA_DIR,
                                  collection_name="youth_finance_policy")

def finance_load_vectorstore():
    embedder = get_openai_embedder()
    return load_chroma(embedder,
                       persist_dir=FINANCE_CHROMA_DIR,
                       collection_name="youth_finance_policy")