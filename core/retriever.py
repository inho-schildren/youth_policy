from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from core.embedder import load_vectorstore
from config import RETRIEVER_K, BM25_WEIGHT, DENSE_WEIGHT, MMR_FETCH_K, MMR_LAMBDA, OPENAI_API_KEY, LLM_MODEL

# def get_retriever(chunks):
#     bm25_retriever = BM25Retriever.from_documents(chunks)
#     bm25_retriever.k = RETRIEVER_K

#     vectorstore = load_vectorstore()
#     dense_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, dense_retriever],
#         weights=[BM25_WEIGHT, DENSE_WEIGHT]
#     )
#     print(f"✅ Ensemble Retriever 생성 완료 (BM25: {BM25_WEIGHT} / Dense: {DENSE_WEIGHT})")
#     return ensemble_retriever

def get_retriever(chunks):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVER_K

    vectorstore = load_vectorstore()
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA
        }
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[BM25_WEIGHT, DENSE_WEIGHT]
    )
    print(f"✅ Ensemble Retriever 생성 완료 (BM25: {BM25_WEIGHT} / Dense: {DENSE_WEIGHT})")
    return ensemble_retriever