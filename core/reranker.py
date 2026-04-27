from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from config import RETRIEVER_K, RERANKER_TOP_N
import os

def get_reranker(retriever):
    compressor = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        top_n=RERANKER_TOP_N,
        model="rerank-multilingual-v3.0"
    )

    reranker = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    print("✅ Reranker 생성 완료")
    return reranker