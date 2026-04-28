from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import RETRIEVER_K, RERANKER_TOP_N
import os

def get_cohere_reranker(retriever):
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

def get_cross_encoder_reranker(retriever):
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3"  # 한국어 지원
    )
    compressor = CrossEncoderReranker(
        model=model,
        top_n=RERANKER_TOP_N
    )
    reranker = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    print("✅ Cross-Encoder Reranker 생성 완료")
    return reranker