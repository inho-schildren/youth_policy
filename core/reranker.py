# from langchain.retrievers import ContextualCompressionRetriever
# from langchain_cohere import CohereRerank
# from config import RETRIEVER_K, RERANKER_TOP_N
# import os

# def get_reranker(retriever):
#     compressor = CohereRerank(
#         cohere_api_key=os.getenv("COHERE_API_KEY"),
#         top_n=RERANKER_TOP_N,
#         model="rerank-multilingual-v3.0"
#     )

#     reranker = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=retriever
#     )
#     print("✅ Reranker 생성 완료")
#     return reranker

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import RETRIEVER_K, RERANKER_TOP_N

def get_reranker(retriever):
    # 1. HuggingFace Cross-Encoder 모델 로드
    # 한국어/다국어 성능이 좋은 모델 추천: "BAAI/bge-reranker-v2-m3" 또는 "Dongjin-kr/ko-reranker"
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

    # 2. Compressor 설정 (top_n 지정)
    compressor = CrossEncoderReranker(
        model=model, 
        top_n=RERANKER_TOP_N
    )

    # 3. ContextualCompressionRetriever 생성
    reranker = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    print("✅ Cross-Encoder Reranker 생성 완료")
    return reranker