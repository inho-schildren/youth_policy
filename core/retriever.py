from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from core.embedder_vectorstore import load_vectorstore
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

def get_finance_selfquery_retriever(chunks=None, vectorstore=None):

    metadata_field_info = [
        AttributeInfo(name="title", description="상품명", type="string"),
        AttributeInfo(name="region", description="지원 지역", type="string"),
        AttributeInfo(name="age_min", description="최소 나이", type="integer"),
        AttributeInfo(name="age_max", description="최대 나이", type="integer"),
        AttributeInfo(name="loan_type", description="대출 유형: 전세/월세/구입", type="string"),
        AttributeInfo(name="interest_rate_min", description="최저 금리", type="float"),
        AttributeInfo(name="interest_rate_max", description="최고 금리", type="float"),
        AttributeInfo(name="income_max_man", description="최대 소득 한도(만원)", type="integer"),
        AttributeInfo(name="loan_limit_man", description="최대 대출 한도(만원)", type="integer"),
        AttributeInfo(name="marital_status", description="혼인 조건: 미혼/기혼/예비신혼부부/무관", type="string"),
        AttributeInfo(name="target", description="대상자: 청년/신혼부부 등", type="string"),
        AttributeInfo(name="requires_no_house", description="무주택 조건 여부", type="boolean"),
    ]

    if vectorstore is None:
        from core.embedder_vectorstore import load_finance_vectorstore
        vectorstore = load_finance_vectorstore()

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    selfquery_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="청년 주거 금융 정책 문서",
        metadata_field_info=metadata_field_info,
        verbose=True
    )

    print("✅ 금융 SelfQuery Retriever 생성 완료")
    return selfquery_retriever