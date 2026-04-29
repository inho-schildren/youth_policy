from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from core.embedder_vectorstore import housing_load_vectorstore, finance_load_vectorstore
from config import RETRIEVER_K, BM25_WEIGHT, DENSE_WEIGHT, MMR_FETCH_K, MMR_LAMBDA, OPENAI_API_KEY, LLM_MODEL

def get_basic_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )
    print(f"✅ Basic Retriever 생성 완료")
    return retriever

def get_ensemble_retriever(chunks, vectorstore):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVER_K

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

def get_contextual_compression_retriever(vectorstore):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )
    )
    print("✅ Contextual Compression Retriever 생성 완료")
    return retriever

def get_selfquery_retriever(vectorstore, metadata_field_info, document_contents):
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    selfquery_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_contents,
        metadata_field_info=metadata_field_info,
        verbose=True
    )
    print("✅ SelfQuery Retriever 생성 완료")
    return selfquery_retriever

FINANCE_METADATA_FIELD_INFO = [
    AttributeInfo(name="title",             description="상품명",                        type="string"),
    AttributeInfo(name="category",          description="카테고리: 금융/주거",             type="string"),
    AttributeInfo(name="region",            description="지원 지역",                     type="string"),
    AttributeInfo(name="target",            description="대상자: 청년/신혼부부 등",        type="string"),
    AttributeInfo(name="age_min",           description="최소 나이",                     type="integer"),
    AttributeInfo(name="age_max",           description="최대 나이",                     type="integer"),
    AttributeInfo(name="marital_status",    description="혼인 조건: 미혼/기혼/예비신혼부부/무관", type="string"),
    AttributeInfo(name="requires_no_house", description="무주택 조건 여부",               type="boolean"),
    AttributeInfo(name="income_max_man",    description="최대 소득 한도(만원)",            type="integer"),
    AttributeInfo(name="loan_limit_man",    description="최대 지원 한도(만원)",            type="integer"),
    AttributeInfo(name="asset_max_man",     description="최대 자산 한도(만원)",            type="integer"),
    AttributeInfo(name="is_first_purchase", description="최초 구매 조건 여부",             type="boolean"),
]

def housing_retriever(chunks):
    vectorstore = housing_load_vectorstore()
    return get_ensemble_retriever(chunks, vectorstore)

def finance_retriever(chunks):
    vectorstore = finance_load_vectorstore()
    return get_ensemble_retriever(chunks, vectorstore)