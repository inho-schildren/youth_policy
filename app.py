import streamlit as st
from pipeline import run_pipeline
from chain.rag_chain import build_chain

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(
    page_title="청년 주택 정책 검색",
    page_icon="🏠",
    layout="wide"
)

# ── 커스텀 CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Playfair+Display:wght@700&display=swap');

    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 메인 타이틀 */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        letter-spacing: -0.5px;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* 섹션 헤더 */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        padding: 0.6rem 1rem;
        border-left: 4px solid #a78bfa;
        background: rgba(167, 139, 250, 0.08);
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem;
    }

    /* 정책 카드 */
    .policy-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: transform 0.2s;
    }

    .policy-card:hover {
        transform: translateY(-2px);
        border-color: rgba(167, 139, 250, 0.4);
    }

    .policy-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .policy-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }

    .badge-housing {
        background: rgba(167, 139, 250, 0.2);
        color: #a78bfa;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }

    .badge-finance {
        background: rgba(52, 211, 153, 0.2);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .policy-item {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
        color: #cbd5e1;
        font-size: 0.88rem;
    }

    .policy-item-label {
        color: #94a3b8;
        min-width: 70px;
        font-size: 0.82rem;
    }

    /* 보고서 */
    .report-container {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
    }

    .report-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #60a5fa;
    }

    .report-section.advantage {
        border-left-color: #34d399;
    }

    .report-section.disadvantage {
        border-left-color: #f87171;
    }

    .report-section.insight {
        border-left-color: #fbbf24;
    }

    /* 검색창 */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
    }

    /* divider */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* 텍스트 색상 */
    .stMarkdown, p, li {
        color: #cbd5e1 !important;
    }

    h1, h2, h3 {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── 파이프라인 초기화 ─────────────────────────────────────
@st.cache_resource
def init():
    retriever = run_pipeline()
    chain = build_chain(retriever)
    return retriever, chain

retriever, chain = init()

# ── 임시 금융 정책 데이터 ─────────────────────────────────
DUMMY_FINANCE_POLICIES = [
    {
        "title": "청년 전용 버팀목 전세자금 대출",
        "summary": "만 19~34세 청년을 위한 저금리 전세자금 대출",
        "target": "만 19~34세 무주택 청년",
        "income": "연소득 5,000만원 이하",
        "support": "최대 2억원 / 금리 연 2.3%",
        "period": "2024.01.01 ~ 2024.12.31"
    },
    {
        "title": "청년 주택드림 청약통장",
        "summary": "청년의 내집 마련을 위한 우대금리 청약통장",
        "target": "만 19~34세 무주택 청년",
        "income": "연소득 3,600만원 이하",
        "support": "금리 최대 4.5% / 이자소득 비과세",
        "period": "2024.01.01 ~ 2024.12.31"
    }
]

# ── UI ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 청년 주택 정책 어시스턴트</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">나에게 맞는 주거·금융 정책을 찾고 종합 보고서를 받아보세요</div>', unsafe_allow_html=True)

st.divider()

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="질문",
        placeholder="예) 금천구에 거주 예정인 신혼부부인데 주거 정책 알려줘",
        label_visibility="collapsed"
    )
with col_btn:
    search = st.button("🔍 검색", use_container_width=True)

if search:
    if not query.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("관련 정책을 검색하고 있습니다..."):
            docs = retriever.invoke(query)
            housing_docs = docs[:2]

        # ── 주거 정책 TOP 2 ───────────────────────────────
        st.markdown('<div class="section-header">🏘️ 추천 주거 정책 TOP 2</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, doc in enumerate(housing_docs):
            with cols[i]:
                st.markdown(f"""
                <div class="policy-card">
                    <div class="policy-badge badge-housing">주거 정책</div>
                    <div class="policy-card-title">{doc.metadata.get('title', '제목 없음')}</div>
                    <div class="policy-item">
                        <span class="policy-item-label">📍 지역</span>
                        <span>{doc.metadata.get('region', '-')}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">👥 대상</span>
                        <span>{doc.metadata.get('target', '-')}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">💰 소득조건</span>
                        <span>{doc.metadata.get('income_condition', '-')}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">🏠 지원유형</span>
                        <span>{doc.metadata.get('support_type', '-')}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">📅 신청기간</span>
                        <span>{doc.metadata.get('application_period', '-')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("상세 내용 보기"):
                    st.write(doc.page_content)

        st.divider()

        # ── 금융 정책 TOP 2 ───────────────────────────────
        st.markdown('<div class="section-header">💳 추천 금융 정책 TOP 2</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, policy in enumerate(DUMMY_FINANCE_POLICIES):
            with cols[i]:
                st.markdown(f"""
                <div class="policy-card">
                    <div class="policy-badge badge-finance">금융 정책</div>
                    <div class="policy-card-title">{policy['title']}</div>
                    <div class="policy-item">
                        <span class="policy-item-label">📋 요약</span>
                        <span>{policy['summary']}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">👥 대상</span>
                        <span>{policy['target']}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">💵 소득조건</span>
                        <span>{policy['income']}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">🎁 지원내용</span>
                        <span>{policy['support']}</span>
                    </div>
                    <div class="policy-item">
                        <span class="policy-item-label">📅 신청기간</span>
                        <span>{policy['period']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── 종합 보고서 ───────────────────────────────────
        st.markdown('<div class="section-header">📋 맞춤 정책 종합 보고서</div>', unsafe_allow_html=True)

        with st.spinner("보고서를 작성하고 있습니다..."):
            housing_context = "\n".join([
                f"[주거정책 {i+1}]\n제목: {doc.metadata.get('title')}\n지역: {doc.metadata.get('region')}\n대상: {doc.metadata.get('target')}\n소득조건: {doc.metadata.get('income_condition')}\n내용: {doc.page_content[:500]}"
                for i, doc in enumerate(housing_docs)
            ])
            finance_context = "\n".join([
                f"[금융정책 {i+1}]\n제목: {p['title']}\n대상: {p['target']}\n소득조건: {p['income']}\n지원내용: {p['support']}"
                for i, p in enumerate(DUMMY_FINANCE_POLICIES)
            ])

            report_prompt = f"""
사용자 질문: {query}

아래 주거정책 2개와 금융정책 2개를 바탕으로 종합 보고서를 작성해줘.
보고서는 반드시 아래 구조로 작성해줘.

## 1. 요약
- 사용자 상황 분석 및 핵심 요약 (3줄 이내)

## 2. 주거 정책 분석
- 각 정책의 핵심 내용
- 장점
- 단점 및 유의사항
- 신청 우선순위 추천

## 3. 금융 정책 분석
- 각 정책의 핵심 내용
- 장점
- 단점 및 유의사항
- 활용 전략

## 4. 주거 + 금융 정책 조합 전략
- 두 정책을 함께 활용하는 최적의 방법
- 중복 수혜 가능 여부

## 5. 주의사항 및 리스크
- 놓치기 쉬운 조건
- 신청 시 주의할 점

## 6. 종합 추천
- 사용자에게 가장 적합한 정책 조합 및 행동 계획

{housing_context}

{finance_context}
"""
            report = chain.invoke(report_prompt)

        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(report)
        st.markdown('</div>', unsafe_allow_html=True)