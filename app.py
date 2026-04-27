import streamlit as st
import json
import re
from pipeline import run_pipeline
import chain.rag_chain as rag

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

    /* ── 배경 ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* ── 메인 타이틀 ── */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
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
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ── 섹션 헤더 ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        padding: 0.8rem 1.2rem;
        border-left: 5px solid #a78bfa;
        background: rgba(167, 139, 250, 0.08);
        border-radius: 0 10px 10px 0;
        margin: 1.5rem 0 1rem;
    }

    /* ── 컬럼 헤더 ── */
    .column-header {
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        padding: 0.85rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .column-header.housing {
        background: rgba(167, 139, 250, 0.15);
        color: #a78bfa;
        border: 1.5px solid rgba(167, 139, 250, 0.35);
    }
    .column-header.finance {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1.5px solid rgba(52, 211, 153, 0.35);
    }

    /* ── 정책 카드 ── */
    .policy-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.7rem 1.5rem 1.4rem;
        margin: 1.4rem 0;
        position: relative;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    .policy-card:hover {
        border-color: rgba(167, 139, 250, 0.45);
        box-shadow: 0 8px 24px rgba(167, 139, 250, 0.15);
    }

    .rank-badge {
        position: absolute; top: -14px; left: 18px;
        color: white; font-size: 0.85rem; font-weight: 700;
        padding: 0.4rem 1rem; border-radius: 20px;
    }
    .rank-badge.gold { background: linear-gradient(135deg, #fbbf24, #f59e0b); box-shadow: 0 4px 12px rgba(251, 191, 36, 0.4); }
    .rank-badge.silver { background: linear-gradient(135deg, #94a3b8, #64748b); box-shadow: 0 4px 12px rgba(148, 163, 184, 0.4); }

    .policy-card-title {
        font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
        margin: 0.7rem 0 1.2rem; padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .policy-row {
        display: flex; gap: 0.7rem; margin-bottom: 0.7rem;
        font-size: 1rem; line-height: 1.6;
    }
    .policy-label { color: #94a3b8; min-width: 95px; font-size: 0.95rem; font-weight: 600; }
    .policy-value { color: #e2e8f0; flex: 1; font-weight: 500; }

    /* ── 신청 링크 버튼 ── */
    .policy-link {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.6rem 1.2rem;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        color: white !important;
        text-decoration: none !important;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .policy-link:hover { opacity: 0.85; }
    .policy-link.finance {
        background: linear-gradient(135deg, #34d399, #60a5fa);
    }

    /* ── 어시스턴트 되묻기 카드 ── */
    .ask-back-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .ask-back-icon {
        color: #a78bfa;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .ask-back-text {
        color: #f1f5f9 !important;
        font-size: 1.08rem !important;
        line-height: 1.8 !important;
    }

    /* ── success 카드 ── */
    .success-card {
        background: rgba(167, 139, 250, 0.12);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        color: #c4b5fd !important;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.8rem 0;
    }

    /* ── spinner ── */
    [data-testid="stSpinner"] p,
    [data-testid="stSpinner"] div,
    .stSpinner p {
        color: #a78bfa !important;
        font-size: 1rem !important;
    }

    /* ── 보고서 히어로 ── */
    .report-hero {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(96, 165, 250, 0.1));
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 22px; padding: 2.2rem; margin: 1rem 0 1.5rem;
        text-align: center; position: relative; overflow: hidden;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    .report-hero::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399, #fbbf24);
    }
    .report-hero-title {
        font-family: 'Playfair Display', serif; font-size: 1.9rem;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .report-hero-summary { color: #cbd5e1; font-size: 1.08rem; line-height: 1.8; margin-top: 0.8rem; }

    /* ── 보고서 섹션 카드 ── */
    .report-section-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px; padding: 1.7rem; margin-bottom: 1.3rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    }
    .report-section-title {
        font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
        margin-bottom: 1.1rem; display: flex; align-items: center; gap: 0.6rem;
    }

    /* ── 장단점 박스 ── */
    .pros-cons-grid {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 1rem; margin-top: 0.8rem;
    }
    .pros-box {
        background: rgba(52, 211, 153, 0.08);
        border-left: 4px solid #34d399;
        border-radius: 12px; padding: 1.1rem 1.3rem;
    }
    .cons-box {
        background: rgba(248, 113, 113, 0.08);
        border-left: 4px solid #f87171;
        border-radius: 12px; padding: 1.1rem 1.3rem;
    }
    .pros-title { color: #34d399; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }
    .cons-title { color: #f87171; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }
    .pros-box ul, .cons-box ul {
        margin: 0; padding-left: 1.3rem; color: #cbd5e1;
        font-size: 0.95rem; line-height: 1.8;
    }

    /* ── 보고서 테이블 ── */
    .report-table {
        width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.98rem;
    }
    .report-table th {
        background: rgba(167, 139, 250, 0.15); color: #a78bfa;
        padding: 0.85rem 1rem; text-align: left;
        border-bottom: 2px solid rgba(167, 139, 250, 0.3); font-weight: 700;
        font-size: 0.95rem;
    }
    .report-table td {
        padding: 0.85rem 1rem; color: #cbd5e1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        vertical-align: top; line-height: 1.7;
    }
    .report-table tr:last-child td { border-bottom: none; }

    /* ── 전략/추천/경고 박스 ── */
    .strategy-box {
        background: rgba(96, 165, 250, 0.08);
        border-left: 4px solid #60a5fa;
        border-radius: 12px; padding: 1.2rem 1.4rem; color: #cbd5e1;
        font-size: 1rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .recommend-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(167, 139, 250, 0.08));
        border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 16px;
        padding: 1.6rem; color: #e2e8f0; font-size: 1.05rem;
        line-height: 1.8; margin-top: 0.5rem;
    }
    .warning-box {
        background: rgba(248, 113, 113, 0.08);
        border-left: 4px solid #f87171;
        border-radius: 12px; padding: 1.1rem 1.4rem; color: #cbd5e1;
        font-size: 1rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .warning-box ul, .recommend-box ul, .strategy-box ul {
        margin: 0; padding-left: 1.3rem;
    }

    /* ── 검색창 ── */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #64748b !important;
        padding: 0.95rem 1.2rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
        font-weight: 400 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2) !important;
    }

    /* ── 버튼 ── */
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 700 !important;
        padding: 0.9rem 2rem !important; font-size: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(167, 139, 250, 0.3) !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; }

    /* ── expander (상세내용) ── */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] div {
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
    }
    [data-testid="stExpander"] strong {
        color: #a78bfa !important;
    }

    /* ── 일반 텍스트 ── */
    .stMarkdown, .stMarkdown p { color: #cbd5e1 !important; font-size: 1rem !important; }
    h1, h2, h3, h4 { color: #e2e8f0 !important; }

    /* divider */
    hr { border-color: rgba(255, 255, 255, 0.08) !important; margin: 1.8rem 0 !important; }

    /* 알림 메시지 */
    .stAlert { border-radius: 12px !important; font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── 파이프라인 초기화 ─────────────────────────────────────
@st.cache_resource
def init():
    retriever = run_pipeline()
    chain = rag.build_chain(retriever)
    return retriever, chain

retriever, chain = init()

DUMMY_FINANCE_POLICIES = [
    {
        "title": "청년 전용 버팀목 전세자금 대출",
        "summary": "만 19~34세 청년을 위한 저금리 전세자금 대출",
        "target": "만 19~34세 무주택 청년",
        "income": "연소득 5,000만원 이하",
        "support": "최대 2억원 / 금리 연 2.3%",
        "period": "2024.01.01 ~ 2024.12.31",
        "link": "https://nhuf.molit.go.kr/"
    },
    {
        "title": "청년 주택드림 청약통장",
        "summary": "청년의 내집 마련을 위한 우대금리 청약통장",
        "target": "만 19~34세 무주택 청년",
        "income": "연소득 3,600만원 이하",
        "support": "금리 최대 4.5% / 이자소득 비과세",
        "period": "2024.01.01 ~ 2024.12.31",
        "link": "https://www.molit.go.kr/"
    }
]


# ── 정책 카드 렌더링 ──────────────────────────────────────
def render_policy_card(rank, title, items, link=None, link_type="housing"):
    rank_class = "gold" if rank == 1 else "silver"
    rank_label = "🥇 1순위" if rank == 1 else "🥈 2순위"

    rows_html = "".join([
        f'<div class="policy-row"><span class="policy-label">{label}</span><span class="policy-value">{value}</span></div>'
        for label, value in items
    ])


    card_html = (
        f'<div class="policy-card">'
        f'<div class="rank-badge {rank_class}">{rank_label}</div>'
        f'<div class="policy-card-title">{title}</div>'
        f'{rows_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ── 보고서 JSON 파싱 ──────────────────────────────────────
def extract_json(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def render_list_html(items):
    if not items:
        return ""
    if isinstance(items, str):
        items = [items]
    return "<ul>" + "".join([f"<li>{item}</li>" for item in items]) + "</ul>"


def render_report(data):
    summary = data.get("summary", "")
    hero_html = (
        f'<div class="report-hero">'
        f'<div class="report-hero-title">📊 맞춤형 정책 종합 보고서</div>'
        f'<div style="color:#94a3b8;font-size:0.95rem;">AI가 분석한 당신을 위한 최적의 정책 가이드</div>'
        f'<div class="report-hero-summary">{summary}</div>'
        f'</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    housing = data.get("housing_analysis", [])
    if housing:
        rows = "".join([
            f'<tr><td><strong style="color:#a78bfa">{h.get("title", "-")}</strong></td>'
            f'<td>{h.get("core", "-")}</td>'
            f'<td>{h.get("priority", "-")}</td></tr>'
            for h in housing
        ])
        section_html = (
            '<div class="report-section-card">'
            '<div class="report-section-title">🏘️ 주거 정책 분석</div>'
            '<table class="report-table">'
            '<thead><tr><th>정책명</th><th>핵심 내용</th><th>추천 우선순위</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(section_html, unsafe_allow_html=True)

        for h in housing:
            pros = render_list_html(h.get("pros", []))
            cons = render_list_html(h.get("cons", []))
            pc_html = (
                '<div class="report-section-card">'
                f'<div class="report-section-title">▸ {h.get("title", "-")}</div>'
                '<div class="pros-cons-grid">'
                f'<div class="pros-box"><div class="pros-title">✅ 장점</div>{pros}</div>'
                f'<div class="cons-box"><div class="cons-title">⚠️ 단점/유의사항</div>{cons}</div>'
                '</div>'
                '</div>'
            )
            st.markdown(pc_html, unsafe_allow_html=True)

    finance = data.get("finance_analysis", [])
    if finance:
        rows = "".join([
            f'<tr><td><strong style="color:#34d399">{f.get("title", "-")}</strong></td>'
            f'<td>{f.get("core", "-")}</td>'
            f'<td>{f.get("strategy", "-")}</td></tr>'
            for f in finance
        ])
        section_html = (
            '<div class="report-section-card">'
            '<div class="report-section-title">💳 금융 정책 분석</div>'
            '<table class="report-table">'
            '<thead><tr><th>정책명</th><th>핵심 내용</th><th>활용 전략</th></tr></thead>'
            f'<tbody>{rows}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(section_html, unsafe_allow_html=True)

        for f in finance:
            pros = render_list_html(f.get("pros", []))
            cons = render_list_html(f.get("cons", []))
            pc_html = (
                '<div class="report-section-card">'
                f'<div class="report-section-title">▸ {f.get("title", "-")}</div>'
                '<div class="pros-cons-grid">'
                f'<div class="pros-box"><div class="pros-title">✅ 장점</div>{pros}</div>'
                f'<div class="cons-box"><div class="cons-title">⚠️ 단점/유의사항</div>{cons}</div>'
                '</div>'
                '</div>'
            )
            st.markdown(pc_html, unsafe_allow_html=True)

    combo = data.get("combination_strategy", "")
    if combo:
        combo_html = (
            '<div class="report-section-card">'
            '<div class="report-section-title">🔗 주거 + 금융 조합 전략</div>'
            f'<div class="strategy-box">{combo}</div>'
            '</div>'
        )
        st.markdown(combo_html, unsafe_allow_html=True)

    warnings = data.get("warnings", [])
    if warnings:
        warn_list = render_list_html(warnings)
        warn_html = (
            '<div class="report-section-card">'
            '<div class="report-section-title">⚠️ 주의사항 및 리스크</div>'
            f'<div class="warning-box">{warn_list}</div>'
            '</div>'
        )
        st.markdown(warn_html, unsafe_allow_html=True)

    final = data.get("final_recommendation", "")
    if final:
        final_html = (
            '<div class="report-section-card">'
            '<div class="report-section-title">🎯 종합 추천 및 행동 계획</div>'
            f'<div class="recommend-box">{final}</div>'
            '</div>'
        )
        st.markdown(final_html, unsafe_allow_html=True)


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
        finished, result = rag.ask_llm(query)

        if not finished:
            # st.chat_message 대신 커스텀 카드 사용
            st.markdown(
                f'<div class="ask-back-card">'
                f'<div class="ask-back-icon">🤖 어시스턴트</div>'
                f'<div class="ask-back-text">{result}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # st.success 대신 커스텀 카드 사용
            st.markdown('<div class="success-card">✅ 검색을 시작합니다!</div>', unsafe_allow_html=True)

            with st.spinner("관련 정책을 검색하고 있습니다..."):
                housing_query = result['housing_query']
                finance_query = result['finance_query']
                housing_docs = retriever.invoke(housing_query)[:2]
                finance_docs = retriever.invoke(finance_query)[:2]

            st.markdown('<div class="section-header">✨ 맞춤 추천 정책</div>', unsafe_allow_html=True)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="column-header housing">🏘️ 주거 정책 TOP 2</div>', unsafe_allow_html=True)
                for i, doc in enumerate(housing_docs):
                    items = [
                        ("📍 지역", doc.metadata.get('region', '-')),
                        ("👥 대상", doc.metadata.get('target', '-')),
                        ("💰 소득조건", doc.metadata.get('income_condition', '-')),
                        ("🏠 지원유형", doc.metadata.get('support_type', '-')),
                        ("📅 신청기간", doc.metadata.get('application_period', '-')),
                    ]
                    link = doc.metadata.get('link') or doc.metadata.get('url') or doc.metadata.get('apply_url')
                    render_policy_card(
                        rank=i + 1,
                        title=doc.metadata.get('title', '제목 없음'),
                        items=items,
                        link=link,
                        link_type="housing"
                    )
                    

            with col_right:
                st.markdown('<div class="column-header finance">💳 금융 정책 TOP 2</div>', unsafe_allow_html=True)
                for i, policy in enumerate(DUMMY_FINANCE_POLICIES):
                    items = [
                        ("📋 요약", policy['summary']),
                        ("👥 대상", policy['target']),
                        ("💵 소득조건", policy['income']),
                        ("🎁 지원내용", policy['support']),
                        ("📅 신청기간", policy['period']),
                    ]
                    render_policy_card(
                        rank=i + 1,
                        title=policy['title'],
                        items=items,
                        link=policy.get('link'),
                        link_type="finance"
                    )
                    

            st.divider()

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

아래 주거정책 2개와 금융정책 2개를 분석해 종합 보고서를 작성해줘.
반드시 아래 JSON 형식으로만 응답해. 마크다운, 설명, 코드블록 없이 순수 JSON만 출력.

{{
  "summary": "사용자 상황 분석 및 핵심 요약 (3줄 이내, 자연스러운 문장)",
  "housing_analysis": [
    {{
      "title": "정책명",
      "core": "핵심 내용 한 줄 요약",
      "priority": "신청 우선순위 추천 한 줄",
      "pros": ["장점1", "장점2", "장점3"],
      "cons": ["단점/유의사항1", "단점/유의사항2"]
    }}
  ],
  "finance_analysis": [
    {{
      "title": "정책명",
      "core": "핵심 내용 한 줄 요약",
      "strategy": "활용 전략 한 줄",
      "pros": ["장점1", "장점2"],
      "cons": ["단점/유의사항1", "단점/유의사항2"]
    }}
  ],
  "combination_strategy": "주거+금융 정책을 함께 활용하는 최적 방법과 중복 수혜 가능 여부 (2-3문장)",
  "warnings": ["놓치기 쉬운 조건1", "신청 시 주의점1", "신청 시 주의점2"],
  "final_recommendation": "사용자에게 가장 적합한 정책 조합 및 구체적인 행동 계획 (3-4문장)"
}}

[제공 데이터]
{housing_context}

{finance_context}
"""
                report_raw = chain.invoke(report_prompt)

            try:
                json_str = extract_json(report_raw)
                report_data = json.loads(json_str)
                render_report(report_data)
            except (json.JSONDecodeError, Exception):
                st.warning("보고서 형식 변환에 실패해 텍스트로 표시합니다.")
                st.markdown(
                    '<div class="report-hero">'
                    '<div class="report-hero-title">📊 맞춤형 정책 종합 보고서</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.markdown(report_raw)