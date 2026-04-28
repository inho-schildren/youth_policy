import streamlit as st
import json
import re
from pipeline import run_pipeline, run_finance_pipeline
from chain.rag_chain import build_chain, ask_llm, state

st.set_page_config(
    page_title="청년 주택 정책 검색",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Playfair+Display:wght@700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Noto Sans KR', sans-serif;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1.5rem 0 0.5rem; letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center; color: #94a3b8;
        font-size: 1.15rem; margin-bottom: 2rem; font-weight: 400;
    }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #e2e8f0;
        padding: 0.8rem 1.2rem; border-left: 5px solid #a78bfa;
        background: rgba(167, 139, 250, 0.08);
        border-radius: 0 10px 10px 0; margin: 1.5rem 0 1rem;
    }
    .column-header {
        font-size: 1.2rem; font-weight: 700; text-align: center;
        padding: 0.85rem; border-radius: 12px; margin-bottom: 1rem;
    }
    .column-header.housing {
        background: rgba(167, 139, 250, 0.15); color: #a78bfa;
        border: 1.5px solid rgba(167, 139, 250, 0.35);
    }
    .column-header.finance {
        background: rgba(52, 211, 153, 0.15); color: #34d399;
        border: 1.5px solid rgba(52, 211, 153, 0.35);
    }
    .policy-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px; padding: 1.7rem 1.5rem 1.4rem;
        margin: 1.4rem 0; position: relative;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
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
    .rank-badge.gold { background: linear-gradient(135deg, #fbbf24, #f59e0b); box-shadow: 0 4px 12px rgba(251,191,36,0.4); }
    .rank-badge.silver { background: linear-gradient(135deg, #94a3b8, #64748b); box-shadow: 0 4px 12px rgba(148,163,184,0.4); }
    .rank-badge.bronze { background: linear-gradient(135deg, #cd7c4a, #a0522d); box-shadow: 0 4px 12px rgba(205,124,74,0.4); }
    .policy-card-title {
        font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
        margin: 0.7rem 0 1.2rem; padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .policy-row { display: flex; gap: 0.7rem; margin-bottom: 0.7rem; font-size: 1rem; line-height: 1.6; }
    .policy-label { color: #94a3b8; min-width: 95px; font-size: 0.95rem; font-weight: 600; }
    .policy-value { color: #e2e8f0; flex: 1; font-weight: 500; }
    .ask-back-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 16px; padding: 1.4rem 1.6rem; margin: 1.2rem 0;
        backdrop-filter: blur(10px); box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .ask-back-icon { color: #a78bfa; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.7rem; }
    .ask-back-text { color: #f1f5f9 !important; font-size: 1.08rem !important; line-height: 1.8 !important; }
    .success-card {
        background: rgba(167,139,250,0.12);
        border: 1px solid rgba(167,139,250,0.35);
        border-radius: 12px; padding: 1rem 1.4rem;
        color: #c4b5fd !important; font-size: 1rem; font-weight: 600; margin: 0.8rem 0;
    }
    .query-info-card {
        background: rgba(167,139,250,0.08);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 12px; padding: 1rem 1.4rem; margin: 0.8rem 0;
    }
    .query-info-label { color: #a78bfa; font-size: 0.9rem; font-weight: 700; margin-bottom: 0.4rem; }
    .query-info-item { display: flex; gap: 0.5rem; margin-bottom: 0.3rem; }
    .query-info-key { color: #94a3b8; font-size: 0.9rem; min-width: 80px; }
    .query-info-val { color: #e2e8f0; font-size: 0.9rem; font-weight: 600; }
    .report-hero {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.1));
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 22px; padding: 2.2rem; margin: 1rem 0 1.5rem;
        text-align: center; position: relative; overflow: hidden;
        box-shadow: 0 6px 24px rgba(0,0,0,0.2); backdrop-filter: blur(10px);
    }
    .report-hero::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399, #fbbf24);
    }
    .report-hero-title {
        font-family: 'Playfair Display', serif; font-size: 1.9rem;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
    }
    .report-hero-summary { color: #cbd5e1; font-size: 1.08rem; line-height: 1.8; margin-top: 0.8rem; }
    .report-section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px; padding: 1.7rem; margin-bottom: 1.3rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15); backdrop-filter: blur(10px);
    }
    .report-section-title {
        font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
        margin-bottom: 1.1rem; display: flex; align-items: center; gap: 0.6rem;
    }
    .pros-cons-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.8rem; }
    .pros-box {
        background: rgba(52,211,153,0.08); border-left: 4px solid #34d399;
        border-radius: 12px; padding: 1.1rem 1.3rem;
    }
    .cons-box {
        background: rgba(248,113,113,0.08); border-left: 4px solid #f87171;
        border-radius: 12px; padding: 1.1rem 1.3rem;
    }
    .pros-title { color: #34d399; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }
    .cons-title { color: #f87171; font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem; }
    .pros-box ul, .cons-box ul { margin: 0; padding-left: 1.3rem; color: #cbd5e1; font-size: 0.95rem; line-height: 1.8; }
    .report-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.98rem; }
    .report-table th {
        background: rgba(167,139,250,0.15); color: #a78bfa;
        padding: 0.85rem 1rem; text-align: left;
        border-bottom: 2px solid rgba(167,139,250,0.3); font-weight: 700; font-size: 0.95rem;
    }
    .report-table td {
        padding: 0.85rem 1rem; color: #cbd5e1;
        border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: top; line-height: 1.7;
    }
    .report-table tr:last-child td { border-bottom: none; }
    .strategy-box {
        background: rgba(96,165,250,0.08); border-left: 4px solid #60a5fa;
        border-radius: 12px; padding: 1.2rem 1.4rem; color: #cbd5e1;
        font-size: 1rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .recommend-box {
        background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(167,139,250,0.08));
        border: 1px solid rgba(251,191,36,0.3); border-radius: 16px;
        padding: 1.6rem; color: #e2e8f0; font-size: 1.05rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .warning-box {
        background: rgba(248,113,113,0.08); border-left: 4px solid #f87171;
        border-radius: 12px; padding: 1.1rem 1.4rem; color: #cbd5e1;
        font-size: 1rem; line-height: 1.8; margin-top: 0.5rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1.5px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important; color: #e2e8f0 !important;
        padding: 0.95rem 1.2rem !important; font-size: 1.1rem !important; font-weight: 500 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa, #60a5fa) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; font-weight: 700 !important;
        padding: 0.9rem 2rem !important; font-size: 1.1rem !important;
        box-shadow: 0 4px 12px rgba(167,139,250,0.3) !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; }
    .stMarkdown, .stMarkdown p { color: #cbd5e1 !important; font-size: 1rem !important; }
    h1, h2, h3, h4 { color: #e2e8f0 !important; }
    hr { border-color: rgba(255,255,255,0.08) !important; margin: 1.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── 파이프라인 초기화 ─────────────────────────────────────
@st.cache_resource
def init():
    housing_retriever = run_pipeline()
    finance_retriever = run_finance_pipeline()
    top3_chain, report_chain = build_chain(housing_retriever, finance_retriever)
    return top3_chain, report_chain

top3_chain, report_chain = init()

# ── JSON 파싱 헬퍼 ────────────────────────────────────────
def extract_json(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text

# ── 정책 카드 렌더링 ──────────────────────────────────────
def render_policy_card(rank, title, policy_type, reason, content):
    rank_class  = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "silver")
    rank_label  = {1: "🥇 1순위", 2: "🥈 2순위", 3: "🥉 3순위"}.get(rank, f"{rank}순위")
    type_color  = "#a78bfa" if policy_type == "주거" else "#34d399"
    type_label  = "🏘️ 주거 정책" if policy_type == "주거" else "💳 금융 정책"

    card_html = f"""
    <div class="policy-card">
        <div class="rank-badge {rank_class}">{rank_label}</div>
        <div style="color:{type_color};font-size:0.85rem;font-weight:700;margin-bottom:0.4rem;">{type_label}</div>
        <div class="policy-card-title">{title}</div>
        <div class="policy-row">
            <span class="policy-label">📌 선정 이유</span>
            <span class="policy-value">{reason}</span>
        </div>
        <div class="policy-row">
            <span class="policy-label">📄 핵심 내용</span>
            <span class="policy-value">{content}</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# ── 리스트 렌더링 헬퍼 ────────────────────────────────────
def render_list_html(items):
    if not items:
        return ""
    if isinstance(items, str):
        items = [items]
    return "<ul>" + "".join([f"<li>{item}</li>" for item in items]) + "</ul>"

# ── 보고서 렌더링 ─────────────────────────────────────────
def render_report(data):
    # 히어로
    summary   = data.get("summary", "")
    hero_html = f"""
    <div class="report-hero">
        <div class="report-hero-title">📊 맞춤형 정책 종합 보고서</div>
        <div style="color:#94a3b8;font-size:0.95rem;">AI가 분석한 당신을 위한 최적의 정책 가이드</div>
        <div class="report-hero-summary">{summary}</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # 정책별 분석
    policy_analysis = data.get("policy_analysis", [])
    if policy_analysis:
        rows = "".join([
            f'<tr>'
            f'<td><strong style="color:#a78bfa">{p.get("title","-")}</strong></td>'
            f'<td style="color:#{"34d399" if p.get("type")=="금융" else "a78bfa"}">{p.get("type","-")}</td>'
            f'<td>{p.get("core","-")}</td>'
            f'</tr>'
            for p in policy_analysis
        ])
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">📋 정책별 분석 요약</div>
            <table class="report-table">
                <thead><tr><th>정책명</th><th>유형</th><th>핵심 내용</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        for p in policy_analysis:
            pros = render_list_html(p.get("pros", []))
            cons = render_list_html(p.get("cons", []))
            type_color = "#34d399" if p.get("type") == "금융" else "#a78bfa"
            st.markdown(f"""
            <div class="report-section-card">
                <div class="report-section-title">
                    <span style="color:{type_color}">▸</span> {p.get("title", "-")}
                </div>
                <div class="pros-cons-grid">
                    <div class="pros-box"><div class="pros-title">✅ 장점</div>{pros}</div>
                    <div class="cons-box"><div class="cons-title">⚠️ 단점/유의사항</div>{cons}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 조합 전략
    combination = data.get("combination", "")
    if combination:
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">🔗 정책 조합 전략</div>
            <div class="strategy-box">{combination}</div>
        </div>
        """, unsafe_allow_html=True)

    # 주의사항
    risks = data.get("risks", "")
    if risks:
        risks_html = render_list_html(risks) if isinstance(risks, list) else risks
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">⚠️ 주의사항 및 리스크</div>
            <div class="warning-box">{risks_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # 종합 추천
    recommendation = data.get("recommendation", "")
    if recommendation:
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">🎯 종합 추천 및 행동 계획</div>
            <div class="recommend-box">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)


# ── UI ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 청년 주택 정책 어시스턴트</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">나에게 맞는 주거·금융 정책을 찾고 종합 보고서를 받아보세요</div>', unsafe_allow_html=True)
st.divider()

with st.form(key='search_form', clear_on_submit=True):
    col_input, col_btn = st.columns([0.8, 0.2])
    with col_input:
        query = st.text_input(
            label="질문",
            placeholder="예) 금천구에 거주 예정인 신혼부부인데 주거 정책 알려줘",
            label_visibility="collapsed",
            key="query_input"
        )
    with col_btn:
        search = st.form_submit_button("🔍 검색", use_container_width=True)

if search:
    if not query.strip():
        st.warning("질문을 입력해주세요.")
    else:
        finished, result = ask_llm(query)

        # ── 추가 정보 요청 ────────────────────────────────
        if not finished:
            st.markdown(f"""
            <div class="ask-back-card">
                <div class="ask-back-icon">🤖 어시스턴트</div>
                <div class="ask-back-text">{result}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── 검색 및 결과 출력 ─────────────────────────────
        else:
            # 수집 정보 요약
            label_map = {
                "category": "문의 분류", "region": "관심 지역", "target": "현재 상황",
                "product_category": "금융 상품", "special_condition": "우대 조건",
                "housing_type": "주택 형태", "residence_requirement": "거주 요건"
            }
            items_html = "".join([
                f'<div class="query-info-item">'
                f'<span class="query-info-key">{label_map.get(k, k)}</span>'
                f'<span class="query-info-val">{v}</span>'
                f'</div>'
                for k, v in state.items() if v
            ])
            st.markdown(f"""
            <div class="query-info-card">
                <div class="query-info-label">📋 입력하신 정보</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="success-card">✅ 검색을 시작합니다!</div>', unsafe_allow_html=True)

            # ── TOP3 추천 ─────────────────────────────────
            st.markdown('<div class="section-header">✨ 맞춤 추천 정책 TOP3</div>', unsafe_allow_html=True)

            with st.spinner("정책을 검색하고 있습니다..."):
                housing_query = result['housing_query']
                finance_query = result['finance_query']
                top3_raw = top3_chain.invoke({
                    "query": f"{housing_query} {finance_query}"
                })

            try:
                json_str  = extract_json(top3_raw)
                top3_data = json.loads(json_str)
                top3_list = top3_data.get("top3", [])
            except:
                top3_list = []
                st.warning("TOP3 파싱 실패")

            for policy in top3_list:
                render_policy_card(
                    rank        = policy.get("rank", 1),
                    title       = policy.get("title", "-"),
                    policy_type = policy.get("type", "-"),
                    reason      = policy.get("reason", "-"),
                    content     = policy.get("content", "-")
                )

            st.divider()

            # ── 종합 보고서 ───────────────────────────────
            st.markdown('<div class="section-header">📋 맞춤 정책 종합 보고서</div>', unsafe_allow_html=True)

            with st.spinner("보고서를 작성하고 있습니다..."):
                report_raw = report_chain.invoke({
                    "query": f"{housing_query} {finance_query}"
                })

            try:
                json_str    = extract_json(report_raw)
                report_data = json.loads(json_str)
                render_report(report_data)
            except:
                st.warning("보고서 형식 변환 실패, 텍스트로 표시합니다.")
                st.markdown(report_raw)