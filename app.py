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

    /* ── 카카오톡 스타일 채팅 ────────────────────── */
    .chat-wrap {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1rem 1rem 0.5rem;
        max-height: 300px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    .chat-wrap::-webkit-scrollbar { width: 6px; }
    .chat-wrap::-webkit-scrollbar-thumb {
        background: rgba(167,139,250,0.4); border-radius: 3px;
    }
    .msg-row { display: flex; margin-bottom: 0.9rem; align-items: flex-end; gap: 0.5rem; }
    .msg-row.me  { flex-direction: row-reverse; }
    .msg-row.bot { flex-direction: row; }
    .avatar {
        width: 36px; height: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; flex-shrink: 0;
    }
    .avatar.me  { background: linear-gradient(135deg,#a78bfa,#60a5fa); }
    .avatar.bot { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.15); }
    .bubble-col { display: flex; flex-direction: column; max-width: 68%; }
    .msg-row.me  .bubble-col { align-items: flex-end; }
    .msg-row.bot .bubble-col { align-items: flex-start; }
    .sender-name { font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.25rem; font-weight: 600; }
    .bubble {
        padding: 0.75rem 1rem; border-radius: 18px;
        font-size: 0.97rem; line-height: 1.6; word-break: break-word;
        box-shadow: 0 2px 6px rgba(0,0,0,0.18);
    }
    .bubble.me {
        background: linear-gradient(135deg,#a78bfa,#60a5fa);
        color: white; font-weight: 500;
        border-bottom-right-radius: 4px;
    }
    .bubble.bot {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.13);
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
    }
    .msg-time { font-size: 0.72rem; color: #64748b; margin-top: 0.2rem; }

    /* ── 정책 카드 (가로 배치용) ─────────────────── */
    .column-header {
        font-size: 1.2rem; font-weight: 700; text-align: center;
        padding: 0.85rem; border-radius: 12px; margin-bottom: 1rem;
    }
    .column-header.housing {
        background: rgba(167, 139, 250, 0.15); color: #e2e8f0;
        border: 1.5px solid rgba(167, 139, 250, 0.35);
    }
    .column-header.finance {
        background: rgba(52, 211, 153, 0.15); color: #34d399;
        border: 1.5px solid rgba(52, 211, 153, 0.35);
    }
    .policy-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.7rem 1.3rem 1.4rem;
        margin: 1.4rem 0 0.5rem;
        position: relative;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        height: 100%;
        display: flex;
        flex-direction: column;
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
        font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
        margin: 0.7rem 0 1rem; padding-bottom: 0.7rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        min-height: 3rem;
    }
    .policy-section { margin-bottom: 0.9rem; font-size: 0.95rem; line-height: 1.6; }
    .policy-section-label {
        color: #94a3b8; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 0.35rem; display: block;
    }
    .policy-section-value { color: #e2e8f0; font-weight: 500; }

    /* ── 보고서 ──────────────────────────────────── */
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
        background: rgba(255,255,255,0.1); color: #f1f5f9;
        padding: 0.85rem 1rem; text-align: left;
        border-bottom: 2px solid rgba(255,255,255,0.2); font-weight: 700; font-size: 0.95rem;
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

    /* ── 입력 / 버튼 ─────────────────────────────── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1.5px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important; color: #e2e8f0 !important;
        padding: 0.95rem 1.2rem !important; font-size: 1.1rem !important; font-weight: 500 !important;
    }
    /* input focus — form 포함, 모든 기본 테두리 제거 후 연한 초록만 */
    [data-baseweb="input"] {
        border: 1.5px solid rgba(255,255,255,0.15) !important;
        box-shadow: none !important;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    [data-baseweb="input"]:focus-within {
        border: 1.5px solid #6ee7b7 !important;
        box-shadow: 0 0 0 3px rgba(110,231,183,0.15) !important;
    }
    input:focus, input:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    /* st.form 기본 빨간 테두리 제거 */
    [data-testid="stForm"] [data-baseweb="input"]:focus-within {
        border: 1.5px solid #6ee7b7 !important;
        box-shadow: 0 0 0 3px rgba(110,231,183,0.15) !important;
    }
    [data-testid="stFormSubmitButton"] ~ div [data-baseweb="input"] { box-shadow: none !important; }
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

# ── 채팅 히스토리 초기화 ──────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# ── 카카오톡 스타일 채팅 렌더링 (components.v1.html 사용) ────
def render_chat_history():
    import streamlit.components.v1 as components
    if not st.session_state.chat_history:
        return

    msgs_html = ""
    for msg in st.session_state.chat_history:
        role = msg["role"]
        text = msg["text"]
        if role == "user":
            msgs_html += f"""
            <div class="msg-row me">
              <div class="avatar me">🙋</div>
              <div class="bubble-col">
                <div class="bubble me">{text}</div>
              </div>
            </div>"""
        else:
            msgs_html += f"""
            <div class="msg-row bot">
              <div class="avatar bot">🤖</div>
              <div class="bubble-col">
                <div class="sender-name">어시스턴트</div>
                <div class="bubble bot">{text}</div>
              </div>
            </div>"""

    html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        font-family: 'Noto Sans KR', sans-serif;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1rem;
        overflow-y: auto;
        height: 100%;
      }}
      .msg-row {{ display: flex; margin-bottom: 0.85rem; align-items: flex-end; gap: 0.5rem; }}
      .msg-row.me  {{ flex-direction: row-reverse; }}
      .msg-row.bot {{ flex-direction: row; }}
      .avatar {{
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
      }}
      .avatar.me  {{ background: linear-gradient(135deg,#a78bfa,#60a5fa); }}
      .avatar.bot {{ background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2); }}
      .bubble-col {{ display: flex; flex-direction: column; max-width: 70%; }}
      .msg-row.me  .bubble-col {{ align-items: flex-end; }}
      .msg-row.bot .bubble-col {{ align-items: flex-start; }}
      .sender-name {{ font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.22rem; font-weight: 600; }}
      .bubble {{
        padding: 0.65rem 0.95rem; border-radius: 18px;
        font-size: 0.95rem; line-height: 1.55; word-break: break-word;
      }}
      .bubble.me {{
        background: linear-gradient(135deg,#a78bfa,#60a5fa);
        color: white; font-weight: 500;
        border-bottom-right-radius: 4px;
      }}
      .bubble.bot {{
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.15);
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
      }}
    </style>
    <div id="chat">{msgs_html}</div>
    <script>
      window.onload = function() {{
        var c = document.getElementById("chat");
        c.scrollTop = c.scrollHeight;
      }};
    </script>
    """
    n = len(st.session_state.chat_history)
    components.html(html, height=min(80 + n * 75, 320), scrolling=True)

# ── 정책 카드 렌더링 (가로 배치용 - 세로형) ────────────────
def render_policy_card_vertical(rank, title, policy_type, reason, content):
    rank_class  = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "silver")
    rank_label  = {1: "🥇 1순위", 2: "🥈 2순위", 3: "🥉 3순위"}.get(rank, f"{rank}순위")
    type_color  = "#93c5fd" if policy_type == "주거" else "#34d399"
    type_label  = "🏘️ 주거 정책" if policy_type == "주거" else "💳 금융 정책"

    card_html = f"""
    <div class="policy-card">
        <div class="rank-badge {rank_class}">{rank_label}</div>
        <div style="color:{type_color};font-size:0.85rem;font-weight:700;margin-bottom:0.4rem;">{type_label}</div>
        <div class="policy-card-title">{title}</div>
        <div class="policy-section">
            <span class="policy-section-label">📌 선정 이유</span>
            <span class="policy-section-value">{reason}</span>
        </div>
        <div class="policy-section">
            <span class="policy-section-label">📄 핵심 내용</span>
            <span class="policy-section-value">{content}</span>
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
    summary   = data.get("summary", "")
    hero_html = f"""
    <div class="report-hero">
        <div class="report-hero-title">📊 맞춤형 정책 종합 보고서</div>
        <div style="color:#94a3b8;font-size:0.95rem;">AI가 분석한 당신을 위한 최적의 정책 가이드</div>
        <div class="report-hero-summary">{summary}</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    policy_analysis = data.get("policy_analysis", [])
    if policy_analysis:
        # ── 📋 정책별 분석 요약 테이블 ─────────────────────
        rows = "".join([
            f'<tr>'
            f'<td><strong style="color:#e2e8f0">{p.get("title","-")}</strong></td>'
            f'<td style="color:#{"34d399" if p.get("type")=="금융" else "60a5fa"}">{p.get("type","-")}</td>'
            f'<td>{p.get("core","-")}</td>'
            f'</tr>'
            for p in policy_analysis
        ])
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">📋 정책별 분석 요약</div>
            <table class="report-table">
                <thead><tr><th style="width:28%">정책명</th><th style="width:12%">유형</th><th>핵심 내용</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── 장단점: 가로 3열 나란히 ────────────────────────
        cols = st.columns(len(policy_analysis)) if len(policy_analysis) > 1 else [st.container()]
        for col, p in zip(cols, policy_analysis):
            pros = render_list_html(p.get("pros", []))
            cons = render_list_html(p.get("cons", []))
            type_color = "#34d399" if p.get("type") == "금융" else "#60a5fa"
            with col:
                st.markdown(f"""
                <div class="report-section-card" style="height:100%;">
                    <div class="report-section-title" style="font-size:1.05rem;">
                        <span style="color:{type_color}">▸</span> {p.get("title", "-")}
                    </div>
                    <div class="pros-box" style="margin-bottom:0.8rem;">
                        <div class="pros-title">✅ 장점</div>{pros}
                    </div>
                    <div class="cons-box">
                        <div class="cons-title">⚠️ 단점/유의사항</div>{cons}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── 정책 조합 전략 ─────────────────────────────────────
    combination = data.get("combination", "")
    if combination:
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">🔗 정책 조합 전략</div>
            <div class="strategy-box">{combination}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 주의사항 및 리스크 ─────────────────────────────────
    risks = data.get("risks", "")
    if risks:
        risks_html = render_list_html(risks) if isinstance(risks, list) else risks
        st.markdown(f"""
        <div class="report-section-card">
            <div class="report-section-title">⚠️ 주의사항 및 리스크</div>
            <div class="warning-box">{risks_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 종합 추천 및 행동 계획 ─────────────────────────────
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

# ── 채팅 히스토리 영역 ────────────────────────────────────
render_chat_history()

# ── 입력 폼 ───────────────────────────────────────────────
with st.form(key="search_form", clear_on_submit=True):
    col_input, col_btn = st.columns([0.8, 0.2])
    with col_input:
        query = st.text_input(
            label="질문",
            placeholder="질문을 입력해주세요.",
            label_visibility="collapsed",
            key="query_input"
        )
    with col_btn:
        search = st.form_submit_button("🔍 검색", use_container_width=True)

if search:
    if not query.strip():
        st.warning("질문을 입력해주세요.")
    else:
        # 사용자 메시지 채팅에 추가
        st.session_state.chat_history.append({"role": "user", "text": query})

        finished, result = ask_llm(query)

        # ── 추가 정보 요청 ────────────────────────────────
        if not finished:
            st.session_state.chat_history.append({"role": "bot", "text": result})
            st.rerun()

        # ── 검색 준비 완료: session_state에 저장 후 rerun ─
        else:
            st.session_state.chat_history.append({
                "role": "bot",
                "text": "✅ 입력하신 정보로 맞춤 정책을 검색하고 있어요. 잠시만 기다려 주세요!"
            })
            st.session_state["pending_result"] = result
            st.rerun()

# ── rerun 후: pending_result 가 있으면 결과 출력 ──────────
if "pending_result" in st.session_state:
    result = st.session_state.pop("pending_result")
    housing_query = result["housing_query"]
    finance_query = result["finance_query"]

    # ── TOP3 추천 ─────────────────────────────────────────
    st.markdown('<div class="section-header">✨ 맞춤 추천 정책 TOP3</div>', unsafe_allow_html=True)

    with st.spinner("정책을 검색하고 있습니다..."):
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

    top3_sorted = sorted(top3_list, key=lambda x: x.get("rank", 99))[:3]
    if top3_sorted:
        cols = st.columns(3)
        for idx, policy in enumerate(top3_sorted):
            with cols[idx]:
                render_policy_card_vertical(
                    rank        = policy.get("rank", idx + 1),
                    title       = policy.get("title", "-"),
                    policy_type = policy.get("type", "-"),
                    reason      = policy.get("reason", "-"),
                    content     = policy.get("content", "-")
                )

    st.divider()

    # ── 종합 보고서 ───────────────────────────────────────
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