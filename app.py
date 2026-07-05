import os
import re
from typing import List, Dict, Any, TypedDict, Literal, Annotated
import operator

import streamlit as st
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

st.set_page_config(page_title="하현 QA 챗봇", page_icon="🧴")

# ---------- 설정 ----------
def get_client():
    key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        return None
    return OpenAI(api_key=key)

client = get_client()
MODEL = "gpt-4.1-mini"

# ---------- 지식베이스 ----------
KNOWLEDGE_BASE = [
    {
        "title": "핵심 성분",
        "content": "PDRN 10%(100,000ppm), 나이아신아마이드 5%, 알란토인 5%, "
                    "8종 히알루론산 복합체, 트라넥삼산 2%, 아데노신 500ppm(제품 사용 전 대표님 확인 필요), "
                    "비타민C 유도체, 베르가못 오일 함유.",
    },
    {
        "title": "개발/제조 정보",
        "content": "강남 소재 피부과 '플래티넘 의원'과 공동개발/협업. "
                    "제조사는 비티바이오테라퓨틱스(주). 7월 23일 트레이드쇼 전시용 앰플.",
    },
    {
        "title": "컴플라이언스 원칙",
        "content": "경쟁사 직접 비교, 의약품 수준 효능 주장, '완치', '즉시 효과', '부작용 없음' 표현은 "
                    "화장품법상 과장광고에 해당할 수 있어 사용을 지양해야 한다.",
    },
]

RISKY_CLAIM_KEYWORDS = ["최고", "1위", "완치", "즉시 효과", "타사 대비", "부작용 없음", "의약품 수준"]


def simple_retrieve(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    query_terms = set(re.findall(r"[가-힣A-Za-z0-9]+", query.lower()))
    scored = []
    for doc in KNOWLEDGE_BASE:
        text = (doc["title"] + " " + doc["content"]).lower()
        doc_terms = set(re.findall(r"[가-힣A-Za-z0-9]+", text))
        score = len(query_terms & doc_terms)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]] or KNOWLEDGE_BASE[:top_k]


def check_compliance(text: str) -> List[str]:
    return [kw for kw in RISKY_CLAIM_KEYWORDS if kw in text]


def call_llm_text(system: str, user: str) -> str:
    response = client.responses.create(
        model=MODEL,
        instructions=system,
        input=user,
        temperature=0.3,
    )
    return response.output_text


# ---------- State ----------
class ChatState(TypedDict, total=False):
    user_query: str
    route: str
    retrieved_docs: List[Dict[str, Any]]
    draft_answer: str
    compliance_flags: List[str]
    final_answer: str
    trace: Annotated[List[str], operator.add]


# ---------- Nodes ----------
def classify_node(state: ChatState) -> Dict[str, Any]:
    query = state["user_query"]
    if any(k in query for k in ["그래프", "구조", "노드", "flow", "워크플로우"]):
        route = "visualize"
    elif any(k in query for k in ["문구", "카피", "마케팅", "홍보", "써줘", "작성"]):
        route = "copy_gen"
    elif any(k in query for k in ["비교", "타사", "경쟁사"]):
        route = "compliance_check"
    elif any(k in query for k in ["성분", "효능", "함량", "제조", "개발", "몇 %", "몇 프로"]):
        route = "rag"
    else:
        route = "chat"
    return {"route": route, "trace": [f"classify_node: route={route}"]}


def rag_node(state: ChatState) -> Dict[str, Any]:
    docs = simple_retrieve(state["user_query"], top_k=2)
    context = "\n\n".join([f"[{d['title']}] {d['content']}" for d in docs])
    system = "너는 하현(Hahyeon) 화장품 브랜드의 제품 정보 조교다. 문서 근거만으로 답하고, 근거 없으면 모른다고 답하라."
    user = f"질문: {state['user_query']}\n\n근거 문서:\n{context}"
    answer = call_llm_text(system, user)
    return {"retrieved_docs": docs, "draft_answer": answer, "trace": [f"rag_node: docs={len(docs)}"]}


def copy_gen_node(state: ChatState) -> Dict[str, Any]:
    docs = simple_retrieve(state["user_query"], top_k=2)
    context = "\n\n".join([f"[{d['title']}] {d['content']}" for d in docs])
    system = "너는 화장품 마케팅 카피라이터다. 근거 문서에 있는 성분/사실만 언급해서 문구를 작성하라."
    user = f"요청: {state['user_query']}\n\n근거 문서:\n{context}"
    draft = call_llm_text(system, user)
    flags = check_compliance(draft)
    if flags:
        draft += f"\n\n⚠️ 검토 필요 표현: {flags} (화장품법상 과장광고 소지 있음)"
    return {"draft_answer": draft, "compliance_flags": flags, "trace": [f"copy_gen_node: flags={flags}"]}


def compliance_check_node(state: ChatState) -> Dict[str, Any]:
    answer = (
        "경쟁사와의 직접 비교, 의약품 수준 효능 주장, '완치'·'즉시 효과'·'부작용 없음' 같은 표현은 "
        "화장품법상 과장광고로 간주될 수 있어 사용을 지양해야 합니다.\n\n"
        "대신 자사 제품의 성분·함량 등 객관적 사실 위주로 표현하는 것을 권장합니다."
    )
    return {"draft_answer": answer, "trace": ["compliance_check_node: policy_reminder"]}


def visualize_node(state: ChatState) -> Dict[str, Any]:
    return {"draft_answer": "__SHOW_GRAPH__", "trace": ["visualize_node: render_graph"]}


def chat_node(state: ChatState) -> Dict[str, Any]:
    system = "너는 하현 화장품 브랜드 업무를 돕는 친절한 AI 비서다. 간결하게 답하라."
    answer = call_llm_text(system, state["user_query"])
    return {"draft_answer": answer, "trace": ["chat_node: general"]}


def review_node(state: ChatState) -> Dict[str, Any]:
    draft = state.get("draft_answer", "")
    notes = []
    if state.get("compliance_flags"):
        notes.append(f"과장광고 위험 표현 감지: {state['compliance_flags']}")
    if not notes:
        notes.append("검토 통과")
    return {"final_answer": draft, "trace": [f"review_node: {' / '.join(notes)}"]}


def route_after_classify(state: ChatState) -> Literal["rag", "copy_gen", "compliance_check", "visualize", "chat"]:
    return state["route"]


# ---------- Graph build ----------
@st.cache_resource
def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("classify", classify_node)
    builder.add_node("rag", rag_node)
    builder.add_node("copy_gen", copy_gen_node)
    builder.add_node("compliance_check", compliance_check_node)
    builder.add_node("visualize", visualize_node)
    builder.add_node("chat", chat_node)
    builder.add_node("review", review_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "rag": "rag",
            "copy_gen": "copy_gen",
            "compliance_check": "compliance_check",
            "visualize": "visualize",
            "chat": "chat",
        },
    )
    builder.add_edge("rag", "review")
    builder.add_edge("copy_gen", "review")
    builder.add_edge("compliance_check", "review")
    builder.add_edge("visualize", "review")
    builder.add_edge("chat", "review")
    builder.add_edge("review", END)

    return builder.compile()


graph = build_graph()

# ---------- UI ----------
st.title("🧴 하현 QA 챗봇")
st.caption("성분·마케팅 문구·컴플라이언스 체크 · '그래프 보여줘'라고 물어보면 워크플로우 구조를 볼 수 있어요.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["content"] == "__SHOW_GRAPH__":
            st.image(graph.get_graph().draw_mermaid_png())
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요 (예: PDRN 함량이 몇 %야?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if client is None:
        answer = "⚠️ OPENAI_API_KEY가 설정되지 않았습니다."
    else:
        with st.spinner("생각 중..."):
            result = graph.invoke({"user_query": prompt})
            answer = result["final_answer"]

    with st.chat_message("assistant"):
        if answer == "__SHOW_GRAPH__":
            st.write("현재 챗봇의 워크플로우 구조입니다:")
            st.image(graph.get_graph().draw_mermaid_png())
        else:
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
