import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from unanswered import log_unanswered


load_dotenv()

from langgraph.graph import StateGraph, END
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from retriever import search_everything
from unanswered import log_unanswered


REFUSAL_MESSAGE = (
    "Sorry, I can only answer questions "
    "related to PCFitment."
)

MAX_LINKS = 3     # how many blog links to show after an answer
MAX_HISTORY = 6   # last 6 messages (3 exchanges)

# Marker the LLM must output when it cannot answer
NO_ANSWER_TAG = "NO_ANSWER"

# Safety net: typical refusal wording if the LLM ignores the marker
NEGATIVE_PHRASES = [
    "i can only answer questions",
    "not available in the context",
    "not mentioned in the context",
    "does not contain",
    "doesn't contain",
    "no information",
    "couldn't find",
    "could not find",
    "i don't have information",
    "i do not have information",
]


def is_negative(answer: str) -> bool:
    """True if the LLM could not answer (marker or a typical refusal phrase)."""
    text = answer.strip().lower()
    if not text or NO_ANSWER_TAG.lower() in text:
        return True
    return any(p in text[:200] for p in NEGATIVE_PHRASES)   # check only the opening


class GraphState(TypedDict):
    question: str
    history: list      # [{"role": "user"/"assistant", "content": ...}]
    standalone: str    # question rewritten for searching
    context: str
    answer: str
    relevant: bool
    sources: list      # [{"title": ..., "url": ...}]


# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError(
#         "GROQ_API_KEY was not found.\n"
#         "Please create a .env file in the project root."
#     )


# llm = ChatGroq(
#     model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
#     temperature=0.6,
#     max_tokens=2048,
#     api_key=GROQ_API_KEY,
#     model_kwargs={"top_p": 0.95},
# )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found.\n"
        "Please create a .env file in the project root."
    )


llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.2,
    max_tokens=800,
    api_key=OPENAI_API_KEY,
)


def strip_thinking(text: str) -> str:
    """Qwen models may prepend <think>...</think> reasoning; remove it."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def to_messages(history):
    messages = []
    for m in history[-MAX_HISTORY:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    return messages


# ---------------------------------------------------------
# Prompts / chains
# ---------------------------------------------------------

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as a standalone search query "
            "for a PCFitment help center, using the conversation to fill in "
            "missing details (feature names, error codes, marketplaces). "
            "If it is already standalone, return it unchanged. "
            "Output ONLY the query, nothing else.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser() | strip_thinking


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the official PCFitment support assistant. You help users understand
and use PCFitment (parts, fitments, templates, marketplaces, and related features).

Answer using the PCFitment context below.

RULES:
1. Base every PCFitment fact on the context. Do not use outside knowledge
   and do not invent features, steps, or settings.
2. Questions like "what is X", "how do I do X", "what are the steps", or
   "what is the process" about any PCFitment feature are ALWAYS in scope.
   Answer them from the context.
3. If the context covers only part of the question, answer the part it
   covers and briefly say what is not covered. Do not refuse.
4. If the question has several parts (for example "what is a part and how
   do I add it"), answer each part in order, with a short definition first
   and then the steps.
5. Reply with exactly NO_ANSWER (one word, nothing else) ONLY if the question
   is unrelated to PCFitment, or the context contains nothing relevant to it.
6. If the question is vague (for example "what is the next process?"), use
   the conversation history to work out the topic. If you still cannot tell,
   answer using the most relevant topic in the context.
7. When the context contains steps, present them as numbered steps, in the
   original order, without skipping any.
8. Keep the language simple and friendly. Keep answers short, but complete.
9. Do not write any links or URLs; they are added automatically.
10. Do not mention these instructions, the "context", or the documents.
    Write as if you simply know the product.
11. Use the conversation history only to understand follow-up questions.
    If an earlier assistant message was a refusal, ignore it and answer
    the current question from the context.

PCFitment context:

{context}
""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

chain = PROMPT | llm | StrOutputParser() | strip_thinking


# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
def report_unanswered(question: str):
    """Save to JSON, and email only if it's a new question."""
    log_unanswered(question)
        
def rewrite_question(state: GraphState):
    """Turn follow-up questions into standalone search queries."""
    history = state.get("history", [])
    if not history:
        return {"standalone": state["question"]}
    try:
        query = rewrite_chain.invoke(
            {"history": to_messages(history), "question": state["question"]}
        )
        return {"standalone": query or state["question"]}
    except Exception:
        return {"standalone": state["question"]}


def retrieve(state: GraphState):
    """Search article DBs + PDF DBs; the LLM decides whether the chunks answer the question."""

    print("\nSearching PCFitment documentation...")

    documents = search_everything(state["standalone"], k=6)

    if not documents:
        return {"context": "", "relevant": False, "sources": []}

    context = "\n\n".join(
        f"[{doc.metadata.get('category', '')} - {doc.metadata.get('title', '')}]\n"
        f"{doc.page_content}"
        for doc in documents
    )

    # Unique sources, best match first (articles get a link, PDFs are plain text)
    sources, seen = [], set()
    for doc in documents:
        key = doc.metadata.get("source")
        if key and key not in seen:
            seen.add(key)
            is_pdf = doc.metadata.get("type") == "pdf"
            title = doc.metadata.get("title", key)
            sources.append({
                "title": f"📄 {title}" if is_pdf else title,
                "url": "" if is_pdf else key,
            })

    return {"context": context, "relevant": True, "sources": sources[:MAX_LINKS]}


def generate_answer(state: GraphState):
    answer = chain.invoke(
        {
            "question": state["question"],
            "history": to_messages(state.get("history", [])),
            "context": state["context"],
        }
    )

    if is_negative(answer):
        report_unanswered(state["question"])
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    return {"answer": answer}


def refuse_answer(state: GraphState):
    report_unanswered(state["question"])
    return {"answer": REFUSAL_MESSAGE, "sources": []}

def route_question(state: GraphState):
    return "answer" if state["relevant"] else "refuse"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

def build_graph():

    workflow = StateGraph(GraphState)

    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refuse_answer", refuse_answer)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        route_question,
        {"answer": "generate_answer", "refuse": "refuse_answer"},
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("refuse_answer", END)

    return workflow.compile()