from graph.state import GraphState
from intent.router import IntentRouter
from llm.client import LLMClient
from rag.qa import answer_from_documents
from langchain.messages import SystemMessage, HumanMessage


intent_router = IntentRouter()
llm_client = LLMClient()


# -----------------------------
# Intent detection node
# -----------------------------
def intent_node(state: GraphState) -> GraphState:
    intent = intent_router.detect_intent(
        user_input=state["user_input"],
        has_document=state["has_document"]
    )
    state["intent"] = intent
    return state


# -----------------------------
# General chat node (LLM)
# -----------------------------
def general_chat_node(state: GraphState) -> GraphState:
    user_input = state.get("user_input", "")

    llm = LLMClient().llm

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_input),
    ]
    state["response"] = llm.invoke(messages).content
    return state

# -----------------------------
# Document QA node (STUB)
# -----------------------------
def document_qa_node(state: GraphState) -> GraphState:
    question = state.get("user_input")
    faiss_index = state.get("faiss")

    answer = answer_from_documents(question, faiss_index)

    state["response"] = answer
    return state
