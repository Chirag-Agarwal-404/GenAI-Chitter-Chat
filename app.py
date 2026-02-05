import uuid
import streamlit as st
from dotenv import load_dotenv

from rag.document_loader import load_pdf_with_langchain
from rag.splitter import split_documents
from rag.local_vectorstore import build_faiss_index

from graph.builder import build_graph

# -----------------------------
# Env
# -----------------------------
load_dotenv()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="GenAI Chitter Chat",
    page_icon="®",
    layout="wide"
)

# -----------------------------
# Stable browser session id (MVP isolation)
# -----------------------------
if "browser_session_id" not in st.session_state:
    st.session_state.browser_session_id = str(uuid.uuid4())

browser_session_id = st.session_state.browser_session_id

# -----------------------------
# User-specific chat state (keyed by browser session)
# -----------------------------
if "users" not in st.session_state:
    st.session_state.users = {}

if browser_session_id not in st.session_state.users:
    st.session_state.users[browser_session_id] = {
        "conversations": {},
        "current_conversation_id": None
    }

user_state = st.session_state.users[browser_session_id]

def create_new_chat() -> str:
    cid = str(uuid.uuid4())
    user_state["conversations"][cid] = {
        "title": None,                 # <-- first user message becomes title
        "messages": [],
        "documents": [],
        "faiss": None,
        "processed_file_keys": set(),  # <-- prevent duplicate processing per chat
    }
    user_state["current_conversation_id"] = cid
    return cid

# Create first chat if none exists
if not user_state["current_conversation_id"]:
    create_new_chat()

current_chat_id = user_state["current_conversation_id"]
current_chat = user_state["conversations"][current_chat_id]

# -----------------------------
# Sidebar: Chat History
# -----------------------------
with st.sidebar:
    st.title("💬 Chat History")
    st.divider()

    if st.button("➕ New Chat"):
        create_new_chat()
        st.rerun()

    # Show newest chats first (nicer UX)
    conversation_ids = list(user_state["conversations"].keys())[::-1]

    for cid in conversation_ids:
        chat_obj = user_state["conversations"][cid]
        title = chat_obj.get("title") or f"New Chat ({cid[:8]})"
        if st.button(title, key=f"chat_{cid}"):
            user_state["current_conversation_id"] = cid
            st.rerun()

# -----------------------------
# Main UI
# -----------------------------
st.title("®GenAI Chitter Chat")

# -----------------------------
# Attach PDFs (per-chat)
# -----------------------------
attach_col, spacer_col = st.columns([1, 6])

with attach_col:
    st.markdown("📎 **Attach PDFs**")

with spacer_col:
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"upload_{current_chat_id}",  # <-- per-chat uploader
        label_visibility="collapsed"
    )

if uploaded_files:
    current_chat.setdefault("faiss", None)
    current_chat.setdefault("processed_file_keys", set())
    current_chat.setdefault("documents", [])

    # Only process new files (ignore duplicates)
    new_files = []
    for f in uploaded_files:
        file_key = (f.name, f.size, f.type)
        if file_key not in current_chat["processed_file_keys"]:
            new_files.append((file_key, f))

    if new_files:
        with st.spinner("Processing new documents..."):
            all_new_docs = []

            for file_key, f in new_files:
                docs = load_pdf_with_langchain(f)     # pages/docs from PDF
                all_new_docs.extend(docs)
                current_chat["processed_file_keys"].add(file_key)

            current_chat["documents"].extend(all_new_docs)

            # Chunk only the new docs
            new_chunks = split_documents(all_new_docs)

            # Build or incrementally update FAISS
            if current_chat["faiss"] is None:
                current_chat["faiss"] = build_faiss_index(new_chunks)
            else:
                current_chat["faiss"].add_documents(new_chunks)

        st.success(
            f"Added {len(new_files)} new file(s) | "
            f"Total pages loaded: {len(current_chat['documents'])} | "
            f"FAISS index updated"
        )
    else:
        st.info("No new PDFs detected (duplicates ignored).")

# -----------------------------
# Show messages
# -----------------------------
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Graph runner
# -----------------------------
chat_graph = build_graph()

def run_chat(user_input: str) -> str:
    faiss_index = current_chat.get("faiss")

    result = chat_graph.invoke({
        "user_input": user_input,
        "messages": current_chat["messages"],
        "faiss": faiss_index,
        "intent": None,
        "response": None,
        "has_document": faiss_index is not None,
    })
    return result["response"]

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask something...")

if prompt:
    # Set chat title from first user message
    if not current_chat.get("title"):
        # Keep it short & clean
        current_chat["title"] = prompt.strip()[:40]

    current_chat["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_chat(prompt)
            st.markdown(response)

    current_chat["messages"].append({"role": "assistant", "content": response})
