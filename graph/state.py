from typing import TypedDict, List, Optional, Literal
from langchain_community.vectorstores import FAISS


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str

class GraphState(TypedDict):
    user_input: str
    messages: List[Message]
    intent: Optional[str]
    response: Optional[str]
    has_document: bool
    faiss : Optional[FAISS]
