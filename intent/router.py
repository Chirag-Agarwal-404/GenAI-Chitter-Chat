from typing import Literal, TypedDict 
from langchain_core.messages import SystemMessage, HumanMessage
from llm.client import LLMClient
from pydantic import BaseModel, Field

# -------------------------------------------------
# Structured output schema
# -------------------------------------------------
class IntentResult(BaseModel):  
    intent: Literal["GENERAL_CHAT", "DOCUMENT_QA"] = Field(
        ...,
        description=(
            "The intent of the user's message.\n\n"
            "GENERAL_CHAT:\n"
            "- Greetings, small talk, opinions, jokes\n"
            "- General questions not requiring uploaded documents\n\n"
            "DOCUMENT_QA:\n"
            "- Questions that require information from uploaded documents\n"
            "- Asking about facts, summaries, or references from the documents"
        )
    )


# -------------------------------------------------
# Intent Router
# -------------------------------------------------
class IntentRouter:
    def __init__(self):
        self.llm_client = LLMClient()

        self.structured_llm = self.llm_client.llm.with_structured_output(
            IntentResult
        )

    def detect_intent(self, user_input: str,has_document: bool) -> str:
        """
        Returns:    
        - GENERAL_CHAT
        - DOCUMENT_QA
        """

        # if not has_document:
        #     return "GENERAL_CHAT"

        system_prompt = (
            "You are an intent classification engine.\n"
            "Decide whether the user's question requires using documents uploaded in a chat.\n\n"
            "CRITICAL RULE:\n"
            "- If the user has NOT uploaded any documents, you MUST output GENERAL_CHAT.\n\n"
            "Rules:\n"
            "- DOCUMENT_QA → questions about uploaded document content, facts, summaries, references\n"
            "- GENERAL_CHAT → greetings, opinions, jokes, general discussion\n\n"
            "Output must be valid JSON with exactly one key: intent."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        result: IntentResult = self.structured_llm.invoke(messages)

        return result.intent
