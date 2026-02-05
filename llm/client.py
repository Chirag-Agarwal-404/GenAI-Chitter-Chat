import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()



class LLMClient:
    def __init__(self):
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.5,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def chat(self, messages):
        """
        messages: list of dicts
        [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
        """

        lc_messages = [
            SystemMessage(
                content=(
                    "You are GenAI Chitter Chat, a helpful enterprise AI assistant.\n\n"
                    "Your capabilities:\n"
                    "- You can have normal conversational chats.\n"
                    "- You can answer questions using documents uploaded in the current chat.\n"
                    "- Documents are chat-specific and may include PDFs.\n\n"
                    "Behavior guidelines:\n"
                    "- Be clear, concise, and friendly.\n"
                    "- If a user asks about a document but no document is available, politely ask them to upload it.\n"
                    "- Do NOT assume or hallucinate document content.\n"
                    "- If the question is general and does not depend on documents, answer normally.\n\n"
                    "Always respond in a professional, helpful tone."
                )
            )
        ]


        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        response = self.llm.invoke(lc_messages)
        return response.content
