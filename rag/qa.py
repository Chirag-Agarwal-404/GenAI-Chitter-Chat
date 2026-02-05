from langchain_core.messages import SystemMessage, HumanMessage
from llm.client import LLMClient
from langchain_community.vectorstores import FAISS


def answer_from_documents(question,  faiss_index, k=4):
    llm_client = LLMClient()
    if faiss_index is None :
        messages = [
            SystemMessage(
                content=(
                    "The user has not uploaded any documents.\n"
                    "Politely inform the user that document-based answers are unavailable "
                    "and ask them to upload a PDF through given upload document option at the top of the chat to enable document Q&A.\n"
                    "Do NOT attempt to answer the question from general knowledge."
                )
            ),
            HumanMessage(content=question),
        ]

        return llm_client.llm.invoke(messages).content
    
    
    docs = faiss_index.similarity_search(question, k=k)

    context = "\n\n".join(
        f"Source {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    messages = [
        SystemMessage(
            content=(
                "You are answering questions using ONLY the provided document context.\n"
                "If the answer is not in the context, say you don't know."
            )
        ),
        HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            )
        )
    ]

    return llm_client.llm.invoke(messages).content
