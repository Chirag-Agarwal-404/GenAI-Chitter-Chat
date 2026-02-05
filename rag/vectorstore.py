import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_faiss_index(documents):
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    return FAISS.from_documents(documents, embeddings)
