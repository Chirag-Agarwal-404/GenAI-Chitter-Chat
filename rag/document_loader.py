import os
import tempfile
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_with_langchain(uploaded_file) -> List[Document]:
    """
    Load PDF using LangChain PyPDFLoader
    - User-safe
    - No shared files
    - Auto cleanup
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        prefix="genai_chitter_"
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.remove(tmp_path)

    return documents

