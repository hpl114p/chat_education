import json
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def read_document_json(path: str) -> dict:
    """
    Reads a JSON file containing documents and returns its content as a dictionary.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        dict: The parsed content of the JSON file.
    """
    try:
        data =pd.read_json(path)
        return data.to_dict(orient='records')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {path}")

class RecursiveChucking:
    def __init__(self, chunk_size=1000, chuck_overlap=100):
        self.MARKDOWN_SEPARATORS = [
            "\n\n\n",  # phân đoạn rất lớn (hiếm gặp, phòng trường hợp đặc biệt)
            "\n\n",    # phân đoạn theo phần lớn (ví dụ: I. THÔNG TIN CHUNG vs II. TUYỂN SINH ĐÀO TẠO)
            "\n- ",    # gạch đầu dòng
            "\n",      # xuống dòng thông thường
            " ",       # cuối cùng chia theo từ nếu không còn gì khác
        ]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,  # The maximum number of characters in a chunk: we selected this value arbitrarily
            chunk_overlap=50,  # The number of characters to overlap between chunks
            add_start_index=True,  # If `True`, includes chunk's start index in metadata
            strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
            separators=self.MARKDOWN_SEPARATORS,
        )

    def split_documents(self, docs: list[dict]):
        RAW_KNOWLEDGE_BASE = [
            Document(
                page_content=doc["content"], 
                metadata={"source": doc["link"], "title": doc["title"], "id": idx}
            )
            for idx, doc in enumerate(docs)
        ]

        langchain_docs = []
        for doc in RAW_KNOWLEDGE_BASE:
            chucks = self.text_splitter.split_documents([doc])
            chucks = [
                Document(
                    page_content=chuck.metadata.get("title", "") + "\n" + chuck.page_content,
                    metadata=chuck.metadata
                )
                for chuck in chucks
            ]
            langchain_docs += chucks

        return langchain_docs
