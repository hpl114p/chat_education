from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import os

class FAISSVectorDBHandler:
    def __init__(
        self, 
        model_name = "hiieu/halong_embedding", 
        path_faiss_index = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\faiss"
    ):
        self.path = path_faiss_index
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name = self.model_name
        )
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_docs(self, docs: list[Document]):
        uuids = [str(uuid4()) for _ in range(len(docs))]
        self.vector_store.add_documents(documents=docs, ids=uuids)
        self.vector_store.save_local(self.path)
        print(f"Success add documents, vector store saved to: {self.path}")

    


