import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rag.data_loader.load_data import RecursiveChucking, read_document_json
from vectorstore import FAISSVectorDBHandler


path_js = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\outputs.json"
docs = read_document_json(path=path_js)

splitter = RecursiveChucking()
langchain_docs = splitter.split_documents(docs)

vector_store = FAISSVectorDBHandler()

vector_store.add_docs(langchain_docs)

query = "kết quả xét tuyển đào tạo từ xa trình độ đại học đợt 2 năm 2025"
relevant = vector_store.get_retriever(query=query)
print(relevant)