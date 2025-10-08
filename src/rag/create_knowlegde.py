from .data_loader.load_data import RecursiveChucking, read_document_json
from .vector_manager.vectorstore import FAISSVectorDBHandler
from .chain import VectorDatabase
from data_manager.process_data import process_file
from langchain.schema import Document as langchain_doc
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def add_documents(file_name):
    path = os.path.join(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\collections", 
                               f"{file_name}_outputs.json")
    print(f"Reading documents from: {path}")
    # from json to langchain documents
    docs = read_document_json(path=path)
    # print(docs)

    splitter = RecursiveChucking()
    langchain_docs = splitter.split_documents(docs=docs)
    print(f"Total docs after splitting: {len(langchain_docs)}")
    print(f"Sample doc length: {len(langchain_docs[0].page_content)}")

    path_faiss_index = os.path.join(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\vector_collections", 
                               f"{file_name}")
    vector_store = FAISSVectorDBHandler(path_faiss_index=path_faiss_index)
    vector_store.add_docs(langchain_docs)
    print(f"Saving vector store to: {path_faiss_index}")

def add_docs_from_files(file_name):
    # Lấy đường dẫn lưu file gốc lấy về từ streamlit
    raw_data_dir = os.path.join(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\raw_data",
                                f"{file_name}")
    documents = process_file(raw_data_dir)
    documents = " ".join(documents)

    # --- Lưu document sau khi process ---
    save_dir = r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\raw_data"
    os.makedirs(save_dir, exist_ok=True)
    text_file_path = os.path.join(save_dir, f"{file_name}_processed.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(documents)

    # --- Split ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = text_splitter.split_text(documents)
    processed_documents = [langchain_doc(page_content=text) for text in texts]

    # --- Lưu vào Faiss ---
    path_faiss_index = os.path.join(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\vector_collections", 
                               f"{file_name}")
    vector_store = FAISSVectorDBHandler(path_faiss_index=path_faiss_index)
    vector_store.add_docs(processed_documents)
    print(f"Saving vector store to: {path_faiss_index}")


# for testing
def retriever_query(file_name):
    path_faiss_index = os.path.join(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\vector_collections", 
                               f"{file_name}")
    vector_store = VectorDatabase(path_faiss_index=path_faiss_index)
    retriever = vector_store.get_retriever()
    return retriever.as_retriever(search_kwargs={"k": 5})

if __name__=='__main__':
    file_name = 'haui'
    add_documents(file_name)
    
    # retriever = retriever_query(file_name)
    # query = "StackAI là gì?"
    # relevant = retriever.invoke(query)
    # print(relevant)