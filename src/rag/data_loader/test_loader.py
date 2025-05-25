from load_data import RecursiveChucking, read_document_json
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4


path_js = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\outputs.json"

model_name = "hiieu/halong_embedding"

embeddings = HuggingFaceEmbeddings(
    model_name = model_name
)

docs = read_document_json(path=path_js)

splitter = RecursiveChucking()

langchain_docs = splitter.split_documents(docs=docs)
print(f"Total docs after splitting: {len(langchain_docs)}")
print(f"Sample doc length: {len(langchain_docs[0].page_content)}")

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

uuids = [str(uuid4()) for _ in range(len(langchain_docs))]

vector_store.add_documents(documents=langchain_docs, ids=uuids)

vector_store.save_local(r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\faiss")

results = vector_store.similarity_search(
    "kết quả xét tuyển đào tạo từ xa trình độ đại học đợt 2 năm 2025",
    k=3
)

print(results)
print("End!")