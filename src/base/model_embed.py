from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore

model_name = "hiieu/halong_embedding"

embeddings = HuggingFaceEmbeddings(model_name=model_name)

print(embeddings.embed_query("hoang phuc lam"))