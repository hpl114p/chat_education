from chain import LLMHandler, VectorDatabase, QuestionAnsweringChain
import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION\.env.init")
gemini_key = os.getenv("gemini_pro_key")

path_faiss_index=r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\faiss_v2"

llm_handler = LLMHandler(model_name="gemini-2.5-pro-exp-03-25", gemini_key=gemini_key)
vector_store = VectorDatabase(path_faiss_index=path_faiss_index)
qa_chain = QuestionAnsweringChain(
    llm_handler=llm_handler,
    vector_db=vector_store,
    num_docs=5,
    apply_rewrite=False,
    apply_rerank=False,
    date_impact=0.01
)

query = "Mã ngành kỹ thuật phần mềm trường Đại học Công nghiệp Hà Nội là gì?"

response, link = qa_chain.run(query)

print("------------------------")
print(response)
print(link)