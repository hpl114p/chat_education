from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION\.env.init")

gemini_key = os.getenv("gemini_key")

# Khởi tạo chatbot với model Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_key)

# Đặt câu hỏi
question = "AI là gì"
response = llm.invoke(question)

# In câu trả lời
print(response)

