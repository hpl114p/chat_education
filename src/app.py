import streamlit as st
import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION\.env.init")
gemini_key = os.getenv("gemini_key")

# Import các thành phần từ chain.py
from rag.chain import LLMHandler, VectorDatabase, QuestionAnsweringChain

# Cấu hình trang
st.set_page_config(page_title="🐐💬 Gemini Chatbot (Streaming)")

# Sidebar
with st.sidebar:
    st.title("🐐💬 Chatbot HaUI")
    st.write("Chatbot trợ giúp thông tin cho phòng đào tạo HaUI.")
    st.markdown("📖 Xem hướng dẫn xây dựng chatbot tại [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)")

# Khởi tạo phiên chatbot nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

# Hàm xóa lịch sử chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]
    st.session_state.chat_history = []

# Nút xóa chat
st.sidebar.button('🧹 Xóa lịch sử Chat', on_click=clear_chat_history)

# Khởi tạo các thành phần chỉ một lần
if "qa_chain" not in st.session_state:
    llm_handler = LLMHandler(model_name="gemini-1.5-flash", gemini_key=gemini_key)
    vector_db = VectorDatabase()
    qa_chain = QuestionAnsweringChain(
        llm_handler=llm_handler,
        vector_db=vector_db,
        num_docs=5,
        apply_rewrite=False,
        apply_rerank=False,
        date_impact=0.01
    )
    st.session_state.qa_chain = qa_chain

# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Nhận đầu vào từ người dùng
if user_input := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    # Hiển thị ngay trên giao diện
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Xử lý truy vấn
    response, links = st.session_state.qa_chain.run(user_input)

    # Hiển thị phản hồi
    with st.chat_message("assistant"):
        st.markdown(response)
        if links:
            st.markdown("🔗 **Nguồn tham khảo:**")
            for link in links:
                st.markdown(f"- [{link}]({link})")
    
    # Lưu vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": response})
