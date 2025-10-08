import streamlit as st
from rag.chain import LLMHandler, VectorDatabase, QuestionAnsweringChain
from rag.create_knowlegde import add_documents, add_docs_from_files
from data_manager.web_crawler import auto_crawl
import os
import time
from dotenv import load_dotenv
from pathlib import Path

parent_dir = Path(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\vector_collections")
subdirs = [d.name for d in parent_dir.iterdir() if d.is_dir()]
# print(subdirs)

# Cấu hình trang
st.set_page_config(page_title="🐐💬 Gemini Chatbot (Streaming)")
st.title("💬 Assitant AI")

# Hàm khởi tạo ứng dụng
def initialize_app(path_faiss_index=r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\faiss_v3"):
    # Tải biến môi trường
    load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION\.env.init")
    gemini_key = os.getenv("gemini_generate")

    # Khởi tạo các thành phần chỉ một lần
    if "qa_chain" not in st.session_state:
        llm_handler = LLMHandler(model_name="gemini-1.5-flash", gemini_key=gemini_key)
        vector_db = VectorDatabase(path_faiss_index=path_faiss_index)
        qa_chain = QuestionAnsweringChain(
            llm_handler=llm_handler,
            vector_db=vector_db,
            num_docs=5,
            apply_rewrite=False,
            apply_rerank=False,
            date_impact=0.01
        )
        st.session_state.qa_chain = qa_chain

# Hàm xóa lịch sử chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Đây là HaUI Chatbot, trợ lý đắc lực dành cho bạn! Bạn muốn tìm kiếm thông tin về những gì?"}]

def handle_local_file():
    """
    Xử lý khi người dùng chọn tải file từ local
    """
    uploaded_file = st.file_uploader("Chọn file dữ liệu của bạn!")
    if uploaded_file is not None:
        with st.spinner("Đang tải dữ liệu..."):
            try:
                # Lưu file vào thư mục tạm
                UPLOAD_DIR = r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\raw_data"
                os.makedirs(UPLOAD_DIR, exist_ok=True)

                file_name = uploaded_file.name
                # file_name = os.path.splitext(uploaded_file.name)[0]
                file_path = os.path.join(UPLOAD_DIR, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Lưu vào faiss
                add_docs_from_files(file_name=file_name)
                st.success(f"Đã tải dữ liệu thành công vào collection '{file_name}'!")
                subdirs.append(file_name)
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")

def handle_url_input():
    """
    Xử lý khi người dùng chọn crawl URL
    """
    collection_name = st.text_input(
        "Tên collection trong Faiss:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Faiss"
    )

    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    
    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang crawl dữ liệu..."):
            try:
                auto_crawl(url, collection_name, 2, 2)
                add_documents(file_name=collection_name)
                st.success(f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")
    subdirs.append(collection_name)
# Sidebar
with st.sidebar:
    with st.sidebar:
        tabs = st.tabs(["💬 Chatbot", "⚙️ Cài đặt", "ℹ️ Thông tin"])

    with tabs[0]:
        # st.title("🐐💬 AI Assistant")
        st.write("Xin chào! Mình là AI Assistant. Giúp bạn giải đáp thắc mắc, tra cứu thông tin một cách nhanh chóng và chính xác nhất!")
        # st.divider()
        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Các nguồn giúp Assistant đưa ra câu trả lời dựa trên những thông tin quan trọng nhất đối với bạn.",
            ["File Local", "URL trực tiếp"]
        )

        if data_source == "File Local":
            handle_local_file()
        else:
            handle_url_input()

        subdirs = [d.name for d in parent_dir.iterdir() if d.is_dir()]
        st.divider()
        st.header("Chọn cơ sở tri thức")
        vector_source = st.radio(
            "Chọn cơ sở tri thức bạn muốn sử dụng:",
            subdirs
        )
        # print(vector_source)
        path_faiss_index = os.path.join(r"B:\PROJECTS\CHATBOT_EDUCATION\src\data_manager\vector_collections", 
                            f"{vector_source}")
        print(path_faiss_index)

        if st.button('Khởi tạo ứng dụng'):
            initialize_app(path_faiss_index=path_faiss_index)
        st.button('🧹 Xóa lịch sử Chat', on_click=clear_chat_history)

    with tabs[1]:
        st.write("Bạn có thể tùy chỉnh các thiết lập tại đây (đang phát triển).")
        st.divider()
        st.button('Khởi tạo có sẵn', on_click=initialize_app)

    with tabs[2]:
        st.subheader("ℹ️ Thông tin")
        st.markdown("""
        **Tên dự án:** Chatbot hỗ trợ tuyển sinh HaUI  
        **Phiên bản:** 1.0  
        **Mô hình ngôn ngữ:** [gemini-1.5-flash](https://ai.google.dev/gemini)    
        **Mô hình embedding:** [hiieu/halong_embedding](https://huggingface.co/hiieu/halong_embedding)   
        **Mô hình xếp hạng lại:** [namdp-ptit/ViRanker](https://huggingface.co/namdp-ptit/ViRanker)   
        **Tác giả:** Lam Hoang  
        📧 Email: hpl114p@gmail.com  
        📅 Cập nhật lần cuối: 26/06/2025
        """)
        st.divider()

        st.markdown("### 🤖 Chatbot hoạt động như thế nào?")
        st.markdown("""
        Chatbot hoạt động bằng cách từ câu hỏi của người dùng, sử dụng kỹ thuật tìm văn bản liên quan đến câu hỏi trong bộ dữ liệu đã được vector hóa (text similarity) và lưu trữ thông qua vector database. 
        Những đoạn văn bản liên quan sẽ được trích xuất và đưa vào mô hình ngôn ngữ lớn (LLM) để sinh ra câu trả lời phù hợp.
        """)

        st.markdown("### 📌 Cách sử dụng chatbot để tra cứu thông tin")
        st.markdown("""
        Để sử dụng chatbot hiệu quả, bạn nên đặt câu hỏi rõ ràng và đầy đủ để mô hình hiểu đúng yêu cầu và đưa ra câu trả lời chính xác.
        Tuy nhiên, một số câu trả lời có thể chưa chính xác tuyệt đối, bạn nên kiểm chứng thông tin hoặc liên hệ hỗ trợ nếu cần.
        """)

        st.markdown("### ❓ Thông tin từ chatbot có đáng tin cậy không?")
        st.markdown("""
        Vì chatbot là một mô hình xác suất, nên vẫn có khả năng cung cấp thông tin không chính xác trong một số trường hợp. 
        Hãy kiểm chứng thông tin trước khi sử dụng hoặc liên hệ hỗ trợ để được giải đáp chính xác nhất.
        """)


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Đây là Asistant AI, trợ lý đắc lực dành cho bạn! Bạn muốn tìm kiếm thông tin về những gì?"}]

# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    # Hiển thị ngay trên giao diện
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input}) 

    # Xử lý truy vấn
    response, links = st.session_state.qa_chain.run(user_input)

    print("---------------------------------------------")
    print(links)
    # Hiển thị phản hồi
    with st.chat_message("assistant"):
        # st.markdown(response)
        response_placeholder = st.empty()
        displayed_text = ""
        for chunk in response.split():
            displayed_text += chunk + " "
            response_placeholder.markdown(displayed_text + "▌")
            time.sleep(0.05)  # điều chỉnh tốc độ hiển thị
        response_placeholder.markdown(displayed_text)  # xoá dấu ▌ cuối cùng

        with st.expander("🔗 Nguồn tham khảo (nhấn để xem)"):
            for i, link in enumerate(links, 1):
                st.markdown(f"{i}. [{link}]({link})")
        
    # Lưu vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": response})