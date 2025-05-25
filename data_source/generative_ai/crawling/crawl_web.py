import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION\.env.init")

gemini_key = os.getenv("gemini_key")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-1.5-flash')


def is_valid_url(url, base_url):
    """
    Kiểm tra xem URL có hợp lệ và liên quan đến website chính hay không.
    :param url: URL cần kiểm tra
    :param base_url: URL gốc để kiểm tra liên kết có cùng domain
    """
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return parsed_url.netloc == parsed_base.netloc  # Chỉ chấp nhận các link cùng domain

def get_links(base_url, num_pages):
    all_links = []
    for page in range(1, num_pages + 1):
        # Tạo URL của từng trang
        if page == 1:
            url = base_url  # Trang 1 đặc biệt: không có ?page=1
        else:
            url = f"{base_url}?page={page}"

        print(f"Crawling page {page}: {url}")

        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8'  # Set encoding đúng tiếng Việt

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Lấy tất cả các row bên trong col-md-8
                rows = soup.select("div.col-md-8 div.row")

                raw_links = [
                    urljoin(url, row.find('a', href=True)['href']) 
                    for row in rows if row.find('a', href=True)
                ]

                # Lọc chỉ giữ lại các link hợp lệ (cùng domain)
                valid_links = [link for link in raw_links if is_valid_url(link, base_url)]
                
                all_links.extend(valid_links)
            else:
                print(f"Failed to access page {page} (status code {response.status_code})")

        except Exception as e:
            print(f"Error fetching page {page}: {e}")

    return all_links

def fetch_important_text(url, visited, base_url, output_file="output.json", status_callback=None):
    total_links_crawled = 0  # Biến đếm số liên kết đã crawl
    total_characters = 0  # Biến đếm tổng số ký tự của nội dung đã crawl

    try:
        print(f"Fetching: {url}")
        visited.add(url)  # Đánh dấu URL là đã truy cập

        # Gửi yêu cầu HTTP
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Lấy tiêu đề bài viết
        h3_tag = soup.select_one("div.col-lg-9.col-md-8 h3")
        title = h3_tag.get_text(strip=True)

        # Lấy nội dung
        important_text = []
        for tag in ['p', 'span', 'table']:
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text:  # Bỏ qua nội dung rỗng
                    important_text.append(text)

        # Xử lý nội dung bằng LLM
        prompt = f"""This is my text, give me the new text with all redundant punctuation marks and some redundant words that dont link to the context of the given text removed(remove redundant stuffs not summarize)
                    Do not omit content, especially tabular content
                    : {" ".join(important_text)}"""
        response = model.generate_content([prompt])
        # Assuming response has a field called "text"
        generated_text = response.text

        with open(output_file, "a", encoding="utf-8") as f:
            if f.tell() > 1:  # Kiểm tra nếu không phải là lần đầu ghi vào file
                f.write(",\n")
            json.dump({"link": url, "title": title,"content": generated_text}, f, ensure_ascii=False, indent=2)
            print(f"Success fetching {url}")

        # Cập nhật số lượng ký tự
        total_characters += len(generated_text)
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")

    return total_characters  # Trả về số link đã crawl và tổng số ký tự

def auto_crawl(base_url, num_pages):
    # Status file path
    status_file = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\status.json"
    output_file = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\outputs.json"
    visited_urls = set()  # Tập hợp để lưu các URL đã truy cập

    path_links = get_links(base_url=base_url, num_pages=num_pages)

    # Initialize the status file
    status_data = {
        "status": "initializing",
        "total_links_crawled": 0,
        "total_characters_crawled": 0,
        "error": None
    }
    update_status_file(status_file, status_data)

    try:
        # Prepare output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("[")  # Start JSON array

        # Update status: crawling started
        status_data["status"] = "crawling"
        update_status_file(status_file, status_data)

        links_crawled = 0
        total_characters = 0
        # Perform crawling and fetch data
        for link in path_links:
            total_characters += fetch_important_text(link, visited_urls, base_url, output_file)
            links_crawled += 1

        # Close JSON array in output file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n]")

        # Update status: completed
        status_data["status"] = "completed"
        status_data["total_links_crawled"] = links_crawled
        status_data["total_characters_crawled"] = total_characters
        update_status_file(status_file, status_data)

        # Return the results
        print(f"Total links crawled: {links_crawled}")
        print(f"Total characters crawled: {total_characters}")
        return links_crawled, total_characters

    except Exception as e:
        # Update status: error
        status_data["status"] = "error"
        status_data["error"] = str(e)
        update_status_file(status_file, status_data)
        raise e


def update_status_file(status_file, status_data):
    """Helper function to update the status file."""
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=4)

if __name__=='__main__':
    base_url = 'https://tuyensinh.haui.edu.vn/tin-tuc'
    num_pages = 1

    auto_crawl(base_url, num_pages)