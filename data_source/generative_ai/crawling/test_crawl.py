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

######################################## Duyet qua cac pages lay links
base_url = 'https://tuyensinh.haui.edu.vn/tin-tuc'
num_pages = 5
all_links = []

for page in range(1, num_pages + 1):
    # Tạo URL của từng trang
    if page == 1:
        url = base_url  # Trang 1 đặc biệt: không có ?page=1
    else:
        url = f"{base_url}?page={page}"

    print(f"Crawling page {page}: {url}")

    response = requests.get(url)
    response.encoding = 'utf-8'  # Set encoding đúng tiếng Việt

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Lấy tất cả các row bên trong col-md-8
        rows = soup.select("div.col-md-8 div.row")

        page_links  = [urljoin(url, row.find('a', href=True)['href']) 
                for row in soup.select("div.col-md-8 div.row") 
                if row.find('a', href=True)]
        
        all_links.extend(page_links)
    else:
        print(f"Failed to access page {page}")

# Xóa trùng lặp nếu cần
# all_links = list(set(all_links))

print("Tổng số link crawl được:", len(all_links))
# for link in all_links:
#     print(link)

# # Lấy liên kết từ trang web, giới hạn số liên kết
# links = [urljoin(url, row.find('a', href=True)['href']) 
#          for row in soup.select("div.col-md-8 div.row") 
#          if row.find('a', href=True)]

# print(links)
# print(len(links))

###################################### Lay noi dung trong cac links

# Save to file
output_file = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\output.json"

# Xóa nội dung cũ trong file nếu có
with open(output_file, "w", encoding="utf-8") as f:
    f.write("[")  # Bắt đầu file JSON

for url in all_links:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Lay tieu de bai viet
    h3_tag = soup.select_one("div.col-lg-9.col-md-8 h3")
    title = h3_tag.get_text(strip=True)

    important_text = []
    for tag in ['p', 'span', 'table']:
        for element in soup.find_all(tag):
            text = element.get_text(strip=True)
            if text:  # Bỏ qua nội dung rỗng
                important_text.append(text)

    prompt = f"""This is my text, give me the new text with all redundant punctuation marks and some redundant words that dont link to the context of the given text removed(remove redundant stuffs not summarize): {" ".join(important_text)}"""
    response = model.generate_content([prompt])
    # Assuming response has a field called "text"
    generated_text = response.text

    # print(generated_text)

    with open(output_file, "a", encoding="utf-8") as f:
        if f.tell() > 1:  # Kiểm tra nếu không phải là lần đầu ghi vào file
            f.write(",\n")
        json.dump({"link": url, "title": title,"content": generated_text}, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_file}")

# Kết thúc file JSON
with open(output_file, "a", encoding="utf-8") as f:
    f.write("\n]")  # Đóng file JSON
