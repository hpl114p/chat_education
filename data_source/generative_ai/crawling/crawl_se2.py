import os
import json
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# Load API key
load_dotenv(dotenv_path=r"B:\PROJECTS\CHATBOT_EDUCATION.env.init")
gemini_key = os.getenv("gemini_key")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Setup Chrome Driver
chrome_options = Options()
chrome_options.add_argument("--headless")  # chạy ẩn
driver = webdriver.Chrome(service=Service("chromedriver.exe"), options=chrome_options)

def is_valid_url(url, base_url):
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return parsed_url.netloc == parsed_base.netloc

def get_links(base_url, num_pages):
    all_links = []
    for page in range(1, num_pages + 1):
        url = base_url if page == 1 else f"{base_url}?page={page}"
        print(f"Crawling page {page}: {url}")
        try:
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            rows = soup.select("div.col-md-8 div.row")
            raw_links = [
                urljoin(url, row.find('a', href=True)['href']) 
                for row in rows if row.find('a', href=True)
            ]
            valid_links = [link for link in raw_links if is_valid_url(link, base_url)]
            all_links.extend(valid_links)
        except WebDriverException as e:
            print(f"Selenium error loading page {page}: {e}")
    return all_links

def fetch_important_text(url, visited, base_url, output_file="output.json"):
    total_characters = 0
    try:
        print(f"Fetching: {url}")
        visited.add(url)
        driver.get(url)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        h3_tag = soup.select_one("div.col-lg-9.col-md-8 h3")
        title = h3_tag.get_text(strip=True) if h3_tag else "No title"

        important_text = []
        for tag in ['p', 'span', 'table']:
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text:
                    important_text.append(text)

        prompt = f"""This is my text, give me the new text with all redundant punctuation marks and some redundant words that dont link to the context of the given text removed(remove redundant stuffs not summarize)
                    Do not omit content, especially tabular content
                    : {" ".join(important_text)}"""
        response = model.generate_content([prompt])
        generated_text = response.text

        with open(output_file, "a", encoding="utf-8") as f:
            if f.tell() > 1:
                f.write(",\n")
            json.dump({"link": url, "title": title, "content": generated_text}, f, ensure_ascii=False, indent=2)
            print(f"Success fetching {url}")
            total_characters += len(generated_text)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return total_characters

def update_status_file(status_file, status_data):
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=4)

def auto_crawl(base_url, num_pages):
    status_file = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\status2.json"
    output_file = r"B:\PROJECTS\CHATBOT_EDUCATION\data_source\data\raw\outputs2.json"
    visited_urls = set()
    path_links = get_links(base_url=base_url, num_pages=num_pages)

    status_data = {
        "status": "initializing",
        "total_links_crawled": 0,
        "total_characters_crawled": 0,
        "error": None
    }
    update_status_file(status_file, status_data)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("[")
        status_data["status"] = "crawling"
        update_status_file(status_file, status_data)

        links_crawled = 0
        total_characters = 0
        for link in path_links:
            total_characters += fetch_important_text(link, visited_urls, base_url, output_file)
            links_crawled += 1

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n]")

        status_data["status"] = "completed"
        status_data["total_links_crawled"] = links_crawled
        status_data["total_characters_crawled"] = total_characters
        update_status_file(status_file, status_data)

        print(f"Total links crawled: {links_crawled}")
        print(f"Total characters crawled: {total_characters}")
        return links_crawled, total_characters

    except Exception as e:
        status_data["status"] = "error"
        status_data["error"] = str(e)
        update_status_file(status_file, status_data)
        raise e

if __name__ == '__main__':
    base_url = "https://tuyensinh.haui.edu.vn/tin-tuc"
    num_pages = 1
    auto_crawl(base_url, num_pages)
    driver.quit()  # đóng trình duyệt khi xong
