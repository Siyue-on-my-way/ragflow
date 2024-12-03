#!/usr/bin/env python3

######################################## 下载启动需要的nltk数据和大模型文件 ########################################
#!/usr/bin/env python3

import os
import requests
import zipfile
from huggingface_hub import snapshot_download
import nltk


# 定义目标目录
TARGET_DIR = "downloads"
PUNKT_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"


def ensure_dir_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def download_file(url, target_dir):
    """下载文件到指定目录"""
    ensure_dir_exists(target_dir)
    filename = os.path.join(target_dir, url.split("/")[-1])
    if not os.path.exists(filename):
        try:
            print(f"Downloading {url} to {filename}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    else:
        print(f"File {filename} already exists, skipping download.")
    return filename


def extract_zip(file_path, extract_to):
    """解压 ZIP 文件到指定目录"""
    ensure_dir_exists(extract_to)
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            print(f"Extracting {file_path} to {extract_to}...")
            zip_ref.extractall(extract_to)
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")


def download_model(repo_id, target_dir):
    """下载 Hugging Face 模型到指定目录"""
    local_dir = os.path.abspath(os.path.join(target_dir, "huggingface.co", repo_id))
    ensure_dir_exists(local_dir)
    try:
        print(f"Downloading Hugging Face repo {repo_id} to {local_dir}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    except Exception as e:
        print(f"Error downloading model {repo_id}: {e}")


if __name__ == "__main__":
    # 定义 punkt 数据的目标路径
    nltk_data_dir = os.path.join(TARGET_DIR, "nltk_data")
    punkt_zip_path = os.path.join(nltk_data_dir, "punkt.zip")
    punkt_extract_path = os.path.join(nltk_data_dir, "tokenizers")

    # 手动下载 punkt 数据
    ensure_dir_exists(nltk_data_dir)
    if not os.path.exists(punkt_extract_path):
        print("Downloading punkt zip ...")
        punkt_zip_file = download_file(PUNKT_URL, nltk_data_dir)
        print(f"extract punkt zipfile to {punkt_extract_path}")
        extract_zip(punkt_zip_file, punkt_extract_path)
    else:
        print(f"Punkt data already exists at {punkt_extract_path}, skipping extraction.")

    # 将路径添加到 NLTK 数据目录
    nltk.data.path.append(nltk_data_dir)

    # 验证 punkt 数据是否正常
    try:
        print("Validating punkt tokenizer...")
        nltk.download("punkt", download_dir=nltk_data_dir)
        from nltk.tokenize import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
        print("Punkt tokenizer is ready to use!")
    except Exception as e:
        print(f"Error validating punkt tokenizer: {e}")

    # 示例：其他任务可以继续添加到这里
    print("All tasks completed!")