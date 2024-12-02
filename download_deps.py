#!/usr/bin/env python3

######################################## 下载启动需要的nltk数据和大模型文件 ########################################
from huggingface_hub import snapshot_download
import nltk
import os
import urllib.request

# 定义目标目录
TARGET_DIR = "downloads"

# URL 列表
urls = [
    "http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb",
    "http://ports.ubuntu.com/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb",
    "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.0.0/tika-server-standard-3.0.0.jar",
    "https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.0.0/tika-server-standard-3.0.0.jar.md5",
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "https://bit.ly/chrome-linux64-121-0-6167-85",
    "https://bit.ly/chromedriver-linux64-121-0-6167-85",
]

# Hugging Face 仓库列表
repos = [
    "InfiniFlow/text_concat_xgb_v1.0",
    "InfiniFlow/deepdoc",
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-reranker-v2-m3",
    "maidalun1020/bce-embedding-base_v1",
    "maidalun1020/bce-reranker-base_v1",
]


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
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    else:
        print(f"File {filename} already exists, skipping download.")


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
    # 下载 URL 文件
    for url in urls:
        download_file(url, TARGET_DIR)

    # 下载 NLTK 数据
    nltk_data_dir = os.path.join(TARGET_DIR, "nltk_data")
    for data in ["wordnet", "punkt", "punkt_tab"]:
        ensure_dir_exists(nltk_data_dir)
        print(f"Downloading nltk {data} to {nltk_data_dir}...")
        try:
            nltk.download(data, download_dir=nltk_data_dir)
        except Exception as e:
            print(f"Error downloading NLTK data {data}: {e}")

    # 下载 Hugging Face 仓库
    for repo_id in repos:
        download_model(repo_id, TARGET_DIR)