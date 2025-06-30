# scripts/data_download.py

import os
from datasets import load_dataset
from config.config import default_config as config

def download_dataset():
    """
    使用 Hugging Face datasets 库下载 OpenWebText 数据集。
    这将自动处理下载、解压和缓存。
    """
    dataset_name = config.get('dataset_name', 'openwebtext')
    print(f"正在下载并准备 '{dataset_name}' 数据集...")
    
    # load_dataset 会处理所有事情，并把数据缓存到 ~/.cache/huggingface/datasets
    dataset = load_dataset(dataset_name, split='train')
    
    print("数据集已成功下载并缓存。")
    print("示例数据:")
    print(dataset[0])

if __name__ == '__main__':
    download_dataset()