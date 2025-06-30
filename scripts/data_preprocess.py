# scripts/data_preprocess.py

import os
from multiprocessing import Pool, cpu_count
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from config.config import default_config as config

# --- 配置 ---
DATASET_NAME = config.get('dataset_name')
TOKENIZER_NAME = config.get('tokenizer_name', 'gpt2')
TRAIN_FILE = config.get('train_path')
VAL_FILE = config.get('val_path')
NUM_PROC = cpu_count()

# --- 主函数 ---
if __name__ == '__main__':
    # 加载数据集
    print(f"正在加载 '{DATASET_NAME}' 数据集...")
    dataset = load_dataset(DATASET_NAME)
    
    # 划分训练集和验证集
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=42, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # 初始化分词器
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    def process_and_encode(example):
        """对单个文本进行分词，并添加EOS token"""
        ids = enc.encode_ordinary(example['text'])  # 使用 encode_ordinary 更快
        ids.append(enc.eot_token)  # 添加<|endoftext|>
        return {'ids': ids, 'len': len(ids)}

    # 在所有进程上并行处理数据集
    print("正在对数据集进行分词...")
    tokenized = split_dataset.map(
        process_and_encode,
        remove_columns=['text'],
        desc="分词",
        num_proc=NUM_PROC,
    )

    # 将所有分词后的数据连接并保存为二进制文件
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = TRAIN_FILE if split == 'train' else VAL_FILE
        # 使用uint16来存储token ID，因为GPT-2词汇表大小为50257
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        total_batches = 1024

        print(f"正在写入 {filename}...")
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'写入 {split} 数据'):
            # 分批处理以避免内存问题
            batch = dset.shard(num_shards=total_batches, index=batch_idx)
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    print("数据预处理完成！")