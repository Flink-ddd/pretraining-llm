# data_loader/data_loader.py

import torch
import numpy as np
from typing import Iterator, Tuple

def get_batch_iterator(
    data_path: str, 
    batch_size: int, 
    context_length: int, 
    device: str = "cpu",
    ddp: bool = False,
    ddp_rank: int = 0,
    ddp_world_size: int = 1
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    从预处理的二进制文件中创建数据批次迭代器。

    Args:
        data_path (str): .bin 文件的路径。
        batch_size (int): 每个批次的序列数。
        context_length (int): 每个序列的长度。
        device (str, optional): 'cpu' 或 'cuda'。默认为 "cpu"。
        ddp (bool): 是否启用分布式数据并行。
        ddp_rank (int): 当前进程的排名。
        ddp_world_size (int): 进程总数。

    Yields:
        tuple: (xb, yb) 输入和目标序列。
    """
    # 使用内存映射加载数据，高效处理大文件
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    # 如果是DDP模式，每个进程只处理数据的一部分
    if ddp:
        data = data[ddp_rank::ddp_world_size]

    while True:
        # 随机选择批次开始的位置
        ix = torch.randint(len(data) - context_length, (batch_size,))
        
        # 创建输入序列 (x) 和目标序列 (y)
        x = torch.stack([torch.from_numpy((data[i:i+context_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
        
        if device == 'cuda':
            # 将数据移动到GPU，non_blocking=True可以加速
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            
        yield x, y