# scripts/train_transformer.py

import os
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator

# --- DDP 设置 ---
ddp = int(os.environ.get('RANK', -1)) != -1 # 检查是否在DDP环境中
if ddp:
    dist.init_process_group(backend=config['ddp_backend'])
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 主进程负责打印和保存
else:
    # 非DDP模式
    ddp_rank = 0
    ddp_world_size = 1
    device = config['device']
    master_process = True

# --- PyTorch 2.0 优化 ---
# 检查是否使用 torch.compile
compile_model = True if os.environ.get("TORCH_COMPILE", "1") == "1" else False

class Trainer:
    def __init__(self, model: nn.Module, train_iterator, val_iterator, optimizer):
        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))

    def _get_lr(self, it):
        """学习率衰减"""
        # 1) linear warmup for train_steps steps
        if it < config['t_train_steps'] * 0.1: # 预热10%的步数
            return config['t_lr'] * it / (config['t_train_steps'] * 0.1)
        # 2) constant lr otherwise
        return config['t_lr']

    @torch.no_grad()
    def _evaluate(self):
        """评估模型"""
        self.model.eval()
        losses = torch.zeros(config['t_eval_iters'])
        for k in range(config['t_eval_iters']):
            X, Y = next(self.val_iterator)
            with torch.autocast(device_type=config['device'], dtype=torch.bfloat16):
                _, loss = self.model(X, Y)
            losses[k] = loss.item()
        self.model.train()
        return losses.mean()

    def _train_step(self, xb, yb):
        """单个训练步"""
        with torch.autocast(device_type=config['device'], dtype=torch.bfloat16):
            logits, loss = self.model(xb, yb)
        
        self.scaler.scale(loss).backward()
        return loss

    def train(self):
        """训练循环"""
        self.model.train()
        t0 = time.time()
        
        for step in range(config['t_train_steps']):
            # 更新学习率
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # DDP同步
            if ddp:
                self.model.require_backward_grad_sync = (step % 2 == 0)

            # 获取数据并训练
            xb, yb = next(self.train_iterator)
            loss = self._train_step(xb, yb)
            
            # 优化器步骤
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # 日志记录和评估
            if step % config['t_eval_steps'] == 0 and master_process:
                t1 = time.time()
                dt = t1 - t0
                val_loss = self._evaluate()
                print(f"Step {step}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, LR={lr:.2e}, Time={dt*1000:.2f}ms")
                t0 = t1

        if master_process:
            print(f"训练完成. 保存模型到 {config['t_out_path']}")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, config['t_out_path'])


def main():
    # --- 模型初始化 ---
    model_args = {k: config[k] for k in ['vocab_size', 'n_embed', 'n_head', 'n_blocks', 'context_length']}
    model = Transformer(**model_args)
    model.to(device)
    
    if compile_model:
        print("正在编译模型... (这可能需要几分钟)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # --- 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
    
    # --- 数据加载器 ---
    train_iterator = get_batch_iterator(
        config['train_path'], config['t_batch_size'], config['context_length'], device,
        ddp, ddp_rank, ddp_world_size
    )
    val_iterator = get_batch_iterator(
        config['val_path'], config['t_batch_size'], config['context_length'], device,
        ddp, ddp_rank, ddp_world_size
    )
    
    # --- 训练 ---
    trainer = Trainer(model, train_iterator, val_iterator, optimizer)
    trainer.train()

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()