# 训练脚本使用说明

## 脚本文件说明

本项目提供了多个训练脚本以适应不同的硬件环境：

1. **train.bat / train.sh** - 标准训练脚本
   - 将batch size从默认的64减小到32以减少内存使用
   - 设置PyTorch内存优化选项

2. **train_mem_optimized.bat / train_mem_optimized.sh** - 内存优化版训练脚本
   - 进一步将batch size减小到16
   - 将训练epoch数从默认的10减小到5
   - 适用于显存较小的GPU环境

## CUDA内存不足问题解决方案

如果遇到"torch.OutOfMemoryError: CUDA out of memory"错误，请尝试以下解决方案：

### 1. 使用内存优化脚本
```bash
# Linux/Mac
bash train_mem_optimized.sh

# Windows
双击运行 train_mem_optimized.bat
```

### 2. 手动设置环境变量
在运行训练脚本前设置：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Windows环境下：
```cmd
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3. 进一步减小batch size
可以尝试将batch size减小到8或更小：
```bash
python Main.py --is_training=True --batch_size=8
```

## 参数说明

- `--is_training=True` - 启用训练模式
- `--is_pretrain=False/True` - 是否使用预训练模型继续训练
- `--batch_size=N` - 设置批处理大小（默认64，建议根据GPU内存调整）
- `--max_epoch=N` - 设置训练轮数（默认10）

## 注意事项

1. 减小batch size可能会影响模型训练效果，但可以显著减少内存使用
2. 减少训练epoch数会缩短训练时间，但可能影响模型收敛
3. 如果仍有内存问题，请考虑使用CPU训练（修改config.py中的use_gpu=False）