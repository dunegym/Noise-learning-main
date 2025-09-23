#!/bin/bash

# 训练脚本
echo "开始训练模型..."

# 设置PyTorch CUDA内存优化选项
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 从头开始训练（使用较小的batch size以减少内存使用）
echo "从头开始训练（优化内存使用）..."
python Main.py --is_training=True --is_pretrain=False --resume_from_latest=False --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=32

echo "如果要继续训练预训练模型，请使用以下命令："
echo "python Main.py --is_training=True --is_pretrain=True --global_step=562800 --resume_from_latest=False --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=32"

echo "如果要从最新的checkpoint恢复训练，请使用以下命令："
echo "python Main.py --is_training=True --is_pretrain=False --resume_from_latest=True --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=32"

echo "训练完成！"