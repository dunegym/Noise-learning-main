#!/bin/bash

# 训练脚本
echo "开始训练模型..."

# 从头开始训练
python Main.py --is_training=True --is_pretrain=False

# 如果要继续训练预训练模型，可以使用以下命令：
# python Main.py --is_training=True --is_pretrain=True --global_step=562800

echo "训练完成！"