#!/bin/bash

# 预测脚本
echo "开始进行预测..."

# 单文件预测（.txt文件）
# python Main.py --is_predicting=True --global_step=562800

# 批量预测（.mat文件）
python Main.py --is_batch_predicting=True --global_step=5200

echo "预测完成！结果已保存到Result文件夹中。"