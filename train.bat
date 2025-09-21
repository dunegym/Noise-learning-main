@echo off
title 训练模型

echo 开始训练模型...
echo.

REM 设置PyTorch CUDA内存优化选项
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM 从头开始训练（使用较小的batch size以减少内存使用）
echo 从头开始训练（优化内存使用）...
python Main.py --is_training=True --is_pretrain=False --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=32

echo.
echo 如果要继续训练预训练模型，请使用以下命令：
echo python Main.py --is_training=True --is_pretrain=True --global_step=562800 --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=32
echo.

echo 训练完成！
pause