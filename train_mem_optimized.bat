@echo off
title 训练模型（内存优化版）

echo 开始训练模型（内存优化版）...
echo.

REM 设置PyTorch CUDA内存优化选项
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM 进一步减少batch size并设置其他内存优化选项
echo 使用进一步优化的内存设置进行训练...
python Main.py --is_training=True --is_pretrain=False --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=16 --max_epoch=5

echo.
echo 如果要继续训练预训练模型，请使用以下命令：
echo python Main.py --is_training=True --is_pretrain=True --global_step=562800 --is_testing=False --is_predicting=False --is_batch_predicting=False --batch_size=16 --max_epoch=5
echo.

echo 训练完成！
pause