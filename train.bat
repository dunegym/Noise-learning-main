@echo off
title 训练模型

echo 开始训练模型...
echo.

REM 从头开始训练
echo 从头开始训练...
python Main.py --is_training=True --is_pretrain=False

REM 如果要继续训练预训练模型，可以取消下面一行的注释并使用：
REM python Main.py --is_training=True --is_pretrain=True --global_step=562800

echo.
echo 训练完成！
pause