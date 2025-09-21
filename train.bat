@echo off
title 训练模型

echo 开始训练模型...
echo.

REM 从头开始训练
echo 从头开始训练...
python Main.py --is_training=True --is_pretrain=False --is_testing=False --is_predicting=False --is_batch_predicting=False

echo.
echo 如果要继续训练预训练模型，请使用以下命令：
echo python Main.py --is_training=True --is_pretrain=True --global_step=562800 --is_testing=False --is_predicting=False --is_batch_predicting=False
echo.

echo 训练完成！
pause