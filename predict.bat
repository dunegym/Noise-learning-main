@echo off
title 预测模型

echo 开始进行预测...
echo.

REM 单文件预测（.txt文件）
REM python Main.py --is_predicting=True --global_step=562800

REM 批量预测（.mat文件）
echo 执行批量预测...
python Main.py --is_batch_predicting=True --global_step=562800

echo.
echo 预测完成！结果已保存到Result文件夹中。
pause