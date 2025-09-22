@echo off
title Predict

call venv\Scripts\activate.bat
pause

echo Starting predicting...
echo.

echo predicting by batch...
python Main.py --is_batch_predicting=True --global_step=562800

echo.
echo success!
pause