@echo off
CALL conda activate cgcnn_2023

REM 
cd /d %~dp0

REM 
cd model\cgcnn
python main.py vacancy_data

echo.
pause >nul

