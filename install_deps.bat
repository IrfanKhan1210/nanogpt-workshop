@echo off
cd /d "d:\Coding\being_arch_3\nano_gpt_workshop\nanogpt-workshop"
call .venv\Scripts\activate.bat
pip install numpy==2.2.6 torch==2.7.1 --index-strategy unsafe-best-match
pause
