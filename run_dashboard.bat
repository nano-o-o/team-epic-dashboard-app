@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: creating environment and installing packages...
  python -m venv .venv || (echo Python 3.12+ not found. Install it from python.org & pause & exit /b 1)
  .venv\Scripts\python.exe -m pip install -r requirements.txt || (pause & exit /b 1)
)
.venv\Scripts\python.exe -m streamlit run dashboard_app.py
pause
