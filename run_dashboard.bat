@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: creating environment and installing packages...
  rem Prefer 3.14 via the py launcher: 3.15 has no pyarrow build yet.
  py -3.14 -m venv .venv 2>nul || python -m venv .venv || (echo Python 3.12 to 3.14 not found. Install 3.14 from python.org & pause & exit /b 1)
  .venv\Scripts\python.exe -m pip install -r requirements.txt || (pause & exit /b 1)
)
.venv\Scripts\python.exe -m streamlit run dashboard_app.py
pause
