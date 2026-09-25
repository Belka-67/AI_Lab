@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -m venv .venv
  if errorlevel 1 goto :error
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m streamlit run app.py
goto :end
:error
echo.
echo Не удалось запустить приложение. Проверьте, установлен ли Python и доступен ли интернет для первой установки.
pause
:end
