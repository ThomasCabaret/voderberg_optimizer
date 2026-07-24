@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  echo Missing .venv. Run install_windows.bat first.
  exit /b 1
)
.venv\Scripts\python.exe -m voderberg_optimizer.cli optimize --settings settings.toml --no-display
endlocal
