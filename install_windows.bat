@echo off
setlocal
py -3 -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python -m pip install -e . --no-build-isolation
if errorlevel 1 exit /b 1
echo.
echo Installation complete.
echo Copy voderberg_srn2_angles45_contact_optimV4.init beside settings.toml.
endlocal
