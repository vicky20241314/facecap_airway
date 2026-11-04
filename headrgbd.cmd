@echo off
rem facecap_airway_delivery launcher (Windows cmd). PowerShell users can call headrgbd.ps1.
setlocal
set "PROJECT=%~dp0"
set "PYTHON=%PROJECT%runtime\env\python.exe"
if not exist "%PYTHON%" (
  echo Runtime Python not found: %PYTHON%
  echo See README.md section "Install" to prepare runtime\env.
  if "%~1"=="" pause
  exit /b 1
)
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONPATH=%PROJECT%"
if defined WITHB_RUNTIME_ROOT (
  set "HEADRGBD_RUNTIME_DIR=%WITHB_RUNTIME_ROOT%"
) else (
  set "HEADRGBD_RUNTIME_DIR=%PROJECT%runtime"
)
if "%~1"=="" (
  "%PYTHON%" -m head_rgbd run
) else (
  "%PYTHON%" -m head_rgbd %*
)
set "HEADRGBD_EXIT=%ERRORLEVEL%"
if "%~1"=="" if not "%HEADRGBD_EXIT%"=="0" (
  echo.
  echo HeadRGBD stopped with error code %HEADRGBD_EXIT%.
  pause
)
exit /b %HEADRGBD_EXIT%
