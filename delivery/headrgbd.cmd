@echo off
rem facecap_airway_delivery launcher (Windows cmd). PowerShell users can call headrgbd.ps1.
setlocal
set "PROJECT=%~dp0"
set "PYTHON=%PROJECT%runtime\env\python.exe"
if not exist "%PYTHON%" (
  echo Runtime Python not found: %PYTHON%
  echo See README.md section "Install" to prepare runtime\env.
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
"%PYTHON%" -m head_rgbd %*
exit /b %ERRORLEVEL%
