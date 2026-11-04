@echo off
rem Continuous-turn RGB-D: double-click opens the GUI; run starts the CLI workflow.
setlocal
set "PROJECT=%~dp0"
set "RUNTIME=%PROJECT%runtime"
if defined WITHB_RUNTIME_ROOT set "RUNTIME=%WITHB_RUNTIME_ROOT%"
set "PYTHON=%RUNTIME%\env\python.exe"
if not exist "%PYTHON%" (
  echo Runtime Python not found: %PYTHON%
  echo See docs\TURN_GUI.md to prepare the runtime.
  if "%~1"=="" pause
  exit /b 1
)
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONPATH=%PROJECT%"
set "HEADRGBD_RUNTIME_DIR=%RUNTIME%"
if "%~1"=="" (
  if exist "%RUNTIME%\env\pythonw.exe" (
    start "" "%RUNTIME%\env\pythonw.exe" -m head_rgbd.experiment.turn_app gui
    if errorlevel 1 exit /b 1
    exit /b 0
  )
  "%PYTHON%" -m head_rgbd.experiment.turn_app gui
) else (
  "%PYTHON%" -m head_rgbd.experiment.turn_app %*
)
set "TURN_EXIT=%ERRORLEVEL%"
if "%~1"=="" if not "%TURN_EXIT%"=="0" pause
exit /b %TURN_EXIT%
