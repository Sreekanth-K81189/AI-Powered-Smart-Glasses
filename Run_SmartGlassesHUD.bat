@echo off
REM Launcher: adds vcpkg and CUDA to PATH, then runs SmartGlassesHUD.exe
REM Set VCPKG_ROOT or CUDA_PATH if your install is elsewhere.
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run_SmartGlassesHUD.ps1" %*
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% neq 0 (
  echo.
  echo Exit code: %EXITCODE%
  if %EXITCODE%==-1073741515 echo DLL not found. Set VCPKG_ROOT and CUDA_PATH if vcpkg/CUDA are elsewhere.
  pause
)
exit /b %EXITCODE%
