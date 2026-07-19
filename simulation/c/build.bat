@echo off
REM Build seir_core.dll on Windows (MSVC or MinGW-w64 / clang-cl).
REM
REM Usage (Developer Command Prompt for VS / PowerShell with MSVC on PATH):
REM     simulation\c\build.bat
REM
REM The sibling build.sh uses `cc` (gcc/clang via Git Bash or WSL) and is the
REM preferred path on Unix-like shells. This .bat exists for users who only
REM have vanilla Windows cmd.exe + MSVC installed.

setlocal
pushd "%~dp0"

where cl >nul 2>nul
if %ERRORLEVEL% equ 0 goto USE_MSVC

where clang >nul 2>nul
if %ERRORLEVEL% equ 0 goto USE_CLANG

where gcc >nul 2>nul
if %ERRORLEVEL% equ 0 goto USE_GCC

echo [ERROR] No supported C compiler found on PATH.
echo         Install Visual Studio Build Tools (cl.exe), LLVM (clang), or
echo         MSYS2 / MinGW-w64 (gcc), then re-run this script.
popd
endlocal
exit /b 1

:USE_MSVC
echo Building with MSVC cl.exe ...
cl /nologo /O2 /LD /D_CRT_SECURE_NO_WARNINGS /Fe:seir_core.dll seir_core.c
if errorlevel 1 goto FAIL
goto DONE

:USE_CLANG
echo Building with clang ...
REM B-P3 (M7): -ffast-math removed (IEEE NaN/Inf + reproducibility; see build.sh).
clang -O3 -shared -o seir_core.dll seir_core.c
if errorlevel 1 goto FAIL
goto DONE

:USE_GCC
echo Building with gcc ...
gcc -O3 -shared -o seir_core.dll seir_core.c
if errorlevel 1 goto FAIL
goto DONE

:FAIL
echo [ERROR] Build failed.
popd
endlocal
exit /b 1

:DONE
echo   OK - seir_core.dll built
popd
endlocal
exit /b 0
