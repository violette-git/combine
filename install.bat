@echo off
REM PowerQuant installer for Windows (Command Prompt / PowerShell)
REM
REM Usage:
REM   install.bat                    full install (auto-detects GPU)
REM   install.bat --skip-powerinfer  Python + TurboQuant only, skip C++ build
REM   install.bat --cpu-only         Force CPU-only build of PowerInfer
REM
REM Requirements:
REM   Python 3.10+  https://python.org/downloads
REM   Git           https://git-scm.com/download/win
REM   cmake         https://cmake.org/download  (only needed for PowerInfer)
REM   VS Build Tools https://visualstudio.microsoft.com/visual-cpp-build-tools/
REM                  (select "Desktop development with C++" workload)

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set VENDOR_DIR=%SCRIPT_DIR%vendor
set POWERINFER_DIR=%VENDOR_DIR%\PowerInfer
set POWERINFER_REPO=https://github.com/Tiiny-AI/PowerInfer
set SKIP_POWERINFER=0
set CPU_ONLY=0

for %%A in (%*) do (
    if "%%A"=="--skip-powerinfer" set SKIP_POWERINFER=1
    if "%%A"=="--cpu-only" set CPU_ONLY=1
)

echo.
echo =============================================
echo  PowerQuant Installer for Windows
echo =============================================
echo.

REM ──────────────────────────────────────────────
REM  1. Python dependencies
REM ──────────────────────────────────────────────
echo === Step 1/4: Installing Python dependencies ===
echo.

REM Detect NVIDIA GPU
set HAS_NVIDIA=0
if %CPU_ONLY%==0 (
    nvidia-smi >nul 2>&1 && set HAS_NVIDIA=1
)

if %HAS_NVIDIA%==1 (
    echo NVIDIA GPU detected -- installing CUDA PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo CUDA PyTorch install failed, falling back to CPU...
        pip install torch
    )
) else (
    echo Installing CPU PyTorch...
    pip install torch
)

pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install Python requirements.
    echo Make sure Python 3.10+ is installed: https://python.org/downloads
    pause
    exit /b 1
)

REM ──────────────────────────────────────────────
REM  2. Clone PowerInfer
REM ──────────────────────────────────────────────
if %SKIP_POWERINFER%==1 (
    echo Skipping PowerInfer ^(--skip-powerinfer^).
    echo The HuggingFace + TurboQuant backend will still work without it.
    goto :install_python_pkg
)

echo.
echo === Step 2/4: Cloning PowerInfer ===
echo.

if not exist "%VENDOR_DIR%" mkdir "%VENDOR_DIR%"

if not exist "%POWERINFER_DIR%" (
    git clone %POWERINFER_REPO% "%POWERINFER_DIR%"
    if errorlevel 1 (
        echo ERROR: git clone failed. Is Git installed?
        echo Download Git from: https://git-scm.com/download/win
        pause
        exit /b 1
    )
) else (
    echo PowerInfer already present -- pulling latest...
    git -C "%POWERINFER_DIR%" pull --ff-only
)

REM ──────────────────────────────────────────────
REM  3. Build PowerInfer (requires cmake + MSVC)
REM ──────────────────────────────────────────────
echo.
echo === Step 3/4: Building PowerInfer ===
echo.

REM Check cmake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: cmake not found.
    echo.
    echo PowerInfer must be compiled from C++ source.
    echo.
    echo Install cmake ^(pick one^):
    echo   winget install Kitware.CMake
    echo   choco install cmake
    echo   Download from https://cmake.org/download/
    echo   ^(tick "Add cmake to system PATH" during setup^)
    echo.
    echo You also need Visual Studio Build Tools:
    echo   winget install Microsoft.VisualStudio.2022.BuildTools
    echo   ^(select "Desktop development with C++" workload^)
    echo.
    echo After installing, close this window and re-run install.bat
    echo.
    echo OR skip the C++ build -- the HuggingFace backend works without it:
    echo   install.bat --skip-powerinfer
    echo.
    goto :install_python_pkg
)

REM Configure cmake args
set CMAKE_GPU_FLAG=
if %CPU_ONLY%==1 (
    echo Building CPU-only ^(--cpu-only specified^).
) else if %HAS_NVIDIA%==1 (
    echo Building with CUDA ^(NVIDIA GPU detected^).
    set CMAKE_GPU_FLAG=-DLLAMA_CUBLAS=ON
) else (
    echo No GPU detected -- building CPU-only.
)

cmake -S "%POWERINFER_DIR%" -B "%POWERINFER_DIR%\build" -DCMAKE_BUILD_TYPE=Release %CMAKE_GPU_FLAG%
if errorlevel 1 (
    echo.
    echo ERROR: cmake configuration failed.
    echo Make sure Visual Studio Build Tools are installed with the
    echo "Desktop development with C++" workload.
    echo   winget install Microsoft.VisualStudio.2022.BuildTools
    echo.
    goto :install_python_pkg
)

cmake --build "%POWERINFER_DIR%\build" --config Release -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo ERROR: Build failed. Check the output above for details.
    goto :install_python_pkg
)

echo PowerInfer built at %POWERINFER_DIR%\build

REM ──────────────────────────────────────────────
REM  4. Install powerquant Python package
REM ──────────────────────────────────────────────
:install_python_pkg
echo.
echo === Step 4/4: Installing powerquant ===
echo.
pip install -e "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ERROR: pip install -e failed.
    pause
    exit /b 1
)

REM ──────────────────────────────────────────────
REM  Done
REM ──────────────────────────────────────────────
echo.
echo =============================================
echo  Installation complete!
echo =============================================
echo.
echo Quick start:
echo.
echo   Interactive chat (HF model + TurboQuant KV compression):
echo     powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit
echo.
echo   Single generation:
echo     powerquant generate --model gpt2 --prompt "The future of AI is"
echo.
echo   Benchmark:
echo     powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit
echo.
echo   System info:
echo     powerquant info
echo.
echo Python API:
echo   from powerquant import Engine
echo   engine = Engine.from_pretrained("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
echo   print(engine.generate("Hello!"))
echo.
pause
