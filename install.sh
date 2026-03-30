#!/usr/bin/env bash
# PowerQuant installer — Linux / macOS / Windows (Git Bash)
#
# Usage:
#   bash install.sh                  # full install (auto-detects GPU)
#   bash install.sh --cpu-only       # CPU-only build of PowerInfer
#   bash install.sh --skip-powerinfer  # Python + TurboQuant only, skip C++ build
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
POWERINFER_DIR="$VENDOR_DIR/PowerInfer"
POWERINFER_REPO="https://github.com/Tiiny-AI/PowerInfer"

CPU_ONLY=false
SKIP_POWERINFER=false

for arg in "$@"; do
  case $arg in
    --cpu-only)        CPU_ONLY=true ;;
    --skip-powerinfer) SKIP_POWERINFER=true ;;
  esac
done

banner() { echo; echo "=== $* ==="; echo; }

# ──────────────────────────────────────────────
# 1. Python dependencies
# ──────────────────────────────────────────────
banner "Step 1/4: Installing Python dependencies"

# Detect NVIDIA GPU
HAS_NVIDIA=false
if ! $CPU_ONLY; then
  if command -v nvidia-smi &>/dev/null; then
    HAS_NVIDIA=true
  elif command -v nvcc &>/dev/null; then
    HAS_NVIDIA=true
  fi
fi

if $HAS_NVIDIA; then
  echo "NVIDIA GPU detected — installing CUDA PyTorch..."
  pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch
else
  echo "Installing CPU PyTorch (no NVIDIA GPU detected)..."
  pip install torch
fi

pip install -r "$SCRIPT_DIR/requirements.txt"

# ──────────────────────────────────────────────
# 2. Clone PowerInfer
# ──────────────────────────────────────────────
if $SKIP_POWERINFER; then
  echo "Skipping PowerInfer (--skip-powerinfer)."
  echo "The HuggingFace + TurboQuant backend will still work without it."
else
  banner "Step 2/4: Cloning PowerInfer"
  mkdir -p "$VENDOR_DIR"
  if [ ! -d "$POWERINFER_DIR" ]; then
    git clone "$POWERINFER_REPO" "$POWERINFER_DIR"
  else
    echo "PowerInfer already present — pulling latest..."
    git -C "$POWERINFER_DIR" pull --ff-only || true
  fi

  # ──────────────────────────────────────────
  # 3. Build PowerInfer (requires cmake)
  # ──────────────────────────────────────────
  banner "Step 3/4: Building PowerInfer"

  # Check for cmake and give a clear error if missing
  if ! command -v cmake &>/dev/null; then
    OS_TYPE="$(uname -s 2>/dev/null || echo Windows)"
    echo ""
    echo "ERROR: cmake not found."
    echo ""
    echo "PowerInfer is a C++ program that must be compiled."
    echo "Install cmake for your platform, then re-run this script."
    echo ""
    case "$OS_TYPE" in
      MINGW*|MSYS*|CYGWIN*|Windows*)
        echo "  Windows options (pick one):"
        echo "    winget install Kitware.CMake"
        echo "    choco install cmake"
        echo "    Download from https://cmake.org/download/ and tick 'Add to PATH'"
        echo ""
        echo "  You also need a C++ compiler. Easiest option:"
        echo "    winget install Microsoft.VisualStudio.2022.BuildTools"
        echo "    (select 'Desktop development with C++' workload)"
        echo ""
        echo "  After installing cmake and a compiler, close and reopen your terminal,"
        echo "  then run:  bash install.sh"
        echo ""
        echo "  OR skip the C++ build entirely (HuggingFace + TurboQuant still work):"
        echo "    bash install.sh --skip-powerinfer"
        ;;
      Darwin*)
        echo "  macOS:  brew install cmake"
        echo "          xcode-select --install    (for the C++ compiler)"
        ;;
      Linux*)
        echo "  Ubuntu/Debian:  sudo apt-get install cmake build-essential"
        echo "  Fedora:         sudo dnf install cmake gcc-c++"
        echo "  Arch:           sudo pacman -S cmake base-devel"
        ;;
    esac
    echo ""
    echo "Stopping here. Python packages and the HuggingFace backend are already"
    echo "installed and working. Only the PowerInfer GGUF backend needs cmake."
    echo ""
    # Install powerquant Python package before exiting so HF backend works now
    pip install -e "$SCRIPT_DIR" 2>/dev/null || true
    echo "powerquant Python package installed. You can use:"
    echo "  powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit"
    exit 1
  fi

  pip install -r "$POWERINFER_DIR/requirements.txt" 2>/dev/null || true

  CMAKE_ARGS=(
    "cmake" "-S" "$POWERINFER_DIR" "-B" "$POWERINFER_DIR/build"
    "-DCMAKE_BUILD_TYPE=Release"
  )

  if $CPU_ONLY; then
    echo "Building CPU-only (--cpu-only specified)."
  elif $HAS_NVIDIA; then
    echo "Building with CUDA (NVIDIA GPU detected)."
    CMAKE_ARGS+=("-DLLAMA_CUBLAS=ON")
  elif command -v rocminfo &>/dev/null; then
    echo "Building with HIP (AMD GPU detected)."
    CMAKE_ARGS+=("-DLLAMA_HIPBLAS=ON")
  else
    echo "No GPU detected — building CPU-only."
  fi

  "${CMAKE_ARGS[@]}"
  N_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  cmake --build "$POWERINFER_DIR/build" --config Release -j"$N_CORES"
  echo "PowerInfer built at $POWERINFER_DIR/build"
fi

# ──────────────────────────────────────────────
# 4. Install powerquant package
# ──────────────────────────────────────────────
banner "Step 4/4: Installing powerquant"
pip install -e "$SCRIPT_DIR"

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
banner "Installation complete"

echo "Quick start:"
echo
echo "  # Interactive chat (HF model + TurboQuant KV compression)"
echo "  powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit"
echo
echo "  # Single generation"
echo "  powerquant generate --model gpt2 --prompt 'The future of AI is'"
echo
echo "  # Needle-in-haystack benchmark"
echo "  powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit"
echo
echo "  # Convert HF model to PowerInfer GGUF"
echo "  powerquant convert --model meta-llama/Llama-2-7b-hf --output llama2-7b.gguf"
echo
echo "  # Chat with PowerInfer GGUF model"
echo "  powerquant chat --model ./llama2-7b.gguf"
echo
echo "  # System info"
echo "  powerquant info"
echo
echo "Python API:"
echo "  from powerquant import Engine"
echo "  engine = Engine.from_pretrained('Qwen/Qwen2.5-3B-Instruct', load_in_4bit=True)"
echo "  print(engine.generate('Hello!'))"
echo
