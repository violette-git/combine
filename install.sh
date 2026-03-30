#!/usr/bin/env bash
# PowerQuant installer
# Installs Python dependencies, clones and builds PowerInfer, then installs powerquant.
#
# Usage:
#   bash install.sh              # full install (CUDA if NVIDIA detected)
#   bash install.sh --cpu-only   # CPU-only build of PowerInfer
#   bash install.sh --skip-powerinfer  # Python-only install (TurboQuant / HF backend only)
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

# Install CUDA-enabled PyTorch if a NVIDIA GPU is present
if ! $CPU_ONLY && command -v nvidia-smi &>/dev/null; then
  echo "NVIDIA GPU detected — installing CUDA PyTorch..."
  pip install torch --index-url https://download.pytorch.org/whl/cu128 || pip install torch
else
  echo "Installing CPU PyTorch..."
  pip install torch
fi

pip install -r "$SCRIPT_DIR/requirements.txt"

# ──────────────────────────────────────────────
# 2. Clone PowerInfer
# ──────────────────────────────────────────────
if $SKIP_POWERINFER; then
  echo "Skipping PowerInfer (--skip-powerinfer)."
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
  # 3. Build PowerInfer
  # ──────────────────────────────────────────
  banner "Step 3/4: Building PowerInfer"
  pip install -r "$POWERINFER_DIR/requirements.txt" 2>/dev/null || true

  CMAKE_ARGS=(
    "cmake" "-S" "$POWERINFER_DIR" "-B" "$POWERINFER_DIR/build"
    "-DCMAKE_BUILD_TYPE=Release"
  )

  if $CPU_ONLY; then
    echo "Building CPU-only."
  elif command -v nvidia-smi &>/dev/null; then
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
