#!/usr/bin/env python3
"""
PowerQuant quickstart installer.
Works in ANY terminal on Windows, Mac, or Linux.

Usage:
    python quickstart.py                # auto-detect GPU, full install
    python quickstart.py --cuda         # force CUDA PyTorch
    python quickstart.py --cpu-only     # force CPU PyTorch
    python quickstart.py --skip-build   # skip building PowerInfer C++
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(REPO_ROOT, "vendor")
POWERINFER_DIR = os.path.join(VENDOR_DIR, "PowerInfer")
POWERINFER_REPO = "https://github.com/Tiiny-AI/PowerInfer"


def banner(msg):
    print()
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)
    print()


def run(cmd, check=True, capture=False):
    """Run a command, print it first, return CompletedProcess."""
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def pip(*args):
    run([sys.executable, "-m", "pip"] + list(args))


# ──────────────────────────────────────────────────────────────
# GPU detection
# ──────────────────────────────────────────────────────────────

def _nvidia_smi_works(path):
    try:
        r = subprocess.run([path], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def detect_nvidia():
    """Return True if an NVIDIA GPU is found."""
    # 1) nvidia-smi in PATH
    smi = shutil.which("nvidia-smi")
    if smi and _nvidia_smi_works(smi):
        print("  NVIDIA GPU detected via PATH (nvidia-smi).")
        return True

    # 2) Common Windows install location (often missing from PATH on laptops)
    win_smi = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if os.path.isfile(win_smi) and _nvidia_smi_works(win_smi):
        print("  NVIDIA GPU detected via NVSMI directory.")
        return True

    # 3) nvcc in PATH
    if shutil.which("nvcc"):
        print("  NVIDIA GPU detected via nvcc.")
        return True

    # 4) Windows wmic device list
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=10,
            )
            if "nvidia" in r.stdout.lower():
                print("  NVIDIA GPU detected via Windows device list.")
                return True
        except Exception:
            pass

    return False


# ──────────────────────────────────────────────────────────────
# Step 1 — PyTorch
# ──────────────────────────────────────────────────────────────

def install_pytorch(has_nvidia, force_cpu):
    banner("Step 1/4: Installing PyTorch")

    if force_cpu:
        print("  --cpu-only: installing CPU PyTorch.")
        pip("install", "torch")
        return

    if has_nvidia:
        print("  NVIDIA GPU found — installing CUDA PyTorch.")
        try:
            pip("install", "torch", "--index-url",
                "https://download.pytorch.org/whl/cu128")
            return
        except subprocess.CalledProcessError:
            print("  cu128 failed, trying cu121...")
        try:
            pip("install", "torch", "--index-url",
                "https://download.pytorch.org/whl/cu121")
            return
        except subprocess.CalledProcessError:
            print("  cu121 failed, falling back to CPU PyTorch.")

    print("  No NVIDIA GPU detected — installing CPU PyTorch.")
    print("  (Re-run with --cuda if your GPU was not detected.)")
    pip("install", "torch")


# ──────────────────────────────────────────────────────────────
# Step 2 — Python requirements
# ──────────────────────────────────────────────────────────────

def install_requirements():
    banner("Step 2/4: Installing Python requirements")
    req = os.path.join(REPO_ROOT, "requirements.txt")
    pip("install", "-r", req)


# ──────────────────────────────────────────────────────────────
# Step 3 — PowerInfer (optional C++ build)
# ──────────────────────────────────────────────────────────────

def clone_powerinfer():
    os.makedirs(VENDOR_DIR, exist_ok=True)
    if os.path.isdir(POWERINFER_DIR):
        print("  PowerInfer already cloned — pulling latest...")
        subprocess.run(
            ["git", "-C", POWERINFER_DIR, "pull", "--ff-only"],
            check=False,
        )
    else:
        run(["git", "clone", POWERINFER_REPO, POWERINFER_DIR])


def build_powerinfer(has_nvidia, force_cpu):
    banner("Step 3/4: Cloning & building PowerInfer")

    if not shutil.which("git"):
        print("  WARNING: git not found — skipping PowerInfer clone.")
        print("  Install Git from https://git-scm.com and re-run to get the PowerInfer backend.")
        return

    clone_powerinfer()

    if not shutil.which("cmake"):
        _print_cmake_help()
        print("  Skipping C++ build — the HuggingFace + TurboQuant backend still works.")
        return

    # Install PowerInfer Python deps if present
    pi_req = os.path.join(POWERINFER_DIR, "requirements.txt")
    if os.path.isfile(pi_req):
        try:
            pip("install", "-r", pi_req)
        except subprocess.CalledProcessError:
            pass

    build_dir = os.path.join(POWERINFER_DIR, "build")
    cmake_args = [
        "cmake", "-S", POWERINFER_DIR, "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    if force_cpu:
        print("  Building CPU-only (--cpu-only).")
    elif has_nvidia:
        print("  Building with CUDA.")
        cmake_args.append("-DLLAMA_CUBLAS=ON")
    elif shutil.which("rocminfo"):
        print("  Building with HIP (AMD GPU).")
        cmake_args.append("-DLLAMA_HIPBLAS=ON")
    else:
        print("  No GPU detected — building CPU-only.")

    try:
        run(cmake_args)
    except subprocess.CalledProcessError:
        print()
        print("  ERROR: cmake configuration failed.")
        print("  Make sure a C++ compiler is installed:")
        if platform.system() == "Windows":
            print("    winget install Microsoft.VisualStudio.2022.BuildTools")
            print("    (select 'Desktop development with C++' workload)")
        elif platform.system() == "Darwin":
            print("    xcode-select --install")
        else:
            print("    sudo apt-get install build-essential   # Ubuntu/Debian")
            print("    sudo dnf install gcc-c++               # Fedora")
        print()
        print("  Skipping PowerInfer build. The HuggingFace backend still works.")
        return

    cpu_count = os.cpu_count() or 4
    try:
        run(["cmake", "--build", build_dir, "--config", "Release",
             f"-j{cpu_count}"])
        print(f"  PowerInfer built at: {build_dir}")
    except subprocess.CalledProcessError:
        print("  ERROR: Build failed. Check output above.")
        print("  Skipping PowerInfer. The HuggingFace backend still works.")


def _print_cmake_help():
    print()
    print("  cmake not found — cannot build PowerInfer C++ backend.")
    print()
    system = platform.system()
    if system == "Windows":
        print("  To install cmake on Windows (pick one):")
        print("    winget install Kitware.CMake")
        print("    choco install cmake")
        print("    https://cmake.org/download/  (tick 'Add to PATH')")
        print()
        print("  You also need a C++ compiler:")
        print("    winget install Microsoft.VisualStudio.2022.BuildTools")
        print("    (select 'Desktop development with C++' workload)")
    elif system == "Darwin":
        print("  brew install cmake")
        print("  xcode-select --install")
    else:
        print("  sudo apt-get install cmake build-essential   # Ubuntu/Debian")
        print("  sudo dnf install cmake gcc-c++               # Fedora")
        print("  sudo pacman -S cmake base-devel              # Arch")
    print()
    print("  After installing cmake, re-run:  python quickstart.py")
    print()


# ──────────────────────────────────────────────────────────────
# Step 4 — install powerquant package
# ──────────────────────────────────────────────────────────────

def install_package():
    banner("Step 4/4: Installing powerquant package")
    pip("install", "-e", REPO_ROOT)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PowerQuant installer — works on Windows, Mac, and Linux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cuda", action="store_true",
                        help="Force CUDA PyTorch even if GPU not auto-detected")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU PyTorch and CPU PowerInfer build")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip PowerInfer C++ build (HuggingFace backend still works)")
    args = parser.parse_args()

    force_cpu = args.cpu_only

    print()
    print("PowerQuant Installer")
    print(f"Python {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print()

    # GPU detection
    if args.cuda:
        print("  --cuda: forcing NVIDIA GPU mode.")
        has_nvidia = True
    elif force_cpu:
        print("  --cpu-only: skipping GPU detection.")
        has_nvidia = False
    else:
        print("Detecting GPU...")
        has_nvidia = detect_nvidia()
        if not has_nvidia:
            print("  No NVIDIA GPU detected.")
            print("  Re-run with --cuda to force CUDA install if your GPU was missed.")

    # Run steps
    install_pytorch(has_nvidia, force_cpu)
    install_requirements()

    if args.skip_build:
        banner("Step 3/4: Skipping PowerInfer build (--skip-build)")
        print("  The HuggingFace + TurboQuant backend will still work.")
    else:
        build_powerinfer(has_nvidia, force_cpu)

    install_package()

    # Done
    banner("Installation complete!")
    print("Quick start:")
    print()
    print("  # Interactive chat with KV compression:")
    print("  powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit")
    print()
    print("  # Single generation:")
    print("  powerquant generate --model gpt2 --prompt 'The future of AI is'")
    print()
    print("  # Benchmark:")
    print("  powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit")
    print()
    print("  # System info (GPU + CUDA diagnosis):")
    print("  powerquant info")
    print()
    print("Python API:")
    print("  from powerquant import Engine")
    print("  engine = Engine.from_pretrained('Qwen/Qwen2.5-3B-Instruct', load_in_4bit=True)")
    print("  print(engine.generate('Hello!'))")
    print()


if __name__ == "__main__":
    main()
