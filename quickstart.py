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


def _find_cl_exe():
    """Find cl.exe from VS Build Tools via vswhere. Returns path or None."""
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        r"Microsoft Visual Studio\Installer\vswhere.exe",
    )
    if not os.path.isfile(vswhere):
        return None
    r = subprocess.run(
        [vswhere, "-latest", "-products", "*", "-property", "installationPath"],
        capture_output=True, text=True,
    )
    vs_path = r.stdout.strip()
    if not vs_path:
        return None
    # Walk VC/Tools/MSVC/<version>/bin/Hostx64/x64/cl.exe
    msvc_root = os.path.join(vs_path, "VC", "Tools", "MSVC")
    if not os.path.isdir(msvc_root):
        return None
    for ver in sorted(os.listdir(msvc_root), reverse=True):
        cl = os.path.join(msvc_root, ver, "bin", "Hostx64", "x64", "cl.exe")
        if os.path.isfile(cl):
            return cl
    return None


def _ensure_ninja():
    """Install ninja via pip if not present. Returns True if available."""
    if shutil.which("ninja"):
        return True
    print("  Installing ninja build system...")
    try:
        pip("install", "ninja")
        # pip-installed ninja lands in Scripts/; reload PATH
        scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
        return bool(shutil.which("ninja"))
    except subprocess.CalledProcessError:
        return False


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

    # Install PowerInfer Python deps, skipping local path entries (./gguf-py etc.)
    pi_req = os.path.join(POWERINFER_DIR, "requirements.txt")
    if os.path.isfile(pi_req):
        with open(pi_req) as f:
            reqs = [
                line.strip() for line in f
                if line.strip() and not line.startswith("#") and not line.startswith(".")
            ]
        if reqs:
            try:
                pip("install", *reqs)
            except subprocess.CalledProcessError:
                pass

    build_dir = os.path.join(POWERINFER_DIR, "build")
    cmake_args = [
        "cmake", "-S", POWERINFER_DIR, "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    extra_env = {}

    if platform.system() == "Windows":
        # cl.exe (MSVC) is not in PATH by default — find it via vswhere.
        cl = _find_cl_exe()
        if cl:
            print(f"  Found MSVC compiler: {cl}")
            extra_env["CC"] = cl
            extra_env["CXX"] = cl
            if _ensure_ninja():
                cmake_args += ["-G", "Ninja"]
                print("  Using Ninja + MSVC.")
        elif shutil.which("gcc"):
            # Fall back to MinGW GCC (bundled with Git for Windows)
            print("  MSVC not found — using MinGW GCC (CPU-only build).")
            extra_env["CC"] = shutil.which("gcc")
            extra_env["CXX"] = shutil.which("g++") or shutil.which("gcc")
            if _ensure_ninja():
                cmake_args += ["-G", "Ninja"]
            # MinGW + CUDA is unsupported; force CPU
            force_cpu = True
            has_nvidia = False
        else:
            print()
            print("  No C++ compiler found in PATH.")
            print("  VS Build Tools is installed but cl.exe couldn't be located.")
            print("  Try opening a 'Developer Command Prompt for VS 2022' and re-running.")
            print("  Skipping PowerInfer build. The HuggingFace backend still works.")
            return

    if force_cpu:
        print("  Building CPU-only.")
    elif has_nvidia:
        print("  Building with CUDA.")
        cmake_args.append("-DLLAMA_CUBLAS=ON")
    elif shutil.which("rocminfo"):
        print("  Building with HIP (AMD GPU).")
        cmake_args.append("-DLLAMA_HIPBLAS=ON")
    else:
        print("  No GPU detected — building CPU-only.")

    env = {**os.environ, **extra_env}
    try:
        print(f"  > {' '.join(cmake_args)}")
        subprocess.run(cmake_args, check=True, env=env)
    except subprocess.CalledProcessError:
        print()
        print("  ERROR: cmake configuration failed.")
        if platform.system() == "Windows":
            print("  Make sure the C++ workload is installed in VS Build Tools:")
            print("    Open 'Visual Studio Installer' → Modify → 'Desktop development with C++'")
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
        subprocess.run(
            ["cmake", "--build", build_dir, "--config", "Release", f"-j{cpu_count}"],
            check=True, env=env,
        )
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
    # Ensure setuptools is new enough to support editable installs via pyproject.toml.
    # Upgrade separately so a failure on one doesn't block the other.
    try:
        pip("install", "--upgrade", "setuptools>=68")
    except subprocess.CalledProcessError:
        pass
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation interrupted.")
        sys.exit(1)
