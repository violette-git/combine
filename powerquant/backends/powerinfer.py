"""
PowerInfer backend: wraps the PowerInfer C++ inference engine.

PowerInfer achieves up to 11x speedup vs llama.cpp by exploiting sparse
neuron activation patterns (hot neurons on GPU, cold on CPU).

Supported model formats: PowerInfer GGUF (.gguf files with activation statistics).
Supported base models: Falcon-40B, LLaMA 2, ProSparse, Bamboo-7B, and variants.

To convert a HuggingFace model to PowerInfer GGUF format, use:
    backend = PowerInferBackend(...)
    backend.convert_model("mistralai/Mistral-7B-v0.1", "mistral-7b.gguf")
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass


@dataclass
class PowerInferGenConfig:
    """Generation parameters for PowerInfer inference."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_threads: int = 8
    vram_budget_gb: Optional[float] = None
    n_gpu_layers: int = -1  # -1 = load as many as VRAM allows
    context_size: int = 4096


class PowerInferBackend:
    """
    Python wrapper around the PowerInfer C++ inference binary.

    PowerInfer is not a Python library — it is a compiled C++ binary.
    This class manages:
      - Locating or building the binary
      - Running inference via subprocess
      - Model conversion from HuggingFace to PowerInfer GGUF format
      - Streaming token output

    The binary communicates via stdin/stdout in a simple line-based protocol.
    """

    # Standard locations to search for the binary
    _BINARY_NAMES = ["main", "powerinfer", "llama-cli"]
    _DEFAULT_BUILD_DIR = Path(__file__).parent.parent.parent / "vendor" / "PowerInfer" / "build"
    _DEFAULT_VENDOR_DIR = Path(__file__).parent.parent.parent / "vendor" / "PowerInfer"

    def __init__(
        self,
        binary_path: Optional[str] = None,
        vendor_dir: Optional[str] = None,
    ):
        """
        Args:
            binary_path: explicit path to the PowerInfer main binary.
                         If None, auto-detected from build directory or PATH.
            vendor_dir: path to the PowerInfer source checkout.
                        If None, defaults to vendor/PowerInfer relative to this package.
        """
        self.vendor_dir = Path(vendor_dir) if vendor_dir else self._DEFAULT_VENDOR_DIR
        self.binary_path = self._resolve_binary(binary_path)

    def _resolve_binary(self, explicit: Optional[str]) -> Optional[Path]:
        """Locate the PowerInfer binary."""
        if explicit:
            p = Path(explicit)
            if p.exists():
                return p
            raise FileNotFoundError(f"PowerInfer binary not found at: {explicit}")

        # Search build directory
        build_dir = self._DEFAULT_BUILD_DIR
        for name in self._BINARY_NAMES:
            for candidate in [
                build_dir / "bin" / name,
                build_dir / "bin" / f"{name}.exe",
                build_dir / name,
            ]:
                if candidate.exists():
                    return candidate

        # Search PATH
        for name in self._BINARY_NAMES:
            found = shutil.which(name)
            if found:
                return Path(found)

        return None  # Not found — will raise when inference is attempted

    @property
    def is_available(self) -> bool:
        """True if the PowerInfer binary is present and executable."""
        return self.binary_path is not None and self.binary_path.exists()

    def _require_binary(self):
        if not self.is_available:
            raise RuntimeError(
                "PowerInfer binary not found. Build it first:\n\n"
                "    bash install.sh\n\n"
                "Or run: powerquant build-powerinfer"
            )

    def build(self, use_cuda: bool = True, use_hip: bool = False) -> bool:
        """
        Build PowerInfer from source (requires cmake and a C++ compiler).

        Returns True on success. Raises on failure.
        """
        if not self.vendor_dir.exists():
            raise RuntimeError(
                f"PowerInfer source not found at {self.vendor_dir}.\n"
                "Run: git clone https://github.com/Tiiny-AI/PowerInfer vendor/PowerInfer"
            )

        build_dir = self.vendor_dir / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            "cmake", "-S", str(self.vendor_dir), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if use_cuda:
            cmake_args.append("-DLLAMA_CUBLAS=ON")
        elif use_hip:
            cmake_args.append("-DLLAMA_HIPBLAS=ON")

        print("Configuring PowerInfer with CMake...")
        subprocess.run(cmake_args, check=True)

        n_cores = os.cpu_count() or 4
        print(f"Building PowerInfer ({n_cores} cores)...")
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", f"-j{n_cores}"],
            check=True,
        )

        # Refresh binary path after build
        self.binary_path = self._resolve_binary(None)
        if not self.is_available:
            raise RuntimeError("Build succeeded but binary not found. Check build output.")

        print(f"PowerInfer binary: {self.binary_path}")
        return True

    def generate(
        self,
        model_path: str,
        prompt: str,
        config: Optional[PowerInferGenConfig] = None,
    ) -> str:
        """
        Run inference and return the generated text.

        Args:
            model_path: path to a .gguf model file
            prompt: input text
            config: generation parameters

        Returns:
            Generated text string (not including the prompt)
        """
        tokens = list(self.generate_stream(model_path, prompt, config))
        return "".join(tokens)

    def generate_stream(
        self,
        model_path: str,
        prompt: str,
        config: Optional[PowerInferGenConfig] = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens one at a time.

        Yields token strings as they are produced by PowerInfer.
        """
        self._require_binary()
        cfg = config or PowerInferGenConfig()

        cmd = self._build_command(model_path, prompt, cfg)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        prompt_done = False
        buffer = ""
        try:
            for char in iter(lambda: process.stdout.read(1), ""):
                buffer += char
                # PowerInfer echoes the prompt then outputs a separator
                if not prompt_done:
                    if prompt in buffer:
                        prompt_done = True
                        buffer = ""
                    continue
                yield char
        finally:
            process.wait()
            if process.returncode != 0:
                stderr = process.stderr.read()
                raise RuntimeError(f"PowerInfer exited with code {process.returncode}:\n{stderr}")

    def _build_command(self, model_path: str, prompt: str, cfg: PowerInferGenConfig) -> list[str]:
        cmd = [
            str(self.binary_path),
            "-m", model_path,
            "-p", prompt,
            "-n", str(cfg.max_new_tokens),
            "-t", str(cfg.n_threads),
            "--temp", str(cfg.temperature),
            "--top-p", str(cfg.top_p),
            "--top-k", str(cfg.top_k),
            "--repeat-penalty", str(cfg.repeat_penalty),
            "-c", str(cfg.context_size),
            "--no-display-prompt",
        ]
        if cfg.vram_budget_gb is not None:
            cmd += ["--vram-budget", str(cfg.vram_budget_gb)]
        if cfg.n_gpu_layers != -1:
            cmd += ["-ngl", str(cfg.n_gpu_layers)]
        return cmd

    def convert_model(
        self,
        hf_model_name: str,
        output_path: str,
        quantize: bool = True,
    ) -> str:
        """
        Convert a HuggingFace model to PowerInfer GGUF format.

        Args:
            hf_model_name: HuggingFace model ID (e.g. "meta-llama/Llama-2-7b-hf")
            output_path: destination .gguf file path
            quantize: if True, apply INT4 quantization after conversion

        Returns:
            Path to the output .gguf file
        """
        convert_script = self.vendor_dir / "convert-hf-to-powerinfer-gguf.py"
        if not convert_script.exists():
            convert_script = self.vendor_dir / "convert.py"
        if not convert_script.exists():
            raise FileNotFoundError(
                f"Conversion script not found in {self.vendor_dir}. "
                "Make sure PowerInfer is cloned to vendor/PowerInfer."
            )

        print(f"Converting {hf_model_name} -> {output_path} ...")
        subprocess.run(
            [sys.executable, str(convert_script), hf_model_name, "--outfile", output_path],
            check=True,
        )

        if quantize:
            quantize_bin = self.binary_path.parent / "quantize"
            if quantize_bin.exists():
                quantized_path = output_path.replace(".gguf", "-q4.gguf")
                print(f"Quantizing to INT4: {quantized_path}")
                subprocess.run(
                    [str(quantize_bin), output_path, quantized_path, "q4_0"],
                    check=True,
                )
                return quantized_path

        return output_path

    def list_models(self, search_dir: str = ".") -> list[str]:
        """Return paths of all .gguf files found under search_dir."""
        return [str(p) for p in Path(search_dir).rglob("*.gguf")]
