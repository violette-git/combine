"""
PowerQuant CLI — powerquant <command> [options]

Commands:
    chat        Interactive chat session with streaming output and optional metrics
    generate    Single prompt generation, returns text + optional metrics table
    benchmark   Needle-in-a-haystack accuracy test across compression configs
    convert     Convert a HuggingFace model to PowerInfer GGUF format
    build       Build the PowerInfer C++ binary from source
    info        Show system info: GPU, PyTorch, PowerInfer availability

Run ``powerquant <command> --help`` for per-command options.
"""

import argparse
import json
import sys
import os


# ── chat ──────────────────────────────────────────────────────────────────────

def cmd_chat(args):
    from .engine import Engine
    from .config import EngineConfig, TurboQuantConfig

    config = EngineConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_turboquant=not args.no_turboquant,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        turboquant=TurboQuantConfig(
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            residual_window=args.residual_window,
            protected_layers=args.protected_layers,
        ),
    )
    engine = Engine(args.model, config)
    engine.chat(system_prompt=args.system, show_metrics=args.metrics)


# ── generate ──────────────────────────────────────────────────────────────────

def cmd_generate(args):
    from .engine import Engine
    from .config import EngineConfig, TurboQuantConfig

    config = EngineConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_turboquant=not args.no_turboquant,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
        turboquant=TurboQuantConfig(
            key_bits=args.key_bits,
            value_bits=args.value_bits,
            residual_window=args.residual_window,
        ),
    )
    engine = Engine(args.model, config)

    if args.stream:
        # Streaming: print tokens live, no metrics available inline
        for token in engine.generate_stream(args.prompt):
            print(token, end="", flush=True)
        print()
        return

    result = engine.generate_with_metrics(args.prompt)
    print(result.text)

    if args.metrics:
        result.print_report()
    elif not args.quiet:
        summary = result.summary()
        if summary:
            print(f"\n{summary}", file=sys.stderr)

    if args.json:
        metrics_dict = result.to_dict()
        metrics_dict["text"] = result.text
        print(json.dumps(metrics_dict, indent=2))


# ── benchmark ─────────────────────────────────────────────────────────────────

def cmd_benchmark(args):
    """
    Needle-in-a-haystack accuracy benchmark comparing TurboQuant compression configs.

    Embeds a short "needle" string into a long filler document, then tests whether
    the model can retrieve the needle under each compression configuration.

    Output columns:
      FOUND = model correctly retrieved the needle
      MISS  = model failed to retrieve the needle
    """
    import gc
    import time
    import torch

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("transformers is required: pip install transformers")
        sys.exit(1)

    from .turboquant.cache import TurboQuantCache
    from .turboquant.compressors_v3 import TurboQuantV3

    NEEDLE = "The secret project code name is AURORA-7749."
    EXPECTED = "AURORA-7749"
    FILLER = (
        "The quarterly financial review meeting covered several topics including "
        "budget allocations for the upcoming fiscal year, departmental spending "
        "reports, and projected revenue streams from various business units. "
        "The committee discussed infrastructure upgrades for the western regional "
        "offices and noted that maintenance schedules should be coordinated with "
        "facilities management. Several action items were assigned to team leads.\n\n"
    )

    benchmark_configs = [
        {"label": "FP16 baseline",      "turboquant": False},
        {"label": "K4/V2  (~5.1x)",     "turboquant": True, "key_bits": 4, "value_bits": 2, "rw": 128},
        {"label": "K4/V4  (~3.2x)",     "turboquant": True, "key_bits": 4, "value_bits": 4, "rw": 128},
        {"label": "K3/V2  (~6.1x)",     "turboquant": True, "key_bits": 3, "value_bits": 2, "rw": 128},
        {"label": "K4/V2  no window",   "turboquant": True, "key_bits": 4, "value_bits": 2, "rw": 0},
    ]

    print(f"\n{'='*72}")
    print("  PowerQuant Benchmark: Needle-in-a-Haystack")
    print(f"  Model : {args.model}")
    print(f"{'='*72}\n")

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    n_layers = model.config.num_hidden_layers
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )

    mem_info = ""
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() // 1024 // 1024
        mem_info = f"  GPU after load : {mem_mb} MB"
    print(f"  Layers : {n_layers}  |  Head dim : {head_dim}")
    if mem_info:
        print(mem_info)
    print()

    def build_prompt(target_tokens: int) -> str:
        filler_len = max(1, len(tokenizer.encode(FILLER)))
        n_reps = max(1, target_tokens // filler_len)
        needle_idx = n_reps // 2
        parts = []
        for i in range(n_reps):
            if i == needle_idx:
                parts.append(f"\n--- Internal Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
            parts.append(FILLER)
        haystack = "".join(parts)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            msgs = [{"role": "user", "content":
                     f"Read this document:\n\n{haystack}\n\n"
                     f"What is the secret project code name? "
                     f"Answer with just the code name."}]
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"Document:\n\n{haystack}\n\nSecret code name:"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    results: dict = {}
    timing: dict = {}
    memory: dict = {}

    for ctx in context_lengths:
        print(f"  Context : ~{ctx} tokens")
        print(f"  {'─'*62}")
        results[ctx] = {}
        timing[ctx] = {}
        memory[ctx] = {}

        prompt = build_prompt(ctx)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=ctx + 512
        )
        input_ids = inputs["input_ids"].to(device)
        actual = input_ids.shape[1]
        print(f"  Actual input tokens : {actual}\n")

        for cfg in benchmark_configs:
            label = cfg["label"]
            cache = None
            if cfg["turboquant"]:
                cache = TurboQuantCache(
                    key_bits=cfg["key_bits"],
                    value_bits=cfg["value_bits"],
                    residual_window=cfg["rw"],
                    n_layers=n_layers,
                )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_before = torch.cuda.memory_allocated() // (1024 * 1024)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False,
                    past_key_values=cache,
                    use_cache=True,
                )
            elapsed = time.perf_counter() - t0

            gpu_peak = gpu_after = 0
            if torch.cuda.is_available():
                gpu_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
                gpu_after = torch.cuda.memory_allocated() // (1024 * 1024)

            new_toks = out[0][input_ids.shape[1]:]
            n_out = new_toks.shape[0]
            response = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            found = EXPECTED.lower() in response.lower()
            tps = n_out / elapsed if elapsed > 0 else 0

            results[ctx][label] = found
            timing[ctx][label] = {"latency_s": round(elapsed, 2), "tok_per_s": round(tps, 1)}
            memory[ctx][label] = {
                "gpu_before_mb": gpu_before if torch.cuda.is_available() else 0,
                "gpu_peak_mb": gpu_peak,
                "gpu_after_mb": gpu_after,
            }

            # KV compression ratio for this config
            ratio_str = "n/a"
            if cfg["turboquant"] and actual > cfg["rw"]:
                comp = TurboQuantV3(
                    head_dim=head_dim,
                    key_bits=cfg["key_bits"],
                    value_bits=cfg["value_bits"],
                    residual_window=cfg["rw"],
                    layer_idx=n_layers // 2,
                    n_layers=n_layers,
                )
                kv_info = comp.memory_bytes(B=1, H=model.config.num_attention_heads, S=actual)
                ratio_str = f"{kv_info['compression_ratio']:.1f}x"

            status = "FOUND" if found else "MISS "
            safe = response[:42].encode("ascii", errors="replace").decode("ascii")
            print(
                f"  {label:<22s}  {status}  "
                f"{elapsed:5.1f}s  {tps:5.1f} t/s  KV:{ratio_str:>5s}"
                f"  \"{safe}\""
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    # ── Accuracy summary table ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  ACCURACY SUMMARY")
    print(f"{'='*72}")
    header = f"  {'Config':<22s}"
    for ctx in context_lengths:
        header += f"  {ctx:>7d}t"
    print(header)
    print(f"  {'─'*22}", end="")
    for _ in context_lengths:
        print(f"  {'─'*8}", end="")
    print()
    for cfg in benchmark_configs:
        label = cfg["label"]
        row = f"  {label:<22s}"
        for ctx in context_lengths:
            found = results.get(ctx, {}).get(label, False)
            row += f"  {'FOUND   ':>8s}" if found else f"  {'MISS    ':>8s}"
        print(row)

    # ── Throughput summary table ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  THROUGHPUT SUMMARY (tok/s, 32 new tokens)")
    print(f"{'='*72}")
    header = f"  {'Config':<22s}"
    for ctx in context_lengths:
        header += f"  {ctx:>7d}t"
    print(header)
    print(f"  {'─'*22}", end="")
    for _ in context_lengths:
        print(f"  {'─'*8}", end="")
    print()
    for cfg in benchmark_configs:
        label = cfg["label"]
        row = f"  {label:<22s}"
        for ctx in context_lengths:
            tps = timing.get(ctx, {}).get(label, {}).get("tok_per_s", 0)
            row += f"  {tps:>7.1f} "
        print(row)
    print(f"{'='*72}\n")

    if args.json:
        out_data = {
            "model": args.model,
            "context_lengths": context_lengths,
            "results": {str(k): v for k, v in results.items()},
            "timing": {str(k): v for k, v in timing.items()},
            "memory": {str(k): v for k, v in memory.items()},
        }
        print(json.dumps(out_data, indent=2))


# ── convert ───────────────────────────────────────────────────────────────────

def cmd_convert(args):
    from .backends.powerinfer import PowerInferBackend

    backend = PowerInferBackend()
    output = backend.convert_model(
        args.model,
        args.output,
        quantize=not args.no_quantize,
    )
    print(f"\nConverted model saved to: {output}")


# ── build ─────────────────────────────────────────────────────────────────────

def cmd_build(args):
    from .backends.powerinfer import PowerInferBackend

    backend = PowerInferBackend()
    backend.build(use_cuda=not args.cpu_only, use_hip=args.hip)
    print("PowerInfer built successfully.")


# ── info ──────────────────────────────────────────────────────────────────────

def cmd_info(args):
    import platform
    from . import __version__

    W = 54
    print(f"\n{'─'*W}")
    print(f"  PowerQuant v{__version__}")
    print(f"{'─'*W}")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  Platform  : {platform.system()} {platform.machine()}")

    try:
        import torch
        print(f"  PyTorch   : {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / 1e9
                free_mb = (props.total_memory - torch.cuda.memory_allocated(i)) // (1024 * 1024)
                print(f"    GPU {i}  : {props.name}  ({total_gb:.1f} GB total, {free_mb} MB free)")
        else:
            print("  GPU       : not available (CPU only)")
    except ImportError:
        print("  PyTorch   : not installed")

    try:
        import transformers
        print(f"  Transformers : {transformers.__version__}")
    except ImportError:
        print("  Transformers : not installed")

    try:
        import bitsandbytes
        print(f"  bitsandbytes : {bitsandbytes.__version__}")
    except ImportError:
        print("  bitsandbytes : not installed (4-bit/8-bit weights unavailable)")

    from .backends.powerinfer import PowerInferBackend
    pi = PowerInferBackend()
    if pi.is_available:
        print(f"  PowerInfer   : {pi.binary_path}  [ready]")
    else:
        print("  PowerInfer   : not built  (run: powerquant build)")

    print(f"{'─'*W}\n")


# ── argument helpers ──────────────────────────────────────────────────────────

def _add_model_args(parser):
    parser.add_argument(
        "--model", "-m", required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct), "
             "local model directory, or path to a .gguf file",
    )


def _add_turboquant_args(parser):
    parser.add_argument(
        "--no-turboquant", action="store_true",
        help="Disable TurboQuant KV compression; use a standard fp16 cache instead",
    )
    parser.add_argument(
        "--key-bits", type=int, default=4, metavar="N",
        help="Bit-width for compressed attention key vectors (default: 4). "
             "Valid: 2–8. Higher = better quality, less compression.",
    )
    parser.add_argument(
        "--value-bits", type=int, default=2, metavar="N",
        help="Bit-width for compressed attention value vectors (default: 2). "
             "Valid: 2–8. Can be lower than key-bits as values are less sensitive.",
    )
    parser.add_argument(
        "--residual-window", type=int, default=128, metavar="N",
        help="Number of recent tokens kept uncompressed in fp16 (default: 128). "
             "Larger window = better generation quality, slightly more memory.",
    )
    parser.add_argument(
        "--protected-layers", type=int, default=4, metavar="N",
        help="First/last N transformer layers use full precision (default: 4). "
             "These layers are most sensitive to quantization error.",
    )


def _add_quant_args(parser):
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model weights in 4-bit NF4 via bitsandbytes (recommended). "
             "Reduces weight memory ~4x with minimal quality loss.",
    )
    g.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model weights in 8-bit via bitsandbytes. "
             "Reduces weight memory ~2x.",
    )


def _add_gen_args(parser):
    parser.add_argument("--max-new-tokens", type=int, default=512, metavar="N",
                        help="Maximum number of tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7). "
                             "Lower = more deterministic, higher = more creative.")


def _add_metrics_args(parser):
    parser.add_argument(
        "--metrics", action="store_true",
        help="Print a detailed metrics report after generation.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output metrics as a JSON object to stdout (machine-readable).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the one-line metric summary printed to stderr.",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="powerquant",
        description=(
            "PowerQuant: efficient LLM inference via PowerInfer sparse activation "
            "+ TurboQuant KV cache compression."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━ Examples ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Interactive chat (HF model, 4-bit weights, TurboQuant KV compression):
    powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit --metrics

  Single generation with metrics:
    powerquant generate --model gpt2 --prompt "The future of AI" --metrics

  Stream output without metrics:
    powerquant generate --model gpt2 --prompt "The future of AI" --stream

  Export metrics as JSON:
    powerquant generate --model gpt2 --prompt "Hello" --json

  Needle-in-haystack benchmark across 2K/4K/8K contexts:
    powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit

  Convert HuggingFace model to PowerInfer GGUF:
    powerquant convert --model meta-llama/Llama-2-7b-hf --output llama2.gguf

  Chat with PowerInfer GGUF model:
    powerquant chat --model ./llama2.gguf

  System info:
    powerquant info
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── chat ──────────────────────────────────────────────────────────────────
    p_chat = subparsers.add_parser(
        "chat",
        help="Interactive chat session with streaming output",
        description=(
            "Start an interactive chat. Responses are streamed token-by-token. "
            "During the session type 'metrics' to toggle per-turn stats, "
            "'clear' to reset history, or 'exit'/'quit' to end."
        ),
    )
    _add_model_args(p_chat)
    _add_turboquant_args(p_chat)
    _add_quant_args(p_chat)
    _add_gen_args(p_chat)
    p_chat.add_argument("--system", default=None, metavar="TEXT",
                        help="System prompt prepended to every turn")
    p_chat.add_argument("--metrics", action="store_true",
                        help="Print one-line metric summary after each response")
    p_chat.set_defaults(func=cmd_chat)

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = subparsers.add_parser(
        "generate",
        help="Single prompt → response with optional metrics",
        description=(
            "Run a single generation. Use --metrics for a full report or "
            "--json to get machine-readable metrics output."
        ),
    )
    _add_model_args(p_gen)
    p_gen.add_argument("--prompt", "-p", required=True, metavar="TEXT",
                       help="Input prompt text")
    _add_turboquant_args(p_gen)
    _add_quant_args(p_gen)
    _add_gen_args(p_gen)
    _add_metrics_args(p_gen)
    p_gen.add_argument("--stream", action="store_true",
                       help="Stream output tokens as they are generated "
                            "(incompatible with --metrics / --json)")
    p_gen.add_argument("--greedy", action="store_true",
                       help="Use greedy decoding instead of sampling")
    p_gen.set_defaults(func=cmd_generate)

    # ── benchmark ─────────────────────────────────────────────────────────────
    p_bench = subparsers.add_parser(
        "benchmark",
        help="Needle-in-haystack accuracy + throughput benchmark",
        description=(
            "Embeds a secret code name into a long filler document and tests whether "
            "the model retrieves it correctly under each TurboQuant compression config. "
            "Outputs accuracy (FOUND/MISS) and throughput (tok/s) tables."
        ),
    )
    _add_model_args(p_bench)
    _add_quant_args(p_bench)
    p_bench.add_argument(
        "--context-lengths", default="2048,4096,8192", metavar="LIST",
        help="Comma-separated context sizes to test (default: 2048,4096,8192)",
    )
    p_bench.add_argument("--json", action="store_true",
                         help="Also print full results as JSON to stdout")
    p_bench.set_defaults(func=cmd_benchmark)

    # ── convert ───────────────────────────────────────────────────────────────
    p_conv = subparsers.add_parser(
        "convert",
        help="Convert a HuggingFace model to PowerInfer GGUF format",
        description=(
            "Downloads a HuggingFace model and converts it to PowerInfer GGUF format "
            "for use with the C++ backend. Optionally applies INT4 quantization."
        ),
    )
    p_conv.add_argument("--model", "-m", required=True, metavar="MODEL_ID",
                        help="HuggingFace model ID (e.g. meta-llama/Llama-2-7b-hf)")
    p_conv.add_argument("--output", "-o", required=True, metavar="PATH",
                        help="Output .gguf file path")
    p_conv.add_argument("--no-quantize", action="store_true",
                        help="Skip INT4 quantization and keep the GGUF in fp16")
    p_conv.set_defaults(func=cmd_convert)

    # ── build ─────────────────────────────────────────────────────────────────
    p_build = subparsers.add_parser(
        "build",
        help="Build the PowerInfer C++ inference binary from source",
        description=(
            "Compiles the PowerInfer C++ engine using CMake. Requires cmake, "
            "a C++17 compiler, and optionally CUDA toolkit for GPU support. "
            "The source must be cloned to vendor/PowerInfer (done by install.sh)."
        ),
    )
    p_build.add_argument("--cpu-only", action="store_true",
                         help="Build without any GPU acceleration")
    p_build.add_argument("--hip", action="store_true",
                         help="Build with AMD HIP/ROCm support instead of CUDA")
    p_build.set_defaults(func=cmd_build)

    # ── info ──────────────────────────────────────────────────────────────────
    p_info = subparsers.add_parser(
        "info",
        help="Show system info and backend availability",
    )
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
