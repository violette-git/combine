"""
PowerQuant CLI.

Commands:
    powerquant chat        — interactive chat (streams responses)
    powerquant generate    — single-prompt generation
    powerquant benchmark   — compare backends / TurboQuant configs
    powerquant convert     — convert HF model to PowerInfer GGUF
    powerquant build       — build PowerInfer C++ binary from source
    powerquant info        — show system info and backend availability
"""

import argparse
import sys
import os


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
    engine.chat(system_prompt=args.system)


def cmd_generate(args):
    from .engine import Engine
    from .config import EngineConfig, TurboQuantConfig

    config = EngineConfig(
        load_in_4bit=args.load_in_4bit,
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
        for token in engine.generate_stream(args.prompt):
            print(token, end="", flush=True)
        print()
    else:
        print(engine.generate(args.prompt))


def cmd_benchmark(args):
    """
    Run a needle-in-a-haystack benchmark comparing TurboQuant configs.
    Tests whether the model can retrieve a hidden string from a long context.
    """
    import torch
    import gc
    from .engine import Engine
    from .config import EngineConfig, TurboQuantConfig

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
        sys.exit(1)

    from .turboquant.cache import TurboQuantCache

    NEEDLE = "The secret project code name is AURORA-7749."
    EXPECTED = "AURORA-7749"
    FILLER = (
        "The quarterly financial review meeting covered several topics including "
        "budget allocations for the upcoming fiscal year, departmental spending "
        "reports, and projected revenue streams from various business units.\n\n"
    )

    print(f"\n{'='*70}")
    print("PowerQuant Benchmark: Needle-in-a-Haystack")
    print(f"Model: {args.model}")
    print(f"{'='*70}\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Loaded. Layers: {n_layers}\n")

    def build_prompt(target_tokens):
        filler_len = max(1, len(tokenizer.encode(FILLER)))
        n_reps = max(1, target_tokens // filler_len)
        needle_idx = n_reps // 2
        parts = []
        for i in range(n_reps):
            if i == needle_idx:
                parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
            parts.append(FILLER)
        haystack = "".join(parts)
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            msgs = [{"role": "user", "content":
                     f"Read this:\n\n{haystack}\n\nWhat is the secret code name? Answer with just the code name."}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"Document:\n\n{haystack}\n\nSecret code name:"

    configs = [
        {"label": "FP16 baseline", "turboquant": False},
        {"label": "K4/V2 (default)", "turboquant": True, "key_bits": 4, "value_bits": 2, "rw": 128},
        {"label": "K4/V4", "turboquant": True, "key_bits": 4, "value_bits": 4, "rw": 128},
        {"label": "K3/V2 (aggressive)", "turboquant": True, "key_bits": 3, "value_bits": 2, "rw": 128},
    ]

    results = {}
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    for ctx in context_lengths:
        print(f"Context: ~{ctx} tokens")
        print("-" * 50)
        results[ctx] = {}
        prompt = build_prompt(ctx)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx + 256)
        input_ids = inputs["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")

        for cfg in configs:
            cache = None
            if cfg["turboquant"]:
                cache = TurboQuantCache(
                    key_bits=cfg["key_bits"],
                    value_bits=cfg["value_bits"],
                    residual_window=cfg["rw"],
                    n_layers=n_layers,
                )

            print(f"  {cfg['label']:<30s}", end=" ", flush=True)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False,
                    past_key_values=cache,
                    use_cache=True,
                )
            new_toks = out[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            found = EXPECTED.lower() in response.lower()
            status = "FOUND" if found else "MISS"
            print(f"{status:5s}  \"{response[:50]}\"")
            results[ctx][cfg["label"]] = found
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Config':<30s}"
    for ctx in context_lengths:
        header += f"  {ctx:>6d}"
    print(header)
    print(f"  {'-'*30}", end="")
    for _ in context_lengths:
        print(f"  {'------':>6s}", end="")
    print()
    for cfg in configs:
        label = cfg["label"]
        row = f"  {label:<30s}"
        for ctx in context_lengths:
            found = results.get(ctx, {}).get(label, False)
            row += f"  {'FOUND':>6s}" if found else f"  {'MISS':>6s}"
        print(row)
    print(f"{'='*70}\n")


def cmd_convert(args):
    from .backends.powerinfer import PowerInferBackend

    backend = PowerInferBackend()
    output = backend.convert_model(
        args.model,
        args.output,
        quantize=not args.no_quantize,
    )
    print(f"Converted model saved to: {output}")


def cmd_build(args):
    from .backends.powerinfer import PowerInferBackend

    backend = PowerInferBackend()
    backend.build(use_cuda=not args.cpu_only, use_hip=args.hip)
    print("PowerInfer built successfully.")


def cmd_info(args):
    import platform
    print(f"\nPowerQuant System Info")
    print(f"{'='*50}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    try:
        import torch
        print(f"PyTorch:  {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1e9
                print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")
        else:
            print("  GPU: not available (CPU only)")
    except ImportError:
        print("PyTorch:  not installed")

    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: not installed")

    from .backends.powerinfer import PowerInferBackend
    pi = PowerInferBackend()
    if pi.is_available:
        print(f"PowerInfer: {pi.binary_path}")
    else:
        print("PowerInfer: not built (run: powerquant build)")

    print()


def _add_model_args(parser):
    parser.add_argument("--model", "-m", required=True,
                        help="HuggingFace model ID, local path, or .gguf file")


def _add_turboquant_args(parser):
    parser.add_argument("--no-turboquant", action="store_true",
                        help="Disable TurboQuant KV compression (use standard fp16 cache)")
    parser.add_argument("--key-bits", type=int, default=4,
                        help="Bit-width for key vectors (default: 4)")
    parser.add_argument("--value-bits", type=int, default=2,
                        help="Bit-width for value vectors (default: 2)")
    parser.add_argument("--residual-window", type=int, default=128,
                        help="Recent tokens kept uncompressed in fp16 (default: 128)")
    parser.add_argument("--protected-layers", type=int, default=4,
                        help="First/last N layers at full precision (default: 4)")


def _add_quant_args(parser):
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model weights in 4-bit (bitsandbytes NF4)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model weights in 8-bit (bitsandbytes)")


def _add_gen_args(parser):
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)


def main():
    parser = argparse.ArgumentParser(
        prog="powerquant",
        description="PowerQuant: PowerInfer sparse inference + TurboQuant KV compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat with 4-bit model + TurboQuant KV compression
  powerquant chat --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit

  # Single prompt generation
  powerquant generate --model gpt2 --prompt "The future of AI is"

  # Run needle-in-haystack benchmark
  powerquant benchmark --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit

  # Convert HF model to PowerInfer GGUF
  powerquant convert --model meta-llama/Llama-2-7b-hf --output llama2-7b.gguf

  # PowerInfer GGUF inference
  powerquant chat --model ./llama2-7b.gguf

  # System info
  powerquant info
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive chat session")
    _add_model_args(p_chat)
    _add_turboquant_args(p_chat)
    _add_quant_args(p_chat)
    _add_gen_args(p_chat)
    p_chat.add_argument("--system", default=None, help="System prompt")
    p_chat.set_defaults(func=cmd_chat)

    # generate
    p_gen = subparsers.add_parser("generate", help="Single prompt generation")
    _add_model_args(p_gen)
    p_gen.add_argument("--prompt", "-p", required=True, help="Input prompt")
    _add_turboquant_args(p_gen)
    _add_quant_args(p_gen)
    _add_gen_args(p_gen)
    p_gen.add_argument("--stream", action="store_true", help="Stream output tokens")
    p_gen.add_argument("--greedy", action="store_true", help="Greedy decoding (no sampling)")
    p_gen.set_defaults(func=cmd_generate)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Needle-in-haystack benchmark")
    _add_model_args(p_bench)
    _add_quant_args(p_bench)
    p_bench.add_argument("--context-lengths", default="2048,4096,8192",
                          help="Comma-separated context sizes to test (default: 2048,4096,8192)")
    p_bench.set_defaults(func=cmd_benchmark)

    # convert
    p_conv = subparsers.add_parser("convert", help="Convert HF model to PowerInfer GGUF")
    p_conv.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    p_conv.add_argument("--output", "-o", required=True, help="Output .gguf path")
    p_conv.add_argument("--no-quantize", action="store_true",
                        help="Skip INT4 quantization (keep fp16 GGUF)")
    p_conv.set_defaults(func=cmd_convert)

    # build
    p_build = subparsers.add_parser("build", help="Build PowerInfer C++ binary from source")
    p_build.add_argument("--cpu-only", action="store_true", help="Build without GPU support")
    p_build.add_argument("--hip", action="store_true", help="Build with AMD HIP/ROCm support")
    p_build.set_defaults(func=cmd_build)

    # info
    p_info = subparsers.add_parser("info", help="Show system info and backend availability")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
