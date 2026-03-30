"""
Example: benchmark TurboQuant compression configs with needle-in-a-haystack.

Tests whether the model can retrieve a hidden code name embedded in a long
document, under various KV cache compression settings.

Expected output:
  Context: ~2048 tokens
    FP16 baseline              FOUND  "AURORA-7749"
    K4/V2 (default ~5x)       FOUND  "AURORA-7749"
    K4/V4 (high quality)      FOUND  "AURORA-7749"
    K3/V2 (aggressive ~6x)    FOUND  "AURORA-7749"

Run:
    python examples/benchmark.py --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit
"""

import argparse
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from powerquant.turboquant.cache import TurboQuantCache


NEEDLE = "The secret project code name is AURORA-7749."
EXPECTED = "AURORA-7749"
FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending "
    "reports, and projected revenue streams. The committee discussed infrastructure "
    "upgrades and noted that maintenance schedules should be coordinated with the "
    "facilities management team. Several action items were assigned to team leads.\n\n"
)

CONFIGS = [
    {"label": "FP16 baseline",      "turboquant": False},
    {"label": "K4/V2 (default)",    "turboquant": True, "key_bits": 4, "value_bits": 2, "rw": 128},
    {"label": "K4/V4 (quality)",    "turboquant": True, "key_bits": 4, "value_bits": 4, "rw": 128},
    {"label": "K3/V2 (aggressive)", "turboquant": True, "key_bits": 3, "value_bits": 2, "rw": 128},
    {"label": "K4/V2 no window",    "turboquant": True, "key_bits": 4, "value_bits": 2, "rw": 0},
]


def build_prompt(tokenizer, target_tokens: int) -> str:
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
                 f"What is the secret project code name? Answer with just the code name."}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return (
        f"Document:\n\n{haystack}\n\n"
        f"Question: What is the secret project code name?\nAnswer:"
    )


def run_config(model, tokenizer, input_ids, cfg: dict, n_layers: int, device) -> tuple[bool, str]:
    cache = None
    if cfg["turboquant"]:
        cache = TurboQuantCache(
            key_bits=cfg["key_bits"],
            value_bits=cfg["value_bits"],
            residual_window=cfg["rw"],
            n_layers=n_layers,
        )

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
    return found, response


def main():
    parser = argparse.ArgumentParser(description="TurboQuant KV benchmark")
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model ID")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--context-lengths", default="2048,4096",
                        help="Comma-separated context sizes (default: 2048,4096)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    print(f"\n{'='*70}")
    print("TurboQuant KV Compression Benchmark")
    print(f"Model: {args.model}")
    print(f"{'='*70}\n")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    n_layers = model.config.num_hidden_layers

    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated() // 1024 // 1024
        print(f"Loaded. GPU memory: {mem_mb} MB, Layers: {n_layers}\n")

    results: dict = {}
    for ctx in context_lengths:
        print(f"Context: ~{ctx} tokens")
        print("-" * 60)
        results[ctx] = {}

        prompt = build_prompt(tokenizer, ctx)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=ctx + 512).to(device)
        input_ids = inputs["input_ids"]
        actual_tokens = input_ids.shape[1]
        print(f"  (actual input tokens: {actual_tokens})\n")

        for cfg in CONFIGS:
            label = cfg["label"]
            print(f"  {label:<30s}", end=" ", flush=True)
            try:
                found, response = run_config(model, tokenizer, input_ids, cfg, n_layers, device)
                status = "FOUND" if found else "MISS "
                safe = response[:55].encode("ascii", errors="replace").decode("ascii")
                print(f"{status}  \"{safe}\"")
                results[ctx][label] = found
            except Exception as e:
                print(f"ERROR: {e}")
                results[ctx][label] = False

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (FOUND = model retrieved the secret code name)")
    print(f"{'='*70}")
    header = f"  {'Config':<30s}"
    for ctx in context_lengths:
        header += f"  {ctx:>6d}t"
    print(header)
    for cfg in CONFIGS:
        label = cfg["label"]
        row = f"  {label:<30s}"
        for ctx in context_lengths:
            found = results.get(ctx, {}).get(label, False)
            row += f"  {'FOUND  ':>8s}" if found else f"  {'MISS   ':>8s}"
        print(row)
    print(f"{'='*70}\n")

    # Memory report
    print("KV Cache Memory (per layer, S=4096, B=1, H=32, D=128):")
    print(f"  {'Config':<30s}  {'Compressed':>12s}  {'FP16 equiv':>12s}  {'Ratio':>6s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*6}")
    from powerquant.turboquant.compressors_v3 import TurboQuantV3
    for cfg in CONFIGS:
        if not cfg["turboquant"]:
            fp16_bytes = 1 * 32 * 4096 * 128 * 2 * 2
            print(f"  {'FP16 baseline':<30s}  {fp16_bytes/1e6:>10.1f}MB  {fp16_bytes/1e6:>10.1f}MB  {'1.00':>6s}x")
            continue
        comp = TurboQuantV3(
            head_dim=128, key_bits=cfg["key_bits"], value_bits=cfg["value_bits"],
            residual_window=cfg["rw"], layer_idx=8, n_layers=n_layers
        )
        info = comp.memory_bytes(B=1, H=32, S=4096)
        ratio = f"{info['compression_ratio']:.2f}"
        print(
            f"  {cfg['label']:<30s}  "
            f"{info['compressed_bytes']/1e6:>10.1f}MB  "
            f"{info['fp16_bytes']/1e6:>10.1f}MB  "
            f"{ratio:>6s}x"
        )
    print()


if __name__ == "__main__":
    main()
