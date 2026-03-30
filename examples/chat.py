"""
Example: interactive chat with PowerQuant.

Demonstrates:
  - HF backend with TurboQuant KV compression + 4-bit weight quantization
  - PowerInfer backend with a local GGUF model

Run:
    # HF model (downloads from HuggingFace Hub)
    python examples/chat.py --model Qwen/Qwen2.5-3B-Instruct --load-in-4bit

    # Local GGUF model (PowerInfer backend)
    python examples/chat.py --model /path/to/model.gguf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from powerquant import Engine


def main():
    parser = argparse.ArgumentParser(description="PowerQuant chat example")
    parser.add_argument("--model", "-m", required=True,
                        help="HuggingFace model ID or .gguf path")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load weights in 4-bit NF4 (HF backend)")
    parser.add_argument("--no-turboquant", action="store_true",
                        help="Disable TurboQuant KV compression")
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--value-bits", type=int, default=2)
    parser.add_argument("--residual-window", type=int, default=128)
    parser.add_argument("--system", default="You are a helpful assistant.",
                        help="System prompt")
    args = parser.parse_args()

    engine = Engine.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        use_turboquant=not args.no_turboquant,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        residual_window=args.residual_window,
    )

    engine.chat(system_prompt=args.system)


if __name__ == "__main__":
    main()
