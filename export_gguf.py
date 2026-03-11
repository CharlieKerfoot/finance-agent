"""
Merge LoRA weights and export to GGUF format for inference.

Usage:
  uv run export_gguf.py
  uv run export_gguf.py --model ckerf/arbagent-llama3-8b-lora --quantize q4_k_m
"""

import argparse
import subprocess
import sys
from pathlib import Path

from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF")
    parser.add_argument(
        "--model", default="ckerf/arbagent-llama3-8b-lora", help="HF model to export"
    )
    parser.add_argument(
        "--merged-dir", default="arbagent_merged_16bit", help="Directory for merged weights"
    )
    parser.add_argument(
        "--quantize", default="q4_k_m", help="GGUF quantization type (e.g. q4_k_m, q8_0, f16)"
    )
    parser.add_argument(
        "--output", default="arbagent-q4.gguf", help="Output GGUF filename"
    )
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    f16_gguf = merged_dir.parent / "arbagent-f16.gguf"

    # Step 1: Merge LoRA into base model
    print(f"Loading and merging {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=False,
    )
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"Merged weights saved to {merged_dir}/")

    # Step 2: Clone llama.cpp if needed
    llama_cpp_dir = Path("llama.cpp")
    if not llama_cpp_dir.exists():
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp"],
            check=True,
        )

    # Step 3: Convert to f16 GGUF
    print("Converting to GGUF (f16)...")
    subprocess.run(
        [
            sys.executable,
            str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ],
        check=True,
    )

    # Step 4: Build llama.cpp quantizer if needed
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("Building llama.cpp...")
        build_dir = llama_cpp_dir / "build"
        cmake_args = ["-DGGML_CUDA=ON"] if _has_cuda() else []
        subprocess.run(
            ["cmake", "-B", str(build_dir), *cmake_args],
            cwd=str(llama_cpp_dir),
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "-j", "8"],
            check=True,
        )

    # Step 5: Quantize
    if args.quantize != "f16":
        print(f"Quantizing to {args.quantize}...")
        subprocess.run(
            [str(quantize_bin), str(f16_gguf), args.output, args.quantize],
            check=True,
        )
        f16_gguf.unlink()  # Clean up intermediate f16 file
    else:
        f16_gguf.rename(args.output)

    print(f"Done: {args.output}")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
