#!/usr/bin/env python3
"""Benchmark local GGUF LLM inference on Jetson-class hardware.

Primary backend: llama-cpp-python (GGUF), built with CUDA enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


DEFAULT_PROMPTS = [
    "Explain edge AI in one paragraph for a beginner.",
    "Write 5 bullet points on optimizing LLM inference on embedded devices.",
    "Summarize why quantization helps on memory-constrained GPUs.",
]


@dataclass
class PromptResult:
    prompt: str
    output: str
    ttft_s: float
    generation_time_s: float
    generated_tokens: int
    tps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local GGUF LLM inference.")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--model-name", default=None, help="Optional friendly name for report")
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional text file with one prompt per line",
    )
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts to run")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per prompt")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context window")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=35,
        help="Number of layers to offload to GPU (reduce if OOM)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="CPU threads for llama.cpp",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one short warmup generation before measurements",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save machine-readable benchmark summary JSON",
    )
    return parser.parse_args()


def get_process_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_peak_rss_mb_linux() -> float:
    # On Linux ru_maxrss is in KB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_prompts(prompts_file: Optional[str], num_prompts: int) -> List[str]:
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS

    if num_prompts <= len(prompts):
        return prompts[:num_prompts]

    # Repeat prompts if user asks for more than available.
    expanded = []
    while len(expanded) < num_prompts:
        expanded.extend(prompts)
    return expanded[:num_prompts]


def load_model(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    seed: int,
):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it with CUDA build flags first."
        ) from exc

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    start = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        seed=seed,
        verbose=False,
    )
    load_time_s = time.perf_counter() - start
    return llm, load_time_s


def tokenize_count(llm, text: str) -> int:
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False)
    return len(tokens)


def run_single_prompt(
    llm,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> PromptResult:
    request_start = time.perf_counter()
    first_token_at = None
    output_chunks: List[str] = []

    stream = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    for chunk in stream:
        text_piece = chunk.get("choices", [{}])[0].get("text", "")
        if not text_piece:
            continue
        if first_token_at is None:
            first_token_at = time.perf_counter()
        output_chunks.append(text_piece)

    finished_at = time.perf_counter()
    output_text = "".join(output_chunks)

    if first_token_at is None:
        # No generated token case.
        ttft_s = finished_at - request_start
        generation_time_s = 0.0
    else:
        ttft_s = first_token_at - request_start
        generation_time_s = finished_at - first_token_at

    generated_tokens = tokenize_count(llm, output_text) if output_text else 0
    tps = generated_tokens / generation_time_s if generation_time_s > 0 else 0.0

    return PromptResult(
        prompt=prompt,
        output=output_text,
        ttft_s=ttft_s,
        generation_time_s=generation_time_s,
        generated_tokens=generated_tokens,
        tps=tps,
    )


def format_mb(value: Optional[float]) -> str:
    return f"{value:.2f} MB" if value is not None else "N/A (install psutil)"


def main() -> int:
    args = parse_args()

    prompts = load_prompts(args.prompts_file, args.num_prompts)

    mem_before_load = get_process_rss_mb()

    llm, load_time_s = load_model(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        seed=args.seed,
    )

    mem_after_load = get_process_rss_mb()

    if args.warmup:
        _ = run_single_prompt(
            llm=llm,
            prompt="Warmup: say hello.",
            max_tokens=min(16, args.max_tokens),
            temperature=0.0,
            top_p=1.0,
        )

    results: List[PromptResult] = []
    for idx, prompt in enumerate(prompts, start=1):
        result = run_single_prompt(
            llm=llm,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results.append(result)

        print(f"\\n--- Prompt {idx} ---")
        print(f"Prompt: {prompt}")
        print(f"TTFT: {result.ttft_s:.3f} s")
        print(f"Generation time: {result.generation_time_s:.3f} s")
        print(f"Generated tokens: {result.generated_tokens}")
        print(f"TPS: {result.tps:.2f} tokens/s")
        print(f"Output preview: {result.output[:200].replace(chr(10), ' ')}")

    mem_after_run = get_process_rss_mb()
    peak_rss_mb = get_peak_rss_mb_linux()

    avg_ttft = statistics.mean(r.ttft_s for r in results)
    avg_tps = statistics.mean(r.tps for r in results)
    total_generated_tokens = sum(r.generated_tokens for r in results)
    avg_generation_time_s = statistics.mean(r.generation_time_s for r in results)

    summary = {
        "model_name": args.model_name or os.path.basename(args.model),
        "model_path": os.path.abspath(args.model),
        "model_load_time_s": load_time_s,
        "avg_ttft_s": avg_ttft,
        "avg_tps": avg_tps,
        "avg_generation_time_s": avg_generation_time_s,
        "total_generated_tokens": total_generated_tokens,
        "num_prompts": len(results),
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "max_tokens": args.max_tokens,
        "memory": {
            "process_rss_before_load_mb": mem_before_load,
            "process_rss_after_load_mb": mem_after_load,
            "process_rss_after_run_mb": mem_after_run,
            "peak_process_rss_mb": peak_rss_mb,
        },
        "per_prompt": [
            {
                "prompt": r.prompt,
                "ttft_s": r.ttft_s,
                "generation_time_s": r.generation_time_s,
                "generated_tokens": r.generated_tokens,
                "tps": r.tps,
            }
            for r in results
        ],
    }

    print("\\n=== Benchmark Summary ===")
    print(f"Model: {summary['model_name']}")
    print(f"Model load time: {load_time_s:.3f} s")
    print(f"Average TTFT: {avg_ttft:.3f} s")
    print(f"Average TPS: {avg_tps:.2f} tokens/s")
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Process RSS before load: {format_mb(mem_before_load)}")
    print(f"Process RSS after load: {format_mb(mem_after_load)}")
    print(f"Process RSS after run: {format_mb(mem_after_run)}")
    print(f"Peak process RSS: {peak_rss_mb:.2f} MB")
    print(
        "Jetson tip: also monitor shared RAM/GPU in jtop to capture full device-level memory pressure."
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved JSON summary: {args.json_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
