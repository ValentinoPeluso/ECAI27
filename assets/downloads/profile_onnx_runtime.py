#!/usr/bin/env python3
"""Capture and summarize an ONNX Runtime profiling trace for intent inference."""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def summarize_ort_profile(profile_path: Path) -> tuple[list[dict[str, Any]], int | None]:
    """Read an ORT trace and return per-node latency metrics and peak arena bytes."""
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_durations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    arena_sizes = []

    for event in events:
        if event.get("cat") != "Node" or "dur" not in event:
            continue
        event_args = event.get("args", {})
        node_name = event.get("name", "unknown").removesuffix("_kernel_time")
        op_name = event_args.get("op_name", "unknown")
        provider = event_args.get("provider", "unknown")
        node_durations[(node_name, op_name, provider)].append(float(event["dur"]))

        arena_held = event_args.get("mem_arena_held")
        if arena_held is not None:
            arena_sizes.append(int(arena_held))

    metrics = [
        {
            "node": node_name,
            "operator": op_name,
            "provider": provider,
            "calls": len(durations),
            "median_ms": statistics.median(durations) / 1000,
            "std_ms": statistics.pstdev(durations) / 1000,
        }
        for (node_name, op_name, provider), durations in node_durations.items()
    ]
    return metrics, max(arena_sizes) if arena_sizes else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="ASR transcript or other text command")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/quantization/tiny-lora/RPI"),
        help="Directory containing tokenizer.json and labels.json",
    )
    parser.add_argument("--onnx-path", type=Path, required=True, help="ONNX model to profile")
    parser.add_argument("--repeat", type=int, default=10, help="Number of profiled inferences")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    tokenizer_path = args.model_dir / "tokenizer.json"
    labels_path = args.model_dir / "labels.json"
    if not tokenizer_path.is_file() or not labels_path.is_file():
        parser.error("--model-dir must contain tokenizer.json and labels.json")
    if not args.onnx_path.is_file():
        parser.error(f"ONNX model not found: {args.onnx_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    encoded = tokenizer.encode(args.text)
    inputs = {
        "input_ids": np.asarray([encoded.ids], dtype=np.int64),
        "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
    }

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(Path.cwd() / "intent_inference")
    session = ort.InferenceSession(
        str(args.onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inference_times_ms = []
    outputs = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        outputs = session.run(["logits"], inputs)[0][0]
        inference_times_ms.append((time.perf_counter() - started) * 1000)
    profile_path = Path(session.end_profiling())

    class_id = int(np.argmax(outputs))
    intent = labels[class_id] if class_id < len(labels) else str(class_id)
    node_metrics, peak_arena = summarize_ort_profile(profile_path)

    print("ONNX Runtime profile")
    print("--------------------")
    print(f"Text:       {args.text}")
    print(f"Model:      {args.onnx_path}")
    print(f"Intent:     {intent}")
    print(f"Inferences: {args.repeat}")
    print(f"Trace:      {profile_path}")
    print("\nModel and inference summary")
    print(f"  ONNX file size:       {args.onnx_path.stat().st_size:,} bytes "
          f"({args.onnx_path.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"  Inference median:     {statistics.median(inference_times_ms):.3f} ms")
    print(f"  Inference std (pop.): {statistics.pstdev(inference_times_ms):.3f} ms")
    if peak_arena is not None:
        print(f"  ORT CPU arena held:   {peak_arena:,} bytes "
              f"({peak_arena / (1024 * 1024):.2f} MiB, trace maximum)")
    else:
        print("  ORT CPU arena held:   unavailable in this ORT trace")

    print("\nPer-node metrics (ORT node events; slowest median first)")
    print(f"{'Node':100} {'Operator':24} {'Provider':24} {'Calls':>7} {'Median (ms)':>13} {'Std (ms)':>11}")
    print("-" * 186)
    for metric in sorted(node_metrics, key=lambda row: row["median_ms"], reverse=True):
        print(
            f"{metric['node']:100} {metric['operator']:24.24} "
            f"{metric['provider']:24.24} {metric['calls']:7d} "
            f"{metric['median_ms']:13.4f} {metric['std_ms']:11.4f}"
        )
    if not node_metrics:
        print("No node events found in the ORT trace; inspect the raw JSON trace.")


if __name__ == "__main__":
    main()
