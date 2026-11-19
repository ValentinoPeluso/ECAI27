#!/usr/bin/env python3
"""Classify a transcript with the exported ONNX intent model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="ASR transcript or other text command")
    parser.add_argument("--model-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    model_dir = args.model_dir
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(
        str(model_dir / "intent_classifier.onnx"),
        providers=["CPUExecutionProvider"],
    )

    full_inference_started = time.perf_counter()
    tokenization_started = time.perf_counter()
    encoded = tokenizer.encode(args.text)
    input_ids = np.asarray([encoded.ids], dtype=np.int64)
    attention_mask = np.asarray([encoded.attention_mask], dtype=np.int64)
    tokenization_seconds = time.perf_counter() - tokenization_started

    model_inference_started = time.perf_counter()
    logits = session.run(
        ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
    )[0][0]
    model_inference_seconds = time.perf_counter() - model_inference_started
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    class_id = int(np.argmax(probabilities))
    full_inference_seconds = time.perf_counter() - full_inference_started
    print("Intent prediction")
    print("-----------------")
    print(f"Text:       {args.text}")
    print(f"Intent:     {labels[class_id]}")
    print(f"Confidence: {probabilities[class_id]:.1%}")
    print("\nInference time")
    print(f"  Tokenization:     {tokenization_seconds * 1000:.2f} ms")
    print(f"  Model inference:  {model_inference_seconds * 1000:.2f} ms")
    print(f"  Full inference:   {full_inference_seconds * 1000:.2f} ms")


if __name__ == "__main__":
    main()
