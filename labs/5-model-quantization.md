---
title: "5. Model Quantization"
---

## Introduction

In this lab, you will apply three post-training quantization (PTQ) strategies to the Intent Classification FP32 ONNX model and compare their accuracy, storage size, RAM requirementes, and CPU inference latency.

Review `ECAI_05-Part-II_Optimization_II`, slides 154–195 before starting the exercises.


- Use a Python notebook environment with NumPy, ONNX, ONNX Runtime (CPU build with the notebook's 4-bit quantizer API and kernel support), PyTorch, Transformers, and Tokenizers. No GPU is needed. The tokenizer needs network access on first use unless already cached.
- Open **** — notebook link to be added. Set the notebook’s working directory so `test.jsonl` and `results/intent_classifier.onnx` resolve. Add `import json` to the imports before loading the dataset.

## From the lecture to the notebook

## Post-Training Quantization

Go to (`5.1 - Model Quantization.ipynb`)[https://deepnote.com/workspace/ECAI-27-e30878e0-87e9-4b7c-b044-c52c3f4dd72f/project/Material-3880e510-b64c-4bb5-b488-c2122d5d9e2d/notebook/51-Model-Quantization-9c11feaf3edc483f802171abb822de64?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=3880e510-b64c-4bb5-b488-c2122d5d9e2d] and quantize the model using the different strategies provided by ONNX Runtime. Measure the test accuracy for each quantized variant.

See also: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

## RPI profiling
Reuse `test_accuracy` on the same test set for all four models. Measure total model storage in bytes, including any external tensor files. For latency, reuse one warmed-up CPU session per model, time repeated `session.run` calls on the same pre-tokenized batch-size-one inputs, and report the median in milliseconds. Keep the machine, provider, and thread settings fixed. Exclude session creation and tokenization from the timed region.

Download [profile_onnx_runtime.py](../assets/downloads/profile_onnx_runtime.py) to the Raspberry Pi. Copy the ONNX models and `tokenizer.json` and `labels.json` into `results/` beside the script.

From the directory containing the script, profile all models, e.g.:

```bash
python3 profile_onnx_runtime.py "Start sending measurements to cloud storage." \
  --model-dir results --onnx-path results/intent_classifier.onnx --repeat 30
```

| Variant | Accuracy (%) | Change vs. FP32 (percentage points) | Storage (bytes) | Median latency (ms) | RAM (bytes) |
|---|---:|---:|---:|---:|---:|
| FP32 | | | | | |
| 4-bit weight-only | | | | | |
| Dynamic 8-bit | | | | | |
| Dynamic 8-bit (reduce range) | | | | | |
| Static INT8 | | | | | |

For RAM, record the trace maximum reported as `ORT CPU arena held`, or enter `N/A` if unavailable. This measures memory held by ONNX Runtime's CPU allocator, not total process RAM.
