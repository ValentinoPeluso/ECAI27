---
title: "4. Intent Classification"
---

## Introduction

In this lab, you will train a small Transformer model to recognize the intent of a text command. The commands are similar to the ones you used in the previous labs: for example, enabling or disabling cloud data transmission, or changing the sensor sampling interval.

The dataset contains complete sentences rather than isolated keywords. This means that the same intent can be expressed in different ways. In the final exercise, you will connect the classifier to the speech-recognition pipeline developed in the previous lab.

## Exercise 1: Training an Intent Classification Model

The training dataset is a custom dataset created with the help of an LLM. It was generated using curated sentence templates and compatible slot filling, and then split into training and test sets.

The dataset contains five classes:

| Intent | Meaning |
|---|---|
| `enable_cloud` | Enable the transmission of sensor readings to cloud storage. |
| `disable_cloud` | Stop cloud transmission and keep readings on the device. |
| `increase_interval` | Increase the time between consecutive measurements. |
| `decrease_interval` | Decrease the time between consecutive measurements. |
| `unknown` | A command that does not correspond to one of the supported actions. |

The dataset is balanced: it contains 400 training sentences (80 per class) and 200 test sentences (40 per class). All sentences are unique, and there is no exact text overlap between the training and test sets.

### 1.1 Run the training pipeline

See [Deepnote: 4.1 - Lab3 - Training](https://deepnote.com/workspace/ECAI-27-e30878e0-87e9-4b7c-b044-c52c3f4dd72f/project/Material-3880e510-b64c-4bb5-b488-c2122d5d9e2d/notebook/41-Lab3-Training-6da5aa6cecda4e8c834a9d6043f70cc7?utm_source=share-modal&utm_medium=product-shared-content&utm_campaign=notebook&utm_content=3880e510-b64c-4bb5-b488-c2122d5d9e2d).

## Exercise 2: Test the Intent Classification Model

### 2.1 Copy the inference files to the Raspberry Pi

Download and Copy the following files to the Raspberry Pi:

```text
results/
    ├── intent_classifier.onnx
    ├── tokenizer.json
    └── labels.json
```

Then download the script [predict_intent.py](../assets/downloads/predict_intent.py).


### 2.2 Test the classifier from the command line

From the directory containing `predict_intent.py`, run the following commands:

```bash
python predict_intent.py "Start sending measurements to cloud storage."
python predict_intent.py "Can you keep the new readings on the device only?"
python predict_intent.py "Can you let the monitor wait one second more before measuring again?"
python predict_intent.py "Decrease the wait between samples by one second, please."
python predict_intent.py "Can you play some music?"
```

The first four commands represent the four supported device actions. The last one is an example of an unrelated request and should be classified as `unknown`.

To specify the model directory explicitly, add the `--model-dir` option:

```bash
python predict_intent.py "Stop uploading sensor measurements." --model-dir results
```

### 2.3 Understand the inference script

The inference script follows four steps:

1. **Load the model information.** It reads `labels.json`, loads the tokenizer from `tokenizer.json`, and creates an ONNX Runtime inference session using the CPU execution provider.
2. **Tokenize the input text.** The tokenizer converts the command into `input_ids` and an `attention_mask`, which are converted into NumPy arrays.
3. **Run model inference.** ONNX Runtime receives the two input arrays and returns the logits for the five classes.
4. **Select and display the prediction.** The script applies the softmax operation to convert logits into probabilities. It selects the class with the highest probability and prints the predicted intent and its confidence.

The script also reports the time spent in tokenization, model execution, and the complete inference operation. The reported times do not include the initial loading of the tokenizer and ONNX model, which happens before the timer starts.

Try different sentences expressing the same intent. Also try commands that are unrelated to the supported actions. Do the predictions remain consistent when you change the wording?

## Exercise 3: Integrating Intent Classification with Speech Recognition

In the previous lab, you developed a speech-recognition pipeline using Moonshine to convert microphone input into text. Extend that exercise by connecting the ASR output to the intent classifier.

The final pipeline should work as follows:

```text
Microphone
    │
    ▼
Moonshine ASR
    │
    ▼
Recognized text
    │
    ▼
Intent classifier (ONNX Runtime)
    │
    ▼
Device action
```

You have to implement the integration code. Reuse the Moonshine speech-recognition code from the previous lab and the ONNX model and tokenizer generated in Exercise 1.

Your program must:

1. Capture a spoken command using the microphone and transcribe it with Moonshine.
2. Pass the recognized text to the intent classifier.
3. Print the transcript, predicted intent, and confidence.
4. Execute the corresponding action on the Raspberry Pi:
   - `enable_cloud`: enable the transmission of sensor readings to the cloud.
   - `disable_cloud`: disable cloud transmission.
   - `increase_interval`: increase the time between consecutive sensor measurements.
   - `decrease_interval`: decrease the time between consecutive sensor measurements.
   - `unknown`: do not execute any device action.
5. Continue listening for commands so that you can test the system interactively.

Test the complete system with different spoken versions of each command, as well as sentences that should be classified as `unknown`. Check both the ASR transcript and the predicted intent: an incorrect transcript can lead to an incorrect intent even when the classifier itself is working as expected.
