"""
Probe the BirdNET tflite model structure, list all tensors, and locate the
true embedding layer.
The embedding layer should be: the input to the classification layer
(CLASS_DENSE) = the output of global pooling, shape (1,1024).
"""

import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

MODEL_PATH = r"D:\BirdNET-Analyzer\_internal\birdnet_analyzer\checkpoints\V2.4\BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
SEG_SAMPLES = 144000

interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
INPUT_IDX = interp.get_input_details()[0]["index"]
interp.resize_tensor_input(INPUT_IDX, (1, SEG_SAMPLES))
interp.allocate_tensors()
interp.set_tensor(INPUT_IDX, np.zeros((1, SEG_SAMPLES), np.float32))
interp.invoke()

print("=== All tensors with shape containing 1024 ===")
for d in interp.get_tensor_details():
    shp = list(d["shape"])
    if 1024 in shp:
        try:
            interp.get_tensor(d["index"]); ok = "readable"
        except Exception:
            ok = "not readable"
        print(f"index={d['index']:<5} shape={shp} {ok}  name={d['name']}")

print("\n=== Output layer info ===")
for d in interp.get_output_details():
    print(f"output index={d['index']} shape={list(d['shape'])} name={d['name']}")

print("\n=== Tensors with names containing POOL / DENSE / EMBED / FEATURE ===")
for d in interp.get_tensor_details():
    if any(k in d["name"].upper() for k in ["POOL", "DENSE", "EMBED", "FEATURE", "LOGIT", "SOFTMAX"]):
        print(f"index={d['index']:<5} shape={list(d['shape'])}  name={d['name']}")