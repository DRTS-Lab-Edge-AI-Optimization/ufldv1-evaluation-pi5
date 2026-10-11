"""Run a UFLD v1 TuSimple ONNX model on a Raspberry Pi 5 CPU.

Example:
  python run_ufld_v1_onnx.py ufld_v1_tusimple.onnx frame.jpg --threads 4
"""
import argparse
import json

import numpy as np
import onnxruntime as ort
from PIL import Image

H, W = 288, 800
GRID = 100
ROWS = 56
LANES = 4
TUSIMPLE_WIDTH = 1280


def preprocess(path):
    image = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1))[None, ...].copy()


def decode(logits):
    # logits has shape [grid+1, rows, lanes]; last grid index is background.
    background = np.argmax(logits, axis=0) == GRID
    z = logits[:-1] - np.max(logits[:-1], axis=0, keepdims=True)
    prob = np.exp(z)
    prob /= np.sum(prob, axis=0, keepdims=True)
    positions = np.sum(prob * np.arange(GRID, dtype=np.float32)[:, None, None], axis=0)
    positions[background] = GRID
    lanes = []
    for lane in range(LANES):
        xs = []
        for row in range(ROWS):
            p = positions[row, lane]
            xs.append(-2 if p == GRID else int(round((p + 0.5) * TUSIMPLE_WIDTH / (GRID - 1))))
        lanes.append(xs)
    return lanes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="ONNX file")
    parser.add_argument("image", help="input image")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op CPU threads")
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.model, sess_options=options, providers=["CPUExecutionProvider"])
    logits = session.run(["lane_logits"], {"images": preprocess(args.image)})[0][0]
    result = {
        "lanes": decode(logits),
        "h_samples": list(range(160, 720, 10)),
        "raw_file": args.image,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
