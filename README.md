# UFLDv1 Evaluation on Raspberry Pi 5

ONNX Runtime CPU evaluation for the UFLD v1 TuSimple baseline on Raspberry Pi 5.

## Contents

- `notebooks/evaluation_ufld_v1_rpi5.ipynb` — accuracy, latency, throughput, utilization, memory, temperature/frequency, operator profiling, tensor-I/O estimate, and optional power-meter energy analysis.
- `Dockerfile`, `compose.yaml`, `requirements-rpi.txt` — isolated ARM64 container setup.
- `run_ufld_v1_onnx.py` — single-image inference example.
- `export_ufld_v1_onnx.py` — export the released PyTorch checkpoint to ONNX on a development computer.
- `README-rpi-docker.md` — Pi setup and launch instructions.

## Quick start

1. Export the model as described in `README-rpi-docker.md` and place it at `models/ufld_v1_tusimple.onnx`.
2. Place `test_label.json` and its referenced TuSimple images under `data/`.
3. Open JupyterLab on port 8888 and run `notebooks/evaluation_ufld_v1_rpi5.ipynb`.

The notebook's estimated tensor-I/O is a graph-based logical estimate, not measured DRAM traffic. Board energy requires readings from an external meter or calibrated sensor.

## Results

The notebook writes `summary.json`, `predictions.jsonl`, and ONNX Runtime profiling output under `results/`. Record the Pi model, cooling, OS, power supply, thread count, and runtime versions with any published measurements.

<img width="975" height="378" alt="image" src="https://github.com/user-attachments/assets/eb8ec8f6-4ea3-4809-9173-8b59a141bdb8" />


## Data and weights

Model files, TuSimple images, and labels are excluded from Git. Follow their respective licenses and terms when downloading and evaluating them.

- **Pruned weights:** pth files of the pruned model can be downloaded at: https://drive.google.com/drive/folders/18yhyHDl8kdn_dsbgMhNkF2TlPokTsTG8?usp=sharing.

- 
### ONNX Files


ONNX files for Quantized and Pruned models are available at: https://drive.google.com/open?id=1wXEp_KevI4iOxYSn1-5Jw0EGCei3s_fu&usp=drive_fs

## License

Add the license approved by the lab or university before public release. Review upstream UFLD v1 licensing and attribution requirements for any copied or modified code.
