# UFLD v1 ONNX evaluation on Raspberry Pi 5

This bundle runs the evaluator in its own ARM64 Docker container, with model and dataset mounted read-only and results written to `results/`.

## Prepare the Pi

Install Docker Engine and the Compose plugin on Raspberry Pi OS 64-bit. Copy this folder to the Pi, then make these folders beside `compose.yaml`:

```sh
mkdir -p models data results notebooks
cp evaluation_ufld_v1_rpi5.ipynb notebooks/
cp /path/to/ufld_v1_tusimple.onnx models/
```

Put TuSimple `test_label.json` in `data/` and make its `clips/...` images reachable from `data/` (the usual layout is `data/clips/...`). If the image data is in `data/test_set/clips`, set `IMAGE_ROOT` accordingly in the notebook.

## Build and launch

```sh
export JUPYTER_TOKEN='choose-a-long-local-token'
docker compose up -d --build
docker compose logs
```

Open `http://<pi-address>:8888` and enter the token. Open `/app/notebooks/evaluation_ufld_v1_rpi5.ipynb`. Stop the isolated app with `docker compose down`.

`mem_limit` and `cpus` are per-container resource limits; change them in `compose.yaml` if desired. The Compose file uses four CPU cores as a starting point. Benchmark several `ORT_THREADS` values because the best setting depends on cooling, OS load, and model/runtime versions.

## Measurements and interpretation

The notebook reports model file size, accuracy/FP/FN, preprocessing and model latency, p50/p95, throughput, process RSS, cgroup memory, CPU utilization, temperature/frequency when host sysfs is mounted, ONNX Runtime node profiling, and an ONNX graph tensor-I/O estimate. Tensor-I/O is a logical estimate (sum of node input/output tensor sizes); it is not measured DRAM traffic and can double-count tensors reused by multiple operators.

Docker cannot directly measure total Raspberry Pi board energy. For energy per inference, enter a measured average system power from an external USB-C power meter (or another calibrated board power sensor) and the idle power measured at the same point. The notebook computes gross and idle-subtracted energy from the measured inference interval. Leave the fields empty if no power measurement is available.

Accuracy uses the TuSimple lane matching rule: a predicted lane matches a ground-truth lane when more than 85% of valid ground-truth points are within 20 pixels; one-to-one matching supplies Accuracy, FP, and FN. Compare against the notebook baseline only with the same dataset subset, decoding, image preprocessing, and checkpoint.
