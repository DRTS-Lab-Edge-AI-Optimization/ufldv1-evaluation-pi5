"""Export the released UFLD v1 TuSimple ResNet-18 checkpoint to ONNX.

Run on the PC that has the UFLD v1 repository and checkpoint:
  python export_ufld_v1_onnx.py --repo C:/path/to/Ultra-Fast-Lane-Detection \
      --weights C:/path/to/weights/tusimple_18.pth --output ufld_v1_tusimple.onnx
"""
import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="UFLD v1 repository root")
    parser.add_argument("--weights", required=True, help="tusimple_18.pth checkpoint")
    parser.add_argument("--output", default="ufld_v1_tusimple.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    import ufld_v1 as v1  # helper used by the baseline notebook

    cfg = v1.load_cfg_v1(os.path.join(repo, "configs", "tusimple.py"), args.weights)
    model = v1.build_v1(cfg, args.weights).cpu().eval()
    example = torch.zeros(1, 3, v1.INPUT_H, v1.INPUT_W, dtype=torch.float32)

    # UFLD v1 output is [batch, griding+1, row_anchors, lanes].
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["images"],
        output_names=["lane_logits"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch"}, "lane_logits": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Saved {os.path.abspath(args.output)}")
    print(f"Input: float32 NCHW [N, 3, {v1.INPUT_H}, {v1.INPUT_W}]")
    print(f"Output: [N, {cfg.griding_num + 1}, {len(v1.H_SAMPLES)}, {cfg.num_lanes}]")


if __name__ == "__main__":
    main()
