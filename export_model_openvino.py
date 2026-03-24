"""
export_model_openvino.py
Chuyển đổi YOLO .pt model sang OpenVINO IR format.

Cách dùng:
  python export_model_openvino.py
  python export_model_openvino.py --model license-plate-finetune-v1x.pt
  python export_model_openvino.py --model yolo26n.pt --imgsz 640
  python export_model_openvino.py --model license-plate-finetune-v1n.pt --half
"""

import argparse
from ultralytics import YOLO


def export(model_path: str, imgsz: int = 640, half: bool = False):
    print(f"[INFO] Nạp model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Xuất sang OpenVINO | imgsz={imgsz} | half={half}")
    out = model.export(
        format="openvino",
        imgsz=imgsz,
        half=half,
        dynamic=False,
    )
    print(f"[INFO] Hoàn tất → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="license-plate-finetune-v1n.pt",
                        help="Đường dẫn file .pt cần chuyển đổi")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh input (mặc định 640)")
    parser.add_argument("--half",  action="store_true",
                        help="Xuất FP16 (chỉ dùng khi device hỗ trợ)")
    args = parser.parse_args()

    export(args.model, args.imgsz, args.half)
