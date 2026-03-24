"""
detect_cars_motobike.py
Phát hiện xe ô tô, xe máy và người bằng YOLO26 + OpenVINO backend.

Cách dùng:
  python detect_cars_motobike.py                        # webcam, OpenVINO (mặc định)
  python detect_cars_motobike.py --source video.mp4     # video file
  python detect_cars_motobike.py --source image.jpg     # ảnh tĩnh
  python detect_cars_motobike.py --model yolo26n.pt     # dùng PyTorch weight
  python detect_cars_motobike.py --api                  # chạy REST API server
  python detect_cars_motobike.py --rtsp rtsp://user:pass@192.168.1.10:554/stream  # camera RTSP
  python detect_cars_motobike.py --rtsp "rtsp://user:pass@192.168.1.10:554/stream" --setup-region  # chon vung nhan dien

LƯU Ý: Khi dùng OpenVINO IR model, device luôn là 'cpu'.
        OpenVINO runtime tự được kích hoạt qua định dạng model, không qua device string.
Model đã export sẵn tại: yolo26n_openvino_model/
"""

import argparse
import os
import time
import threading
import socket
import json
import cv2
import numpy as np
import requests
import ssl
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import supervision as sv
import openvino

import builtins as _builtins
_orig_print = _builtins.print
def print(*args, **kwargs):
    import time as _time
    _orig_print(f"[{_time.strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)
_builtins.print = print


def add_ov_libs_to_path():
    try:
        # Tùy thuộc vào phiên bản OpenVINO, các tệp DLL có thể nằm ở các vị trí khác nhau
        # Ví dụ: trong môi trường ảo pip, nó nằm trong site-packages/openvino/libs/
        ov_lib_path = os.path.join(os.path.dirname(openvino.__file__), "libs")
        if os.path.isdir(ov_lib_path):
            if ov_lib_path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ov_lib_path + os.pathsep + os.environ["PATH"]
                print(f"Đã thêm OpenVINO libs: {ov_lib_path} vào PATH.")
            
            # Giải quyết lỗi DLL load failed cho onnxruntime_providers_openvino trên Windows (Python >= 3.8)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(ov_lib_path)
                print(f"Đã gọi os.add_dll_directory cho OpenVINO libs.")
        else:
            print("Không cần thêm thủ công hoặc không tìm thấy thư mục libs.")
    except Exception as e:
        print(f"Lỗi khi thêm thủ công PATH: {e}")

add_ov_libs_to_path()

# ── Bypass SSL toàn diện (fast-alpr / open-image-models / huggingface_hub) ──────
# Patch 1: urllib.request.urlopen không truyền context
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Patch 2: ssl.create_default_context() – tạo context tường minh rồi tắt verify
_ssl_cdc_orig = ssl.create_default_context
def _ssl_no_verify_context(*args, **kwargs):
    ctx = ssl._create_unverified_context()  # bỏ qua CA hoàn toàn
    return ctx
ssl.create_default_context = _ssl_no_verify_context

# Patch 3: urllib.request.urlopen – ép context=unverified dù thư viện truyền context riêng
import urllib.request as _urllib_req
_urlopen_orig = _urllib_req.urlopen
def _urlopen_no_verify(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
    kwargs["context"] = ssl._create_unverified_context()
    return _urlopen_orig(url, data=data, timeout=timeout, **kwargs)
_urllib_req.urlopen = _urlopen_no_verify

# Patch 4: requests.Session – force verify=False (không dùng setdefault)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    _req_session_orig = requests.Session.request
    def _req_session_no_verify(self, method, url, **kwargs):
        kwargs["verify"] = False          # force, không dùng setdefault
        return _req_session_orig(self, method, url, **kwargs)
    requests.Session.request = _req_session_no_verify
except Exception:
    pass
# ────────────────────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HKV_IP = os.getenv("HKV_IP", "")
HKV_USER = os.getenv("HKV_USER", "admin")
HKV_PASS = os.getenv("HKV_PASS", "")
HKV_SNAPSHOT_URL = os.getenv("HKV_SNAPSHOT_URL", "http://{ip}/cgi-bin/snapshot.cgi?channel=1&subtype=0")
ENABLE_ALPR = os.getenv("ENABLE_ALPR", "true").strip().lower() not in ("0", "false", "no")

ALPR_DETECTOR_MODEL = os.getenv("ALPR_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end")
ALPR_OCR_MODEL = os.getenv("ALPR_OCR_MODEL", "cct-xs-v1-global-model")
PLATE_DETECTOR_ONNX = os.getenv("PLATE_DETECTOR_ONNX", "license-plate-finetune-v1n.onnx")
PLATE_OCR_MODEL = os.getenv("PLATE_OCR_MODEL", "cct-xs-v1-global-model")
PLATE_DETECTOR_CONF = float(os.getenv("PLATE_DETECTOR_CONF", "0.25"))
SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", "2"))  # 1 = xử lý mọi frame (không bỏ qua)
OV_DEVICE   = os.getenv("OV_DEVICE", "CPU")       # OpenVINO device: CPU | GPU | AUTO | NPU


def _ov_device(requested: str) -> str:
    """Chuẩn hóa tên device cho OpenVINO (loại bỏ tiền tố 'intel:' nếu có)."""
    d = requested.upper().replace("INTEL:", "")
    return d if d in ("CPU", "GPU", "AUTO", "NPU") else "CPU"


class OVDetector:
    """
    Chạy ONNX detector (YOLOv8/v11 output format) qua OpenVINO runtime.
    - detect()       → sv.Detections  (dùng cho RTSP stream + ByteTrack)
    - detect_boxes() → list [[x1,y1,x2,y2]]  (dùng cho ALPR crop)
    """

    def __init__(self, onnx_path: str, ov_device: str = "CPU",
                 input_size: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.45):
        core  = openvino.Core()
        model = core.read_model(onnx_path)
        self._compiled  = core.compile_model(model, ov_device)
        self._infer_req = self._compiled.create_infer_request()
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        print(f"[OV] Đã tải {onnx_path} trên {ov_device}")

    def _infer(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        inp = cv2.resize(img_bgr, (self.input_size, self.input_size))
        inp = inp[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]      # NCHW
        self._infer_req.infer({0: inp})
        raw   = self._infer_req.get_output_tensor(0).data   # [1, 4+nc, anchors]
        preds = raw[0].T                                    # [anchors, 4+nc]
        return preds, h0, w0

    def _nms(self, preds, h0, w0, conf_thres, iou_thres):
        nc = preds.shape[1] - 4
        if nc == 1:
            confs   = preds[:, 4]
            cls_ids = np.zeros(len(preds), dtype=int)
        else:
            confs   = preds[:, 4:].max(axis=1)
            cls_ids = preds[:, 4:].argmax(axis=1)

        mask = confs >= conf_thres
        preds, confs, cls_ids = preds[mask], confs[mask], cls_ids[mask]
        if len(preds) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        sx, sy = w0 / self.input_size, h0 / self.input_size
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = np.clip((cx - bw / 2) * sx, 0, w0)
        y1 = np.clip((cy - bh / 2) * sy, 0, h0)
        x2 = np.clip((cx + bw / 2) * sx, 0, w0)
        y2 = np.clip((cy + bh / 2) * sy, 0, h0)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in xyxy],
            confs.tolist(), conf_thres, iou_thres,
        )
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
        idx = indices.flatten()
        return xyxy[idx], confs[idx], cls_ids[idx]

    def detect(self, img_bgr, conf_thres: float = None) -> sv.Detections:
        """Trả về sv.Detections — dùng cho RTSP stream + ByteTrack."""
        ct = conf_thres if conf_thres is not None else self.conf_thres
        preds, h0, w0 = self._infer(img_bgr)
        xyxy, confs, cls_ids = self._nms(preds, h0, w0, ct, self.iou_thres)
        if len(xyxy) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            confidence=confs.astype(np.float32),
            class_id=cls_ids,
        )

    def detect_boxes(self, img_bgr, conf_thres: float = None):
        """Trả về list [[x1,y1,x2,y2]] — dùng cho ALPR crop."""
        ct = conf_thres if conf_thres is not None else self.conf_thres
        preds, h0, w0 = self._infer(img_bgr)
        xyxy, _, _ = self._nms(preds, h0, w0, ct, self.iou_thres)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in xyxy]


alpr_instance = None
def get_alpr():
    global alpr_instance
    if alpr_instance is None:
        try:
            from fast_alpr import ALPR
            alpr_instance = ALPR(
                detector_model=ALPR_DETECTOR_MODEL,
                ocr_model=ALPR_OCR_MODEL
            )
            print(f"[ALPR] Đã tải model nhận diện biển số thành công (Detector: {ALPR_DETECTOR_MODEL}, OCR: {ALPR_OCR_MODEL}).")
        except Exception as e:
            print(f"[ALPR] Không thể tải model: {e}")
            alpr_instance = False
    return alpr_instance

plate_detector_instance = None
def get_plate_detector():
    """Trả về OVDetector cho PLATE_DETECTOR_ONNX, lazy-init một lần."""
    global plate_detector_instance
    if plate_detector_instance is None:
        try:
            plate_detector_instance = OVDetector(
                PLATE_DETECTOR_ONNX,
                ov_device=_ov_device(OV_DEVICE),
                conf_thres=PLATE_DETECTOR_CONF,
            )
        except Exception as e:
            print(f"[ALPR] Không thể tải plate detector ONNX (OpenVINO): {e}")
            plate_detector_instance = False
    return plate_detector_instance

plate_ocr_instance = None
def get_plate_ocr():
    global plate_ocr_instance
    if plate_ocr_instance is None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            plate_ocr_instance = LicensePlateRecognizer(hub_ocr_model=PLATE_OCR_MODEL)
            print(f"[ALPR] Đã tải fast-plate-ocr: {PLATE_OCR_MODEL}")
        except Exception as e:
            print(f"[ALPR] Không thể tải fast-plate-ocr: {e}")
            plate_ocr_instance = False
    return plate_ocr_instance

alpr_executor = ThreadPoolExecutor(max_workers=4)
track_id_to_plate = {}
tracked_entered_ids = set()
last_recognized_plate = {}  # {cam_index: plate_text} – biển số cuối cùng theo từng làn
last_recognized_time  = {}  # {cam_index: time_text}  – thời gian nhận diện theo từng làn


def send_plate_via_socket(plate_text: str, cam_index: int, track_id: int,
                          socket_ip: str, socket_port: int):
    """
    Gửi thông tin biển số vừa nhận diện đến server qua TCP socket dưới dạng JSON.
    Mỗi camera có socket_ip / socket_port riêng cấu hình trong camera.json.
    Chỉ gửi khi plate_text hợp lệ (không phải NO_PLATE / rỗng).
    """
    if not socket_ip or not socket_port:
        return

    payload = {
        "cam":       cam_index + 1,           # Số làn (1-based cho dễ đọc)
        "plate":     plate_text,
        "track_id":  track_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    msg = json.dumps(payload, ensure_ascii=False) + "\n"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((socket_ip, int(socket_port)))
            s.sendall(msg.encode("utf-8"))
        print(f"[SOCKET] Cam {cam_index+1} → {socket_ip}:{socket_port} | {msg.strip()}")
    except Exception as e:
        print(f"[SOCKET] Lỗi gửi Cam {cam_index+1} → {socket_ip}:{socket_port} | {e}")


def send_vehicle_enter_via_socket(cam_index: int, track_id: int, class_name: str,
                                   conf: float, bbox: tuple,
                                   socket_ip: str, socket_port: int):
    """
    Gửi sự kiện xe mới vào vùng đến server qua TCP socket dưới dạng JSON.
    Được gọi ngay khi phát hiện xe đi vào region polygon.
    """
    if not socket_ip or not socket_port:
        return

    x1, y1, x2, y2 = bbox
    payload = {
        "event":      "vehicle_enter",
        "cam":        cam_index + 1,
        "track_id":   track_id,
        "class":      class_name,
        "confidence": round(conf, 4),
        "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    msg = json.dumps(payload, ensure_ascii=False) + "\n"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect((socket_ip, int(socket_port)))
            s.sendall(msg.encode("utf-8"))
        print(f"[SOCKET] vehicle_enter Cam {cam_index+1} → {socket_ip}:{socket_port} | {msg.strip()}")
    except Exception as e:
        print(f"[SOCKET] Lỗi gửi vehicle_enter Cam {cam_index+1} → {socket_ip}:{socket_port} | {e}")


def process_alpr_task(track_id, cam_index, ip, user, password, snapshot_url_template,
                      save_plates=False, socket_ip="", socket_port=0):
    global last_recognized_plate, last_recognized_time
    # Mỗi cam_index có entry riêng → biển số chỉ hiện đúng màn hình của làn đó
    cam_track_key = (cam_index, track_id)
    if not ip:
        print("[ALPR] Chưa cấu hình IP (HKV_IP hoặc từ camera.json)")
        return
    url = snapshot_url_template.format(ip=ip)
    try:
        resp = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=3)
        if resp.status_code == 401:
            resp = requests.get(url, auth=HTTPBasicAuth(user, password), timeout=3)

        if resp.status_code == 200:
            img_arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is not None:
                if save_plates:
                    os.makedirs("images", exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000) % 1000).zfill(3)
                    filename = f"images/plate_{timestamp}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"[ALPR] Đã chụp và lưu ảnh: {filename}")

                plate_text = None

                # --- Bước 1: ONNX (OpenVINO) phát hiện biển số + fast-plate-ocr ---
                detector = get_plate_detector()
                ocr = get_plate_ocr()
                if detector and ocr:
                    boxes = detector.detect_boxes(img)
                    plate_crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
                    if plate_crops:
                        texts = ocr.run(plate_crops)
                        for text in texts:
                            if text and text.strip():
                                plate_text = text.strip().rstrip("_")
                                break
                        if plate_text:
                            print(f"[ALPR] YOLOv11+OCR: {plate_text} (Cam {cam_index+1} Xe ID {track_id})")
                        else:
                            print(f"[ALPR] YOLOv11: phát hiện biển số nhưng OCR không đọc được chữ (Cam {cam_index+1})")
                    else:
                        print(f"[ALPR] YOLOv11: không phát hiện biển số → thử fast_alpr (Cam {cam_index+1})")

                # --- Bước 2: Fallback fast_alpr nếu YOLOv11 không phát hiện được biển số ---
                if plate_text is None:
                    alpr = get_alpr()
                    if alpr:
                        alpr_results = alpr.predict(img)
                        if alpr_results and len(alpr_results) > 0:
                            for r in alpr_results:
                                if r.ocr and r.ocr.text:
                                    plate_text = r.ocr.text
                                    print(f"[ALPR] fast_alpr fallback: {plate_text} (Cam {cam_index+1} Xe ID {track_id})")
                                    break

                # --- Kết quả ---
                ts = time.strftime("%H:%M:%S %d/%m/%Y")
                if plate_text:
                    track_id_to_plate[cam_track_key] = plate_text
                    last_recognized_plate[cam_index] = plate_text
                    last_recognized_time[cam_index]  = ts
                    print(f"[ALPR] Biển số: {plate_text} (Cam {cam_index+1} Xe ID {track_id}) lúc {ts}")
                    send_plate_via_socket(plate_text, cam_index, track_id, socket_ip, socket_port)
                else:
                    last_recognized_plate[cam_index] = "NO_PLATE"
                    last_recognized_time[cam_index]  = ts
                    print(f"[ALPR] Không nhận diện được biển số cho Cam {cam_index+1} Xe ID {track_id}")
            else:
                print(f"[ALPR] Lỗi giải mã ảnh chụp từ camera.")
        else:
            print(f"[ALPR] Lỗi gọi API chụp ảnh: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[ALPR] Lỗi xử lý ALPR: {e}")


def process_alpr_from_crop(track_id: int, cam_index: int, plate_img,
                            socket_ip: str = "", socket_port: int = 0,
                            save_plates: bool = False):
    """
    OCR trực tiếp trên ảnh crop biển số từ frame video.
    Không cần gọi API snapshot Hikvision — dùng khi model plate detector
    chạy trực tiếp trên stream (license-plate-finetune-v1n).
    """
    global last_recognized_plate, last_recognized_time
    cam_track_key = (cam_index, track_id)

    try:
        if plate_img is None or plate_img.size == 0:
            print(f"[ALPR] Crop biển số rỗng — bỏ qua (Cam {cam_index+1} ID {track_id})")
            return

        # Upscale nếu crop quá nhỏ để OCR chính xác hơn
        h, w = plate_img.shape[:2]
        if h < 32 or w < 64:
            scale = max(32 / max(h, 1), 64 / max(w, 1))
            plate_img = cv2.resize(plate_img, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)

        if save_plates:
            os.makedirs("images", exist_ok=True)
            ts_file = time.strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000) % 1000).zfill(3)
            filename = f"images/plate_{ts_file}_T{track_id}.jpg"
            cv2.imwrite(filename, plate_img)
            print(f"[ALPR] Đã lưu ảnh biển số: {filename}")

        plate_text = None
        ts = time.strftime("%H:%M:%S %d/%m/%Y")

        # --- Bước 1: fast-plate-ocr trực tiếp trên crop ---
        ocr = get_plate_ocr()
        if ocr:
            texts = ocr.run([plate_img])
            for text in texts:
                if text and text.strip():
                    plate_text = text.strip().rstrip("_")
                    break
            if plate_text:
                print(f"[ALPR] OCR crop: {plate_text} (Cam {cam_index+1} ID {track_id})")
            else:
                print(f"[ALPR] OCR crop không đọc được chữ (Cam {cam_index+1} ID {track_id})")

        # --- Bước 2: fallback fast_alpr ---
        if plate_text is None:
            alpr = get_alpr()
            if alpr:
                alpr_results = alpr.predict(plate_img)
                if alpr_results:
                    for r in alpr_results:
                        if r.ocr and r.ocr.text:
                            plate_text = r.ocr.text
                            print(f"[ALPR] fast_alpr fallback: {plate_text} (Cam {cam_index+1} ID {track_id})")
                            break

        # --- Kết quả ---
        if plate_text:
            track_id_to_plate[cam_track_key] = plate_text
            last_recognized_plate[cam_index] = plate_text
            last_recognized_time[cam_index]  = ts
            print(f"[ALPR] Biển số: {plate_text} (Cam {cam_index+1} ID {track_id}) lúc {ts}")
            send_plate_via_socket(plate_text, cam_index, track_id, socket_ip, socket_port)
        else:
            last_recognized_plate[cam_index] = "NO_PLATE"
            last_recognized_time[cam_index]  = ts
            print(f"[ALPR] Không nhận diện được biển số — Cam {cam_index+1} ID {track_id}")

    except Exception as e:
        print(f"[ALPR] Lỗi xử lý ALPR từ crop: {e}")


def load_region_points():
    points = []
    for key in ["REGION_A", "REGION_B", "REGION_C", "REGION_D"]:
        val = os.getenv(key)
        if val:
            try:
                x, y = map(int, val.split(","))
                points.append([x, y])
            except Exception:
                pass
    if len(points) == 4:
        return np.array(points, np.int32).reshape((-1, 1, 2))
    return None

def save_region_points_to_env(points):
    from dotenv import set_key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    for i, key in enumerate(["REGION_A", "REGION_B", "REGION_C", "REGION_D"]):
        val = f"{points[i][0]},{points[i][1]}"
        set_key(env_path, key, val)
    # Cập nhật tạm thời cho biến global nếu vẫn dùng
    global REGION_POLYGON
    REGION_POLYGON = np.array(points, np.int32).reshape((-1, 1, 2))
    print(f"[INFO] Đã lưu 4 điểm vào {env_path}")
    print("[INFO] LƯU Ý: Chức năng cấu hình vùng chọn Interactive hiện chỉ ghi vào .env mặc định.")


REGION_POLYGON = load_region_points()

# ----------- Khai báo hỗ trợ setup region click chuột --------------
mouse_points = []
def select_points_callback(event, x, y, flags, param):
    global mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < 4:
            mouse_points.append([x, y])
            print(f"[{len(mouse_points)}/4] Chọn điểm: ({x}, {y})")

def setup_region_interactively(frame, window_name="Setup Region"):
    global mouse_points
    mouse_points = []
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, select_points_callback)
    
    print("=====================================================")
    print("Hãy click chuột trái chọn lần lượt 4 đỉnh A, B, C, D")
    print("của vùng Polygon. Nhấn 'c' để xóa làm lại, 'Enter' để lưu.")
    print("=====================================================")
    
    while True:
        temp_frame = frame.copy()
        
        # Vẽ các điểm đã chọn và đường nối
        for i, pt in enumerate(mouse_points):
            cv2.circle(temp_frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, chr(ord('A') + i), (pt[0]+10, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if i > 0:
                cv2.line(temp_frame, tuple(mouse_points[i-1]), tuple(pt), (255, 0, 0), 2)
        if len(mouse_points) == 4:
            cv2.line(temp_frame, tuple(mouse_points[3]), tuple(mouse_points[0]), (255, 0, 0), 2)
            cv2.putText(temp_frame, "Da du 4 diem. Nhan Enter de luu hoac 'c' de xoa", 
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('c'):
            mouse_points.clear()
            print("Đã xóa làm lại.")
        elif key == 13 or key == ord('\r'): # Enter key
            if len(mouse_points) == 4:
                save_region_points_to_env(mouse_points)
                cv2.destroyWindow(window_name)
                break
            else:
                print("Lỗi: Phải chọn đủ 4 điểm mới có thể lưu!")
                
    return True
# ---------------------------------------------------------------------------

# Class IDs của model license-plate-finetune
VEHICLE_CLASSES = {
    0: "license_plate",
}

COLOR_MAP = {
    0: (0, 215, 255),   # vàng – biển số
}

DEFAULT_MODEL = "license-plate-finetune-v1n.onnx"   # ONNX plate detector (OpenVINO)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> YOLO:
    # Chỉ định task='detect' để tránh warning khi load OpenVINO IR model
    model = YOLO(model_path, task="detect")
    return model


def get_device(model_path: str, requested_device: str) -> str:
    """
    Khi model là OpenVINO IR (thư mục):
      - 'cpu'  → OpenVINO CPU plugin
      - 'GPU'  → OpenVINO GPU plugin (Intel iGPU/dGPU) — giải phóng CPU
      - 'AUTO' → OpenVINO tự chọn device nhanh nhất
    Khi model là .pt PyTorch: dùng requested_device nguyên.
    """
    if os.path.isdir(model_path):
        d = requested_device.upper()
        if d in ("GPU", "AUTO", "NPU", "CPU"):
            return f"intel:{d}"  # Ultralytics 8.4+ OpenVINO syntax
        return "intel:AUTO"  # Mặc định dùng intel:AUTO để an toàn
    return requested_device


def predict_image(model: YOLO, image_path: str, conf: float, device: str) -> list:
    """
    Detect objects trong một ảnh, trả về list detection dạng dict.
    Dùng cho cả REST API và hiển thị CLI.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    results = model.predict(
        source=frame,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=conf,
        device=device,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id":   cls_id,
                "class_name": VEHICLE_CLASSES[cls_id],
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "width":  x2 - x1,
                    "height": y2 - y1,
                },
            })
    return detections


def draw_boxes(frame, sv_detections: sv.Detections, conf_threshold: float,
               save_plates: bool = False, cam_info: dict = None,
               cam_index: int = 0, min_box_height: int = 0):
    """
    Vẽ bounding box + nhãn + bộ đếm lên frame.
    sv_detections : sv.Detections đã được Supervision ByteTrack cập nhật tracker_id.
    """
    count = {name: 0 for name in VEHICLE_CLASSES.values()}

    # min_box_height ưu tiên từ camera config, fallback xuống tham số truyền vào
    _min_h = cam_info.get("min_box_height", min_box_height) if cam_info else min_box_height

    # Lấy vùng Polygon ưu tiên từ camera config, fallback xuống .env
    poly_pts = None
    if cam_info and "region_points" in cam_info and len(cam_info["region_points"]) == 4:
        poly_pts = np.array(cam_info["region_points"], np.int32).reshape((-1, 1, 2))
    else:
        poly_pts = REGION_POLYGON

    if poly_pts is not None:
        cv2.polylines(frame, [poly_pts], isClosed=True, color=(255, 0, 255), thickness=2)

    for j in range(len(sv_detections)):
        cls_id   = int(sv_detections.class_id[j])
        conf     = float(sv_detections.confidence[j])
        track_id = int(sv_detections.tracker_id[j]) if sv_detections.tracker_id is not None else None

        if cls_id not in VEHICLE_CLASSES or conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, sv_detections.xyxy[j])

        # Lọc xe xa: bỏ qua box có chiều cao nhỏ hơn ngưỡng tối thiểu
        if _min_h > 0 and (y2 - y1) < _min_h:
            continue

        # Kiểm tra polygon chỉ để trigger ALPR — không dùng để ẩn bounding box
        if poly_pts is not None:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Kiểm tra xem bounding box có giao cắt/chạm vào vùng polygon không
            pts_to_check = [
                (float(cx), float(y2)),  # Giữa cạnh dưới
                (float(cx), float(cy)),  # Tâm điểm
                (float(x1), float(y2)),  # Góc dưới trái
                (float(x2), float(y2)),  # Góc dưới phải
                (float(x1), float(y1)),  # Góc trên trái
                (float(x2), float(y1)),  # Góc trên phải
                (float(x1), float(cy)),  # Giữa cạnh trái
                (float(x2), float(cy)),  # Giữa cạnh phải
            ]

            is_inside = False
            for pt in pts_to_check:
                if cv2.pointPolygonTest(poly_pts, pt, False) >= 0:
                    is_inside = True
                    break

            # Cẩn thận thêm: Nếu polygon nằm lọt thỏm giữa bounding box
            if not is_inside:
                for pt in poly_pts:
                    px, py = pt[0]
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        is_inside = True
                        break

            # Chỉ trigger ALPR khi biển số trong vùng và đã có tracker_id
            if is_inside and track_id is not None:
                cam_track_key = (cam_index, track_id)
                if cam_track_key not in tracked_entered_ids:
                    tracked_entered_ids.add(cam_track_key)
                    print(f"[REGION] Biển số mới vào vùng — Cam {cam_index+1} | ID:{track_id} | Conf:{conf:.2f} | Box:({x1},{y1})-({x2},{y2})")
                    sock_ip   = cam_info.get("socket_ip",  "")   if cam_info else ""
                    sock_port = cam_info.get("socket_port",  0)  if cam_info else 0
                    send_vehicle_enter_via_socket(
                        cam_index, track_id, VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                        conf, (x1, y1, x2, y2), sock_ip, sock_port
                    )
                    if ENABLE_ALPR:
                        # Crop biển số trực tiếp từ frame, không cần snapshot Hikvision
                        plate_crop = frame[y1:y2, x1:x2].copy()
                        alpr_executor.submit(process_alpr_from_crop, track_id, cam_index,
                                             plate_crop, sock_ip, sock_port, save_plates)
                    else:
                        print(f"[ALPR] Bỏ qua nhận dạng biển số (ENABLE_ALPR=false) — Cam {cam_index+1} ID:{track_id}")

        label = VEHICLE_CLASSES[cls_id]
        color = COLOR_MAP[cls_id]
        count[label] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        txt = f"{label} {conf:.2f}"
        if track_id is not None:
            txt = f"ID:{track_id} {txt}"
            cam_track_key = (cam_index, track_id)
            if cam_track_key in track_id_to_plate:
                txt += f" [{track_id_to_plate[cam_track_key]}]"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, txt, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Bảng đếm góc trên-trái – lấy biển số của đúng làn này (cam_index)
    cam_plate = last_recognized_plate.get(cam_index, "")
    cam_time  = last_recognized_time.get(cam_index, "")

    overlay = frame.copy()

    # Tính toán chiều cao vùng đen dựa trên việc có biển số hay không
    bg_height = 25 * len(count) + 15
    if cam_plate:
        bg_height += 60  # Thêm không gian cho biển số và timestamp

    cv2.rectangle(overlay, (0, 0), (220, bg_height), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = 28
    for name, cnt in count.items():
        color = next(c for k, c in COLOR_MAP.items() if VEHICLE_CLASSES[k] == name)
        cv2.putText(frame, f"  {name}: {cnt}", (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 25

    if cam_plate:
        y += 5
        cv2.putText(frame, f"  BS: {cam_plate}", (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(frame, f"  {cam_time}", (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def run(source, model_path: str, conf: float, device: str):
    """Chạy detect theo thời gian thực (webcam / video / ảnh tĩnh)."""
    print(f"[INFO] Nạp model: {model_path}")
    model = load_model(model_path)
    device = get_device(model_path, device)
    backend = "OpenVINO" if os.path.isdir(model_path) else f"PyTorch ({device})"
    print(f"[INFO] Backend: {backend} | Conf ≥ {conf}")

    src = 0 if source in ("0", 0) else source
    is_image = isinstance(src, str) and src.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    if is_image:
        detections = predict_image(model, src, conf, device)
        frame = cv2.imread(src)
        results = model.predict(
            source=frame, classes=list(VEHICLE_CLASSES.keys()),
            conf=conf, device=device, verbose=False,
        )
        frame = draw_boxes(frame, results, conf)
        cv2.imshow("YOLO26 - OpenVINO", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Video / Webcam ---
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được nguồn: {source}")
        return

    print("[INFO] Nhấn 'q' để thoát | 's' để chụp màn hình")
    shot_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=conf,
            device=device,
            verbose=False,
        )

        frame = draw_boxes(frame, results, conf)

        h, w = frame.shape[:2]
        cv2.putText(frame, "YOLO26 + OpenVINO", (w - 240, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        cv2.imshow("YOLO26 - Cars & Motorbikes | OpenVINO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"screenshot_{shot_idx:04d}.jpg"
            cv2.imwrite(name, frame)
            print(f"[INFO] Đã lưu: {name}")
            shot_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# RTSP Realtime Detection
# ---------------------------------------------------------------------------

class RTSPReader:
    """
    Đọc frame từ RTSP stream trong một thread riêng.
    Giải quyết vấn đề buffer lag: chỉ giữ frame mới nhất,
    bỏ qua các frame cũ chưa kịp xử lý.
    """
    def __init__(self, url: str, reconnect_delay: float = 3.0, read_fps: float = 60.0):
        self.url             = url
        self.reconnect_delay = reconnect_delay
        self._frame          = None
        self._lock           = threading.Lock()
        self._running        = False
        self._thread         = None
        self.connected       = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def read(self):
        """Trả về (True, frame) nếu có, ngược lại (False, None)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def _read_loop(self):
        # ----------------------------------------------------------------
        # TCP transport + thử hardware decode (Intel DXVA2/QSV trên Windows)
        # Software decode H264 1080p ~25fps có thể chiếm 40-80% 1 core CPU.
        # Hardware decode chuyển việc đó sang GPU/iGPU → CPU gần bằng 0.
        # ----------------------------------------------------------------
        hw_options = [
            # Ưu tiên 1: Intel QuickSync (iGPU Intel Arc/Iris)
            "rtsp_transport;tcp|video_codec;h264_qsv|hwaccel;qsv",
            # Ưu tiên 2: DirectX Video Acceleration (Windows generic)
            "rtsp_transport;tcp|hwaccel;dxva2",
            # Fallback: software decode thuần CPU (đã dùng trước đây)
            "rtsp_transport;tcp",
        ]

        for hw_opt in hw_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = hw_opt
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                label = hw_opt.split("|")[0].replace("rtsp_transport;tcp", "TCP")
                hwaccel = hw_opt.split("hwaccel;")[-1] if "hwaccel" in hw_opt else "software"
                print(f"[RTSP] Kết nối OK | decode={hwaccel}")
                break
            cap.release()
        else:
            # Tất cả đều thất bại
            print(f"[RTSP] Không kết nối được: {self.url} → thử lại sau {self.reconnect_delay}s")
            print(f"[RTSP] Gợi ý: Đảm bảo URL trong ngoặc kép khi chạy lệnh")
            self.connected = False
            time.sleep(self.reconnect_delay)
            return

        # Giảm buffer còn 1 frame → không tích lũy frame cũ
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = True

        while self._running:
            # Liên tục đọc để tránh nghẽn luồng RTSP gây nhòe hoặc lỗi ffmpeg "error while decoding"
            ret, frame = cap.read()
            if not ret:
                print("[RTSP] Mất kết nối, đang tự kết nối lại...")
                self.connected = False
                cap.release()
                time.sleep(self.reconnect_delay)
                # Gọi đệ quy để thử lại từ đầu (kể cả thử lại hw decode)
                self._read_loop()
                return

            with self._lock:
                self._frame = frame

        cap.release()


def run_rtsp(rtsp_urls: list, model_path: str, conf: float, device: str,
             reconnect_delay: float = 3.0, max_fps: float = 15.0,
             skip_frames: int = 1, infer_size: int = 640, save_plates: bool = False,
             cam_configs: list = None, min_box_height: int = 0,
             no_display: bool = False):
    """
    Nhận dạng realtime từ nhiều camera RTSP (tối đa 4 stream song song).
    skip_frames    : chỉ inference 1 trong mỗi skip_frames frame (giảm CPU)
    infer_size     : resize frame xuống trước inference (mặc định 640)
    min_box_height : bỏ qua xe có bounding box nhỏ hơn N pixel (0 = không lọc)
    no_display     : True = chạy headless, không mở cửa sổ OpenCV
    """
    import numpy as np
    if len(rtsp_urls) > 4:
        print("[WARNING] Chỉ hỗ trợ tối đa 4 camera đồng thời. Lấy 4 URL đầu tiên.")
        rtsp_urls = rtsp_urls[:4]

    num_cams = len(rtsp_urls)
    ov_dev   = _ov_device(device)
    print(f"[INFO] Nạp model: {model_path} (OpenVINO/{ov_dev}) cho {num_cams} camera")
    detectors = [OVDetector(model_path, ov_device=ov_dev,
                            input_size=infer_size, conf_thres=conf)
                 for _ in range(num_cams)]
    # Mỗi camera có một ByteTrack tracker riêng → track_id độc lập giữa các làn
    trackers = [sv.ByteTrack() for _ in range(num_cams)]
    backend  = f"OpenVINO ({ov_dev})"
    print(f"[INFO] Backend: {backend} | Conf ≥ {conf}")
    print(f"[INFO] Tracker: Supervision ByteTrack (mỗi camera 1 tracker độc lập)")
    print(f"[INFO] Max FPS: {max_fps} | Skip: 1/{skip_frames} frames | Size: {infer_size}")
    for i, u in enumerate(rtsp_urls):
        print(f"[INFO] Cam {i+1} RTSP URL: {u}")
    if no_display:
        print("[INFO] Chế độ HEADLESS — không hiển thị cửa sổ video. Nhấn Ctrl+C để dừng.")
    else:
        print("[INFO] Phím: 'q' thoát | 's' chụp màn hình")

    readers = [RTSPReader(url, reconnect_delay=reconnect_delay, read_fps=max_fps).start() for url in rtsp_urls]

    frame_interval  = 1.0 / max(max_fps, 1)
    fps_timer       = time.perf_counter()
    fps_count       = 0
    fps_display     = 0.0
    last_fps_print  = time.perf_counter()
    shot_idx        = 0
    frame_counter   = 0
    last_results_list = [sv.Detections.empty() for _ in range(num_cams)]
    grid_w, grid_h = 960, 540

    # --no-display bỏ qua --setup-region (không có cửa sổ để click)
    setup_done = False if (args.setup_region and not no_display) else True

    try:
      while True:
        frames = []
        rets = []
        for reader in readers:
            ret, frame = reader.read()
            rets.append(ret)
            if ret:
                frames.append(frame)
            else:
                blank = np.zeros((grid_h, grid_w, 3), dtype='uint8')
                status = "Dang ket noi..." if not reader.connected else "Cho frame..."
                cv2.putText(blank, status, (grid_w//2 - 100, grid_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                frames.append(blank)

        if not any(rets):
            if no_display:
                time.sleep(0.2)
            else:
                grid_frames = [f if f.shape[:2] == (grid_h, grid_w) else cv2.resize(f, (grid_w, grid_h)) for f in frames]
                while len(grid_frames) < 4:
                    grid_frames.append(np.zeros((grid_h, grid_w, 3), dtype='uint8'))

                if num_cams == 1:
                    final_display = frames[0]
                else:
                    top_row    = np.hstack((grid_frames[0], grid_frames[1]))
                    bottom_row = np.hstack((grid_frames[2], grid_frames[3]))
                    final_display = np.vstack((top_row, bottom_row))

                cv2.namedWindow("YOLO26 Multi-RTSP | OpenVINO", cv2.WINDOW_NORMAL)
                cv2.imshow("YOLO26 Multi-RTSP | OpenVINO", final_display)
                if cv2.waitKey(200) & 0xFF == ord("q"):
                    break
            continue

        if not setup_done and sum(rets) >= 1:
            # Lấy frame cam đầu tiên để setup
            setup_frame = None
            for i, r in enumerate(rets):
                if r:
                    setup_frame = frames[i]
                    break
            
            if num_cams > 1:
               # Nếu nhiều cam thì setup trên frame tổng
               grid_frames = [f if f.shape[:2] == (grid_h, grid_w) else cv2.resize(f, (grid_w, grid_h)) for f in frames]
               while len(grid_frames) < 4:
                   grid_frames.append(np.zeros((grid_h, grid_w, 3), dtype='uint8'))
               top_row = np.hstack((grid_frames[0], grid_frames[1]))
               bottom_row = np.hstack((grid_frames[2], grid_frames[3]))
               setup_frame = np.vstack((top_row, bottom_row))

            setup_region_interactively(setup_frame, window_name="Setup Region (Enter to save)")
            setup_done = True
            
            # Flush existing keys
            cv2.waitKey(1)
            continue

        frame_counter += 1
        t_frame_start  = time.perf_counter()

        # ---------- Inference chỉ mỗi skip_frames frame ----------
        if frame_counter % skip_frames == 0:
            for i in range(num_cams):
                if rets[i]:
                    # 1. OVDetector detect → sv.Detections
                    sv_det = detectors[i].detect(frames[i], conf_thres=conf)
                    # 2. Supervision ByteTrack cập nhật tracker_id
                    sv_det = trackers[i].update_with_detections(sv_det)
                    last_results_list[i] = sv_det

        # ---------- Draw boxes & Overlay ----------
        # draw_boxes luôn được gọi để ALPR polygon trigger hoạt động đúng.
        # Khi headless, bỏ qua toàn bộ rendering lên màn hình.
        displays = []
        for i in range(num_cams):
            if rets[i]:
                cfg = cam_configs[i] if cam_configs and i < len(cam_configs) else None
                disp = draw_boxes(frames[i].copy(), last_results_list[i], conf, save_plates, cfg,
                                  cam_index=i, min_box_height=min_box_height)
            else:
                disp = frames[i].copy()

            if not no_display:
                h, w = disp.shape[:2]
                cv2.rectangle(disp, (0, 0), (min(w, 350), 38), (20, 20, 20), -1)
                cv2.putText(disp, f"Cam {i+1} | YOLO26", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                conn_txt   = "CONNECTED" if readers[i].connected else "RECONNECTING..."
                conn_color = (0, 200, 80) if readers[i].connected else (0, 80, 255)
                cv2.putText(disp, conn_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)
            displays.append(disp)

        # ---------- FPS counter (dùng chung cả 2 chế độ) ----------
        fps_count += 1
        elapsed_total = time.perf_counter() - fps_timer
        if elapsed_total >= 1.0:
            fps_display = fps_count / elapsed_total
            fps_count   = 0
            fps_timer   = time.perf_counter()
            if no_display and time.perf_counter() - last_fps_print >= 60.0:
                print(f"[INFO] FPS: {fps_display:.1f}", flush=True)
                last_fps_print = time.perf_counter()

        # ---------- Throttle ----------
        elapsed_frame = time.perf_counter() - t_frame_start
        if elapsed_frame < frame_interval:
            time.sleep(frame_interval - elapsed_frame)

        if no_display:
            # Headless: không cần imshow — Ctrl+C sẽ dừng (xử lý ở ngoài vòng lặp)
            continue

        # ---------- Grid Display (chỉ khi có cửa sổ) ----------
        if num_cams == 1:
            final_display = displays[0]
        else:
            grid_frames = [cv2.resize(d, (grid_w, grid_h)) for d in displays]
            while len(grid_frames) < 4:
                grid_frames.append(np.zeros((grid_h, grid_w, 3), dtype='uint8'))
            top_row    = np.hstack((grid_frames[0], grid_frames[1]))
            bottom_row = np.hstack((grid_frames[2], grid_frames[3]))
            final_display = np.vstack((top_row, bottom_row))

        fh, fw = final_display.shape[:2]
        cv2.putText(final_display, f"Total FPS: {fps_display:.1f}",
                    (fw - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

        cv2.namedWindow("YOLO26 Multi-RTSP | OpenVINO", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO26 Multi-RTSP | OpenVINO", final_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = f"rtsp_shot_{shot_idx:04d}.jpg"
            cv2.imwrite(name, final_display)
            print(f"[INFO] Đã lưu: {name}")
            shot_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Nhận Ctrl+C — đang dừng...")
    finally:
        for reader in readers:
            reader.stop()
        if not no_display:
            cv2.destroyAllWindows()
        print("[INFO] Đã dừng RTSP detection.")


# ---------------------------------------------------------------------------
# REST API  (FastAPI + uvicorn)
# ---------------------------------------------------------------------------

def run_api(model_path: str, conf: float, device: str, host: str, port: int):
    """Khởi động FastAPI server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("[ERROR] Chưa cài fastapi/uvicorn. Chạy: pip install fastapi uvicorn[standard]")
        return

    device = get_device(model_path, device)
    backend = "OpenVINO" if os.path.isdir(model_path) else f"PyTorch ({device})"

    print(f"[INFO] Nạp model: {model_path} | Backend: {backend}")
    model = load_model(model_path)
    print(f"[INFO] API server: http://{host}:{port}")

    app = FastAPI(
        title="YOLO26 Detection API",
        description="Phát hiện xe và người bằng YOLO26 + OpenVINO",
        version="1.0.0",
    )

    # ---- Request / Response schema ----
    # Pydantic class body không thể dùng biến local của function cha trực tiếp
    _default_conf = conf

    class DetectRequest(BaseModel):
        image_path: str               # Đường dẫn tuyệt đối hoặc tương đối tới ảnh
        conf: float = _default_conf   # Ghi đè confidence per-request (tùy chọn)

    class BBox(BaseModel):
        x1: int
        y1: int
        x2: int
        y2: int
        width: int
        height: int

    class Detection(BaseModel):
        class_id:   int
        class_name: str
        confidence: float
        bbox: BBox

    class DetectResponse(BaseModel):
        image_path:          str
        total:               int
        counts:              dict
        processing_time_ms:  float
        detections:          list[Detection]

    # ---- Endpoints ----

    @app.get("/")
    def root():
        return {
            "service": "YOLO26 Detection API",
            "backend": backend,
            "model":   model_path,
            "classes": VEHICLE_CLASSES,
            "endpoints": {
                "POST /detect": "Detect objects trong ảnh",
                "GET  /health": "Kiểm tra server",
            },
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path, "backend": backend}

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        if not os.path.isfile(req.image_path):
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy file: {req.image_path}",
            )

        try:
            t0 = time.perf_counter()
            detections = predict_image(model, req.image_path, req.conf, device)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Tổng hợp số lượng theo class
        counts: dict = {}
        for d in detections:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

        return DetectResponse(
            image_path=req.image_path,
            total=len(detections),
            counts=counts,
            processing_time_ms=elapsed_ms,
            detections=detections,
        )

    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO26 - Detect Cars, Motorbikes & Persons với OpenVINO"
    )
    parser.add_argument("--api",    action="store_true",
                        help="Chạy REST API server (FastAPI)")
    parser.add_argument("--host",   type=str,  default="0.0.0.0",
                        help="API host (mặc định: 0.0.0.0)")
    parser.add_argument("--port",   type=int,  default=8000,
                        help="API port (mặc định: 8000)")
    parser.add_argument("--rtsp",   type=str,  default=None,
                        help="URL camera RTSP (hỗ trợ nhiều URL, phân cách bằng dấu phẩy)")
    parser.add_argument("--reconnect-delay", type=float, default=3.0,
                        help="Thời gian chờ trước khi kết nối lại RTSP (giây, mặc định: 3.0)")
    parser.add_argument("--max-fps", type=float, default=15.0,
                        help="FPS tối đa cho RTSP inference, giảm để giảm CPU (mặc định: 15)")
    parser.add_argument("--source", type=str,  default="0",
                        help="Nguồn video: '0' webcam | đường dẫn video/ảnh (bỏ qua khi --api hoặc --rtsp)")
    parser.add_argument("--model",  type=str,  default=DEFAULT_MODEL,
                        help=f"Model path — plate detector (mặc định: {DEFAULT_MODEL}). "
                             "Các kích thước: v1n (nano), v1s (small), v1m (medium), v1l (large), v1x (xlarge)")
    parser.add_argument("--conf",   type=float, default=0.45,
                        help="Ngưỡng confidence (mặc định: 0.45)")
    parser.add_argument("--device", type=str,  default="CPU",
                        help="OpenVINO device: CPU | GPU | AUTO | NPU (mặc định: CPU)")
    parser.add_argument("--skip-frames", type=int, default=SKIP_FRAMES,
                        help=f"Inference 1/N frame để giảm CPU (mặc định: {SKIP_FRAMES}, lấy từ env SKIP_FRAMES). Đặt 1 để xử lý mọi frame")
    parser.add_argument("--infer-size", type=int, default=640,
                        help="Resize ảnh trước inference (mặc định: 640, giảm xuống 320 để giảm CPU)")
    parser.add_argument("--min-box-height", type=int, default=0,
                        help="Bỏ qua xe có bounding box cao < N pixel – lọc xe xa (mặc định: 0 = không lọc). "
                             "Có thể cấu hình riêng từng làn qua 'min_box_height' trong camera.json")
    parser.add_argument("--no-display", action="store_true",
                        help="Chạy headless – không mở cửa sổ OpenCV. "
                             "Dùng khi chạy trên server/dịch vụ không có màn hình. "
                             "Dừng bằng Ctrl+C. (--setup-region bị bỏ qua khi bật cờ này)")
    parser.add_argument("--setup-region", action="store_true",
                        help="Hiện UI tương tác để click 4 điểm vẽ vùng nhận diện, lưu vào .env")
    parser.add_argument("--save-plates", action="store_true",
                        help="Lưu ảnh chụp từ ALPR vào thư mục 'images'")
    parser.add_argument("--test-snapshot", action="store_true",
                        help="Test HTTP API chụp ảnh từ Hikvision, lưu ra file snapshot_test.jpg và thoát")
    parser.add_argument("--test-alpr", type=str, default="",
                        help="Test đọc biển số từ file ảnh chỉ định (--test-alpr path/to/img.jpg) và thoát")
    args = parser.parse_args()

    if args.test_snapshot:
        print(f"[*] Đang test chụp ảnh Snapshot từ {HKV_IP}...")
        url = HKV_SNAPSHOT_URL.format(ip=HKV_IP)
        try:
            resp = requests.get(url, auth=HTTPDigestAuth(HKV_USER, HKV_PASS), timeout=5)
            if resp.status_code == 401:
                resp = requests.get(url, auth=HTTPBasicAuth(HKV_USER, HKV_PASS), timeout=5)
            if resp.status_code == 200:
                with open("snapshot_test.jpg", "wb") as f:
                    f.write(resp.content)
                print("[OK] Đã chụp thành công và lưu vào snapshot_test.jpg")
            else:
                print(f"[FAIL] Lỗi gọi API chụp ảnh: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[ERROR] Quá trình test snapshot bị lỗi: {e}")
        exit(0)

    if args.test_alpr:
        img_path = args.test_alpr
        print(f"[*] Đang test đọc biển số từ ảnh: {img_path}")
        if not os.path.exists(img_path):
            print(f"[ERROR] Không tìm thấy file: {img_path}")
            exit(1)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Không thể đọc được ảnh (hỏng file): {img_path}")
            exit(1)

        plate_text = None
        img_drawn  = None

        # --- Bước 1: ONNX (OpenVINO) detect + fast-plate-ocr ---
        detector = get_plate_detector()
        ocr      = get_plate_ocr()
        if detector and ocr:
            boxes = detector.detect_boxes(img)
            if boxes:
                img_drawn = img.copy()
                plate_crops = []
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    plate_crops.append(img[y1:y2, x1:x2])
                texts = ocr.run(plate_crops)
                print(f"[OV+OCR] Phát hiện {len(plate_crops)} biển số:")
                for i, text in enumerate(texts):
                    print(f"  -> Biển số [{i+1}]: {text}")
                    if text and text.strip() and plate_text is None:
                        plate_text = text.strip().rstrip("_")
            else:
                print("[OV] Không phát hiện biển số → thử fast_alpr")
        else:
            print("[OV] Không load được detector/OCR → thử fast_alpr")

        # --- Bước 2: Fallback fast_alpr nếu YOLOv11 không phát hiện được biển số ---
        if plate_text is None:
            alpr = get_alpr()
            if alpr:
                alpr_results = alpr.predict(img)
                print(f"[fast_alpr fallback] Kết quả: {alpr_results}")
                if alpr_results and len(alpr_results) > 0:
                    for i, r in enumerate(alpr_results):
                        text = r.ocr.text if r.ocr else "N/A"
                        print(f"  -> Biển số [{i+1}]: {text}")
                        if r.ocr and r.ocr.text and plate_text is None:
                            plate_text = r.ocr.text
                    img_drawn = alpr.draw_predictions(img)
                else:
                    print("[fast_alpr] Không nhận diện được biển số nào.")
            else:
                print("[ERROR] Không load được fast_alpr.")

        if plate_text:
            print(f"\n[RESULT] Biển số: {plate_text}")
        else:
            print("\n[RESULT] Không nhận diện được biển số nào từ ảnh này.")

        if img_drawn is not None:
            cv2.imshow("Test ALPR", img_drawn)
            print("[INFO] Nhấn phím bất kỳ (trên cửa sổ ảnh) để thoát...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit(0)

    if args.rtsp:
        cam_configs_json = None
        if args.rtsp.endswith(".json") and os.path.exists(args.rtsp):
            import json
            with open(args.rtsp, 'r', encoding='utf-8') as f:
                cam_configs_json = json.load(f)
            rtsp_list = [cam.get("rtsp_url") for cam in cam_configs_json if cam.get("rtsp_url")]
        else:
            rtsp_list = [url.strip() for url in args.rtsp.split(",") if url.strip()]
            
        run_rtsp(
            rtsp_urls       = rtsp_list,
            model_path      = args.model,
            conf            = args.conf,
            device          = args.device,
            reconnect_delay = args.reconnect_delay,
            max_fps         = args.max_fps,
            skip_frames     = args.skip_frames,
            infer_size      = args.infer_size,
            save_plates     = args.save_plates,
            cam_configs     = cam_configs_json,
            min_box_height  = args.min_box_height,
            no_display      = args.no_display,
        )
    elif args.api:
        run_api(
            model_path=args.model,
            conf=args.conf,
            device=args.device,
            host=args.host,
            port=args.port,
        )
    else:
        run(
            source=args.source,
            model_path=args.model,
            conf=args.conf,
            device=args.device,
        )