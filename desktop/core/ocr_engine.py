import os
import sys
import torch  # CRITICAL WINDOWS DLL PROTECTION: Must import torch before paddle
import uuid
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

try:
    from app.core.ocr.base import merge_adjacent_boxes
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from app.core.ocr.base import merge_adjacent_boxes

def get_background_color_hex(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return "#FFFFFF"
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return "#FFFFFF"
    border_pixels = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[h - 1, :])
    if h > 2:
        border_pixels.extend(crop[1:h - 1, 0])
        border_pixels.extend(crop[1:h - 1, w - 1])
    border_pixels = np.array(border_pixels)
    if len(border_pixels) == 0:
        return "#FFFFFF"
    median_color = np.median(border_pixels, axis=0).astype(int)
    # BGR to RGB
    b, g, r = median_color[0], median_color[1], median_color[2]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}".upper()

class OCREngine:
    def __init__(self, engine_type: str = "paddle", use_gpu: bool = True, lang: str = "japan"):
        self.engine_type = engine_type
        if self.engine_type == "cpu_paddle":
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu
        self.lang = lang
        self._paddle_ocr = None
        self._easyocr_reader = None

    def _init_paddle(self):
        if self._paddle_ocr is None:
            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["FLAGS_use_onednn"] = "0"
            os.environ["FLAGS_use_mkldnn"] = "0"

            has_paddle_cuda = False
            try:
                import paddle
                paddle.set_flags({"FLAGS_use_onednn": False})
                paddle.set_num_threads(1)
                has_paddle_cuda = paddle.device.is_compiled_with_cuda() and (paddle.device.cuda.device_count() > 0)
            except Exception:
                pass

            from paddleocr import PaddleOCR
            is_paddleocr_3x = hasattr(PaddleOCR, "_paddlex_pipeline_name")

            if is_paddleocr_3x:
                # PaddleOCR 3.x / PaddleX pipeline
                can_use_gpu = bool(self.use_gpu and has_paddle_cuda)
                if self.use_gpu and not has_paddle_cuda:
                    print("[*] PaddlePaddle CUDA not available in current environment. Using CPU mode.")
                    self.use_gpu = False

                device_str = "gpu" if can_use_gpu else "cpu"
                print(f"[*] Initializing PaddleOCR 3.x (lang={self.lang}, device={device_str}, model=PP-OCRv3 Mobile)...")

                kwargs = {
                    "lang": self.lang,
                    "ocr_version": "PP-OCRv3",  # 默认使用轻量级 Mobile 模型，内存仅占 ~150MB，避免 Medium 模型耗尽系统内存导致死机
                    "device": device_str,
                    "cpu_threads": 2,
                    "use_textline_orientation": True,
                    "enable_mkldnn": False,  # 禁用 oneDNN 规避 Windows CPU PIR double attribute 转换异常
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                }

                try:
                    self._paddle_ocr = PaddleOCR(**kwargs)
                except Exception as e:
                    if device_str == "gpu":
                        print(f"[!] Warning: PaddleOCR GPU init failed: {e}. Retrying CPU mode...")
                        self.use_gpu = False
                        kwargs["device"] = "cpu"
                        kwargs.pop("ocr_version", None)
                        self._paddle_ocr = PaddleOCR(**kwargs)
                    else:
                        raise
            else:
                # PaddleOCR 2.x legacy or unit test mock
                use_gpu_flag = bool(self.use_gpu)
                device_str = "gpu" if use_gpu_flag else "cpu"
                print(f"[*] Initializing PaddleOCR (lang={self.lang}, device={device_str}, use_gpu={use_gpu_flag})...")
                try:
                    self._paddle_ocr = PaddleOCR(
                        lang=self.lang,
                        device=device_str,
                        use_gpu=use_gpu_flag,
                        use_textline_orientation=True
                    )
                except Exception as e:
                    print(f"[!] Warning: PaddleOCR init with use_gpu={use_gpu_flag} failed: {e}. Retrying CPU mode...")
                    self.use_gpu = False
                    try:
                        self._paddle_ocr = PaddleOCR(
                            lang=self.lang,
                            device="cpu",
                            use_gpu=False,
                            use_textline_orientation=True
                        )
                    except Exception as e2:
                        print(f"[!] Warning: PaddleOCR CPU init failed: {e2}. Fallback to generic init...")
                        self._paddle_ocr = PaddleOCR(lang=self.lang)

    def _init_easyocr(self):
        if self._easyocr_reader is None:
            import easyocr
            langs = ['ja', 'en'] if self.lang in ['japan', 'ja'] else ['ch_sim', 'en']
            print(f"[*] Initializing EasyOCR (langs={langs}, gpu={self.use_gpu})...")
            self._easyocr_reader = easyocr.Reader(langs, gpu=self.use_gpu)

    def detect_and_recognize(self, image: np.ndarray, progress_callback=None) -> List[Dict[str, Any]]:
        """
        Runs OCR on an OpenCV BGR image and returns a list of translation blocks.
        Each block: {
            "id": str,
            "original_text": str,
            "translated_text": str,
            "xmin": float (0-100),
            "ymin": float (0-100),
            "xmax": float (0-100),
            "ymax": float (0-100),
            "bg_color": str (#HEX),
            "text_color": str (#HEX),
            "type": "bubble" | "onomatopoeia"
        }
        """
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        raw_boxes = []

        if progress_callback:
            progress_callback(10, "正在加载 OCR 识别引擎...")

        # 1. Run detection
        if self.engine_type == "easyocr":
            try:
                self._init_easyocr()
                if progress_callback:
                    progress_callback(30, "正在使用 EasyOCR 分析图像文字...")
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self._easyocr_reader.readtext(rgb)
                for bbox, text, conf in results:
                    if not text.strip() or conf < 0.2:
                        continue
                    pts = np.array(bbox, dtype=np.int32)
                    xmin = int(np.min(pts[:, 0]))
                    ymin = int(np.min(pts[:, 1]))
                    xmax = int(np.max(pts[:, 0]))
                    ymax = int(np.max(pts[:, 1]))
                    raw_boxes.append({
                        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                        "text": text.strip(), "conf": float(conf)
                    })
            except Exception as e:
                print(f"[-] EasyOCR error: {e}")
        else:
            # PaddleOCR default
            try:
                self._init_paddle()
                if progress_callback:
                    progress_callback(30, "正在使用 PaddleOCR 识别漫画对话...")
                
                # PaddleOCR inference
                results = self._paddle_ocr.ocr(image)
                if results and len(results) > 0 and results[0]:
                    if isinstance(results[0], dict):
                        # PaddleOCR 3.x (PaddleX pipeline) format
                        res_dict = results[0]
                        rec_texts = res_dict.get('rec_texts', [])
                        rec_scores = res_dict.get('rec_scores', [])
                        rec_polys = res_dict.get('rec_polys', [])
                        if (rec_polys is None or len(rec_polys) == 0) and 'dt_polys' in res_dict:
                            rec_polys = res_dict.get('dt_polys', [])

                        for i in range(len(rec_texts)):
                            text = str(rec_texts[i]).strip()
                            conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                            if not text or conf < 0.25:
                                continue
                            poly = rec_polys[i] if (rec_polys is not None and i < len(rec_polys)) else None
                            if poly is not None and len(poly) >= 4:
                                pts = np.array(poly, dtype=np.int32)
                                xmin = int(np.min(pts[:, 0]))
                                ymin = int(np.min(pts[:, 1]))
                                xmax = int(np.max(pts[:, 0]))
                                ymax = int(np.max(pts[:, 1]))
                                raw_boxes.append({
                                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                                    "text": text, "conf": conf
                                })
                            elif 'rec_boxes' in res_dict and i < len(res_dict['rec_boxes']):
                                box = res_dict['rec_boxes'][i]
                                raw_boxes.append({
                                    "xmin": int(box[0]), "ymin": int(box[1]),
                                    "xmax": int(box[2]), "ymax": int(box[3]),
                                    "text": text, "conf": conf
                                })
                    elif isinstance(results[0], list):
                        # PaddleOCR 2.x list format
                        for line in results[0]:
                            try:
                                pts = np.array(line[0], dtype=np.int32)
                                text_info = line[1]
                                text = text_info[0]
                                conf = float(text_info[1])
                                if not text.strip() or conf < 0.25:
                                    continue
                                xmin = int(np.min(pts[:, 0]))
                                ymin = int(np.min(pts[:, 1]))
                                xmax = int(np.max(pts[:, 0]))
                                ymax = int(np.max(pts[:, 1]))
                                raw_boxes.append({
                                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                                    "text": text.strip(), "conf": conf
                                })
                            except Exception:
                                continue
            except Exception as e:
                print(f"[-] PaddleOCR 发生异常: {e}。正在自动无缝回退至 EasyOCR 引擎...")
                try:
                    self._init_easyocr()
                    if progress_callback:
                        progress_callback(40, "PaddleOCR 异常，已自动切换为 EasyOCR 识别...")
                    res = self._easyocr_reader.readtext(image)
                    for bbox, text, conf in res:
                        if not text.strip() or conf < 0.25:
                            continue
                        pts = np.array(bbox, dtype=np.int32)
                        xmin = int(np.min(pts[:, 0]))
                        ymin = int(np.min(pts[:, 1]))
                        xmax = int(np.max(pts[:, 0]))
                        ymax = int(np.max(pts[:, 1]))
                        raw_boxes.append({
                            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                            "text": text.strip(), "conf": float(conf)
                        })
                except Exception as e_easy:
                    print(f"[-] EasyOCR 回退识别亦发生异常: {e_easy}")

        if progress_callback:
            progress_callback(70, "正在智能聚合气泡与段落排版...")

        # 2. Merge overlapping / closely adjacent text lines in speech bubbles
        merged_boxes = self._merge_adjacent_boxes(raw_boxes, w_img, h_img)

        # 3. Sort boxes by Japanese manga reading order: Right-to-Left, Top-to-Bottom
        merged_boxes = self._sort_manga_reading_order(merged_boxes, w_img)

        # 4. Convert to TranslationBlock format
        blocks = []
        for box in merged_boxes:
            xmin = max(0, min(box["xmin"], w_img - 1))
            ymin = max(0, min(box["ymin"], h_img - 1))
            xmax = max(0, min(box["xmax"], w_img))
            ymax = max(0, min(box["ymax"], h_img))
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = image[ymin:ymax, xmin:xmax]
            bg_color = get_background_color_hex(crop)

            # Determine type (bubble vs onomatopoeia) based on aspect ratio & background
            is_bubble = True
            aspect = (xmax - xmin) / max(1, (ymax - ymin))
            if aspect > 4.0 or aspect < 0.15:
                is_bubble = False

            block = {
                "id": str(uuid.uuid4())[:8],
                "original_text": box["text"],
                "translated_text": "",
                "xmin": round((xmin / w_img) * 100.0, 2),
                "ymin": round((ymin / h_img) * 100.0, 2),
                "xmax": round((xmax / w_img) * 100.0, 2),
                "ymax": round((ymax / h_img) * 100.0, 2),
                "bg_color": bg_color,
                "text_color": "#000000",
                "type": "bubble" if is_bubble else "onomatopoeia",
                "confidence": float(box.get("conf", 1.0)),
                "line_count": int(box.get("line_count", 1))
            }
            blocks.append(block)

        if progress_callback:
            progress_callback(95, f"OCR 识别完成，共提取 {len(blocks)} 个对话气泡")

        return blocks

    def _merge_adjacent_boxes(self, boxes: List[Dict[str, Any]], w_img: int, h_img: int) -> List[Dict[str, Any]]:
        return merge_adjacent_boxes(boxes, w_img, h_img)

    def _sort_manga_reading_order(self, boxes: List[Dict[str, Any]], w_img: int) -> List[Dict[str, Any]]:
        """
        Sorts boxes Right-to-Left and Top-to-Bottom.
        """
        # Bucket by vertical rows (~10% height) then sort columns right to left
        return sorted(boxes, key=lambda b: (int(b["ymin"] / 100) * 100, -b["xmin"]))
