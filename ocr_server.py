import os
import sys

# Guide user to install dependencies if import fails
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import numpy as np
    import cv2
    import torch
except ImportError:
    print("\n[错误] 缺少必要依赖，请先执行以下命令安装:")
    print("pip install flask flask-cors opencv-python numpy torch torchvision\n")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # 启用 CORS 跨域

# 尝试优先初始化 PaddleOCR (在日文/竖排漫画识别精度上处于业界领先地位)
USE_PADDLE = False
paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    print("[*] 检测到已安装 PaddleOCR，正在初始化日文与竖排识别模型...")
    try:
        # 3.x 版本移除了 use_gpu 和 show_log 参数，改用 device 参数
        # 且 use_angle_cls 在 3.x 中已被弃用或用 use_textline_orientation 代替
        # 针对 CPU 运行，显式禁用 enable_mkldnn=False 规避 oneDNN static graph bug
        # 显式禁用 use_doc_orientation_classify 和 use_doc_unwarping，防止其预处理扭曲漫画坐标
        import paddle
        has_cuda = paddle.device.is_compiled_with_cuda()
        device_str = "gpu" if has_cuda else "cpu"
        paddle_ocr = PaddleOCR(
            lang="japan", 
            device=device_str, 
            use_textline_orientation=True, 
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False
        )
        USE_PADDLE = True
        print(f"[+] PaddleOCR (日文) 3.x 引擎初始化成功！运行设备: {device_str.upper()}")
    except Exception as e3:
        print(f"[*] PaddleOCR 3.x 初始化未成功 ({e3})，正在尝试 2.x 兼容模式...")
        try:
            # 2.x 兼容模式
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang="japan", use_gpu=torch.cuda.is_available())
            USE_PADDLE = True
            print("[+] PaddleOCR (日文) 2.x 引擎初始化成功！")
        except Exception as e2:
            print(f"[-] PaddleOCR 2.x 初始化也失败: {e2}")
            print("[*] 尝试最基础的初始化方式...")
            try:
                paddle_ocr = PaddleOCR(lang="japan")
                USE_PADDLE = True
                print("[+] PaddleOCR (日文) 基础引擎初始化成功！")
            except Exception as e_base:
                print(f"[-] 所有 PaddleOCR 初始化尝试均失败: {e_base}")
except Exception as e:
    print(f"[*] 未安装或初始化 PaddleOCR 失败 (将尝试使用 EasyOCR 降级方案): {e}")

# 备用 EasyOCR 引擎
easy_reader = None
if not USE_PADDLE:
    try:
        import easyocr
        print(f"[*] 正在初始化 EasyOCR 备用模型 (支持语言: 日文 + 英文)...")
        easy_reader = easyocr.Reader(['ja', 'en'], gpu=torch.cuda.is_available())
        print("[+] EasyOCR 引擎初始化成功！")
    except Exception as e:
        print(f"[-] 初始化 EasyOCR 失败: {e}")
        print("\n[警告] 没有任何本地 OCR 引擎可用！请执行: pip install paddleocr paddlepaddle\n")

device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
engine_name = "PaddleOCR" if USE_PADDLE else "EasyOCR"
print(f"[*] 当前活动引擎: {engine_name}，计算后端: {device}")

@app.route('/ocr', methods=['POST'])
def run_ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    img_bytes = file.read()
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    formatted_blocks = []

    # 1. 尝试使用高精度 PaddleOCR 识别
    if USE_PADDLE and paddle_ocr:
        try:
            # 3.x 版本移除了 cls 参数且返回格式不同
            # 兼容 2.x 和 3.x 两种格式进行识别与解析
            try:
                results = paddle_ocr.ocr(img)
            except TypeError:
                results = paddle_ocr.ocr(img, cls=True)

            if results and results[0]:
                if isinstance(results[0], dict):
                    # 3.x 格式: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
                    res_dict = results[0]
                    rec_texts = res_dict.get('rec_texts', [])
                    rec_scores = res_dict.get('rec_scores', [])
                    rec_polys = res_dict.get('rec_polys', [])
                    for i in range(len(rec_texts)):
                        text = rec_texts[i]
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        poly = rec_polys[i] if i < len(rec_polys) else None
                        if poly is not None:
                            # 转换坐标为 [[x, y], ...]
                            if hasattr(poly, 'tolist'):
                                box = [[int(pt[0]), int(pt[1])] for pt in poly.tolist()]
                            else:
                                box = [[int(pt[0]), int(pt[1])] for pt in poly]
                            formatted_blocks.append({
                                "text": text,
                                "box": box,
                                "confidence": float(confidence)
                            })
                else:
                    # 2.x 格式: [[ [coordinates, (text, confidence)], ... ]]
                    for line in results[0]:
                        bbox = line[0]
                        text = line[1][0]
                        confidence = line[1][1]
                        box = [[int(pt[0]), int(pt[1])] for pt in bbox]
                        formatted_blocks.append({
                            "text": text,
                            "box": box,
                            "confidence": float(confidence)
                        })
        except Exception as e:
            print(f"[-] PaddleOCR 运行异常，将切换至 EasyOCR 备用处理: {e}")

    # 2. 降级备用：EasyOCR 识别
    if not formatted_blocks and easy_reader:
        try:
            results = easy_reader.readtext(img, detail=1)
            for bbox, text, confidence in results:
                box = [[int(pt[0]), int(pt[1])] for pt in bbox]
                formatted_blocks.append({
                    "text": text,
                    "box": box,
                    "confidence": float(confidence)
                })
        except Exception as e:
            print(f"[-] EasyOCR 运行异常: {e}")

    print(f"[+] 识别引擎 {engine_name} 成功处理，提取出 {len(formatted_blocks)} 个文本区域", flush=True)
    return jsonify({"blocks": formatted_blocks})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "engine": "PaddleOCR" if USE_PADDLE else "EasyOCR",
        "device": device,
        "cuda_available": torch.cuda.is_available()
    })

if __name__ == '__main__':
    print("[*] 正在 127.0.0.1:5000 启动本地 OCR 服务...")
    app.run(host='127.0.0.1', port=5000, debug=False)
