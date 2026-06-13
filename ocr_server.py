import os
import sys

# Guide user to install dependencies if import fails
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import numpy as np
    import cv2
except ImportError:
    print("\n[错误] 缺少必要依赖，请先执行以下命令安装:")
    print("pip install flask flask-cors opencv-python numpy\n")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # 启用 CORS 跨域

# 尝试优先初始化 PaddleOCR (在日文/竖排漫画识别精度上处于业界领先地位)
USE_PADDLE = False
paddle_ocr = None
has_cuda = False
try:
    from paddleocr import PaddleOCR
    print("[*] 检测到已安装 PaddleOCR，正在初始化日文与竖排识别模型...")
    try:
        # 3.x 版本移除了 use_gpu 和 show_log 参数，改用 device 参数
        # 且 use_angle_cls 在 3.x 中已被弃用或用 use_textline_orientation 代替
        # 针对 CPU 运行，显式禁用 enable_mkldnn=False 规避 oneDNN static graph bug
        # 显式禁用 use_doc_orientation_classify 和 use_doc_unwarping，防止其预处理扭曲漫画坐标
        import paddle
        try:
            paddle.set_flags({"FLAGS_use_onednn": False})
        except Exception:
            pass
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
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang="japan", use_gpu=has_cuda)
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
                sys.exit(1)
except Exception as e:
    print(f"[-] 未安装或初始化 PaddleOCR 失败: {e}")
    sys.exit(1)

device = "GPU (CUDA)" if has_cuda else "CPU"
print(f"[*] 当前活动引擎: PaddleOCR，计算后端: {device}")

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

    # 使用高精度 PaddleOCR 识别
    if paddle_ocr:
        try:
            # 3.x 版本移了使用 predict，兼容 2.x 和 3.x 两种格式进行识别与解析
            try:
                results = paddle_ocr.ocr(img)
            except TypeError:
                results = paddle_ocr.ocr(img, cls=True)

            if results and results[0]:
                print(f"[DEBUG] results[0] type: {type(results[0])}", flush=True)
                if isinstance(results[0], dict):
                    res_dict = results[0]
                    rec_texts = res_dict.get('rec_texts', [])
                    rec_scores = res_dict.get('rec_scores', [])
                    rec_polys = res_dict.get('rec_polys', [])
                    print(f"[DEBUG] 3.x format: rec_texts length={len(rec_texts)}", flush=True)
                    if len(rec_texts) > 0:
                        print(f"[DEBUG] Sample texts: {rec_texts[:3]}", flush=True)
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
            print(f"[-] PaddleOCR 运行异常: {e}")
            return jsonify({"error": f"PaddleOCR processing error: {e}"}), 500

    print(f"[+] PaddleOCR 成功处理，提取出 {len(formatted_blocks)} 个文本区域", flush=True)
    return jsonify({"blocks": formatted_blocks})

# Helper functions for precision masking and inpainting
def get_background_color(crop):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return [255, 255, 255]
    border_pixels = []
    border_pixels.extend(crop[0, :])
    border_pixels.extend(crop[h-1, :])
    if h > 2:
        border_pixels.extend(crop[1:h-1, 0])
        border_pixels.extend(crop[1:h-1, w-1])
        
    border_pixels = np.array(border_pixels)
    if len(border_pixels) == 0:
        return [255, 255, 255]
    median_color = np.median(border_pixels, axis=0).astype(int)
    return [int(c) for c in median_color]

def get_text_mask(crop, bg_color):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg_gray = int(0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0])
    diff = cv2.absdiff(gray, bg_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return thresh

def dilate_mask(mask, dilation_pixels=4):
    if dilation_pixels <= 0:
        return mask
    kernel_size = dilation_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)

def blend_inpainted_image(original_img, inpainted_img, mask, feather_radius=5):
    if feather_radius <= 0:
        result = original_img.copy()
        result[mask > 0] = inpainted_img[mask > 0]
        return result
    
    ksize = feather_radius * 2 + 1
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    alpha = np.expand_dims(alpha, axis=2)
    
    blended = inpainted_img.astype(np.float32) * alpha + original_img.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)

# Check for simple-lama-inpainting support
USE_LAMA = False
lama_inpainter = None
try:
    from simple_lama_inpainting import SimpleLama
    print("[*] 检测到已安装 simple-lama-inpainting，正在初始化 LaMa 图像修复模型...")
    lama_inpainter = SimpleLama()
    USE_LAMA = True
    print("[+] LaMa 图像修复模型加载成功！")
except Exception as e:
    print("[*] 提示: 未安装 simple-lama-inpainting，将自动使用 OpenCV 优化修复算法（支持高精羽化融合）。")
    print("[*] 如需启用深度学习修复效果，可执行: pip install simple-lama-inpainting")

@app.route('/inpaint', methods=['POST'])
def inpaint_image():
    import json
    from flask import make_response
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
        
    file = request.files['image']
    blocks_str = request.form.get('blocks', '[]')
    style_str = request.form.get('style', '{}')
    
    try:
        blocks = json.loads(blocks_str)
        style = json.loads(style_str)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON parameters: {e}"}), 400
        
    # Decode image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400
        
    h_img, w_img = img.shape[:2]
    
    erased_img = img.copy()
    inpaint_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    # Calculate dynamic dilation sizes
    base_dim = max(h_img, w_img)
    bubble_dilation = max(1, int(base_dim * 0.002))       # e.g., 2-4px
    onomatopoeia_dilation = max(2, int(base_dim * 0.004)) # e.g., 4-8px
    
    for block in blocks:
        xmin = int((block.get('xmin', 0) / 100.0) * w_img)
        ymin = int((block.get('ymin', 0) / 100.0) * h_img)
        xmax = int((block.get('xmax', 0) / 100.0) * w_img)
        ymax = int((block.get('ymax', 0) / 100.0) * h_img)
        
        xmin = max(0, min(xmin, w_img - 1))
        ymin = max(0, min(ymin, h_img - 1))
        xmax = max(0, min(xmax, w_img))
        ymax = max(0, min(ymax, h_img))
        
        w_box = xmax - xmin
        h_box = ymax - ymin
        if w_box <= 0 or h_box <= 0:
            continue
            
        block_type = block.get('type', 'bubble')
        
        # Skip if style config ignores onomatopoeia
        if block_type == 'onomatopoeia' and style.get('onomatopoeiaMode') == 'ignore':
            continue
            
        crop = erased_img[ymin:ymax, xmin:xmax]
        bg_color = get_background_color(crop)
        text_mask = get_text_mask(crop, bg_color)
        
        if block_type == 'bubble':
            # 1. Bubble text: fill the precise dilated text mask with the bubble background color
            dilated = dilate_mask(text_mask, bubble_dilation)
            crop[dilated == 255] = bg_color
            erased_img[ymin:ymax, xmin:xmax] = crop
        else:
            # 2. Onomatopoeia/Background text: accumulate dilated precise mask for inpainting
            dilated = dilate_mask(text_mask, onomatopoeia_dilation)
            inpaint_mask[ymin:ymax, xmin:xmax] = cv2.bitwise_or(inpaint_mask[ymin:ymax, xmin:xmax], dilated)
            
    # Inpaint accumulated background/onomatopoeia text regions
    if np.sum(inpaint_mask) > 0:
        inpainted = None
        if USE_LAMA and lama_inpainter:
            try:
                from PIL import Image as PILImage
                img_rgb = cv2.cvtColor(erased_img, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(img_rgb)
                mask_pil = PILImage.fromarray(inpaint_mask)
                
                result_pil = lama_inpainter(img_pil, mask_pil)
                inpainted_rgb = np.array(result_pil)
                inpainted = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
                print("[+] 使用 LaMa 深度学习修复引擎处理完成")
            except Exception as e:
                print(f"[-] LaMa 修复引擎运行异常: {e}，将自动切换至 OpenCV 修复。")
                inpainted = None
                
        if inpainted is None:
            # OpenCV Telea
            inpainted = cv2.inpaint(erased_img, inpaint_mask, 3, cv2.INPAINT_TELEA)
            
        # Smooth with Bilateral Filter to preserve sharp lines while flat-filling
        inpainted_filtered = cv2.bilateralFilter(inpainted, 5, 50, 50)
        
        # Feather and blend back using the soft mask
        feather_radius = max(2, int(base_dim * 0.003))
        erased_img = blend_inpainted_image(erased_img, inpainted_filtered, inpaint_mask, feather_radius)
        
    _, buffer = cv2.imencode('.png', erased_img)
    response = make_response(buffer.tobytes())
    response.headers.set('Content-Type', 'image/png')
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "engine": "PaddleOCR",
        "device": device,
        "cuda_available": has_cuda,
        "lama_available": USE_LAMA
    })

if __name__ == '__main__':
    print("[*] 正在 127.0.0.1:5000 启动本地 OCR 服务...")
    app.run(host='127.0.0.1', port=5000, debug=False)
