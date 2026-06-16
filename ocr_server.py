import os

# Ensure torch is imported first to avoid DLL/OMP conflict on Windows
try:
    import torch
except ImportError:
    pass

# Add nvidia DLL directories from venv to the DLL search path on Windows
if os.name == "nt":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    site_packages = os.path.join(base_dir, "venv", "Lib", "site-packages")
    if os.path.exists(site_packages):
        nvidia_dir = os.path.join(site_packages, "nvidia")
        if os.path.exists(nvidia_dir):
            for folder in os.listdir(nvidia_dir):
                bin_path = os.path.join(nvidia_dir, folder, "bin")
                if os.path.exists(bin_path):
                    try:
                        os.add_dll_directory(bin_path)
                    except Exception:
                        pass

# Limit CPU threads to prevent CPU 100% max-out and freezing when running on CPU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import warnings

# Suppress all python warnings (like RequestsDependencyWarning, ccache UserWarning, DeprecationWarning)
warnings.filterwarnings("ignore")

# Force stdout/stderr to use UTF-8 to avoid encoding errors (e.g. GBK on Windows) when printing Japanese characters
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Guide user to install dependencies if import fails
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import numpy as np
    import cv2
    import flask.cli
    # Suppress Flask development server banner warning
    flask.cli.show_server_banner = lambda *args: None
except ImportError:
    print("\n[错误] 缺少必要依赖，请先执行以下命令安装:")
    print("pip install flask flask-cors opencv-python numpy\n")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # 启用 CORS 跨域

# ==========================================
# OCR Engine Selection and Initialization
# ==========================================
import json
import time

CONFIG_FILE = "ocr_config.json"
active_engine = "paddle"
paddle_ocr = None
easyocr_reader = None
has_cuda = False
device_str = "cpu"
gpu_info = ""

def get_saved_choice():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("ocr_engine", "1")
        except Exception:
            pass
    return "1"

def save_choice(choice):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({"ocr_engine": choice}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[-] 保存配置失败: {e}")

def select_menu_interactive():
    print("\n" + "="*60)
    print("      AetherLens 本地高精度 OCR 服务启动器")
    print("="*60)
    print("请选择运行引擎与显卡兼容模式 (输入 1-4 序号):")
    print("\n[1] [推荐] PP-OCRv4 (高精度，适合 RTX 20/30/40 等较新显卡)")
    print("    - 状态: GPU 加速 (若 CUDA 可用)")
    print("    - 备注: 默认高精度，老显卡在 CUDA 12 下可能会识别乱码。")
    print("\n[2] [兼容] PP-OCRv3 (老显卡 GPU 加速，如 GTX 10/9/Titan/P40)")
    print("    - 状态: GPU 加速 (若 CUDA 可用)")
    print("    - 备注: 避开老卡在 v4 + CUDA 12 下的乱码 bug，速度与兼容性极佳。")
    print("\n[3] [通用] EasyOCR (通用 GPU，适合所有 NVIDIA 显卡，基于 PyTorch)")
    print("    - 状态: GPU 加速 (若 CUDA 可用)")
    print("    - 备注: 需要安装 easyocr。对老显卡极其友好，完美支持老显卡 GPU 加速。")
    print("\n[4] [CPU] 强制使用 CPU 运行 (极度稳定，不占用显存)")
    print("    - 状态: 仅使用 CPU 运算")
    print("    - 备注: 适合没有显卡、不支持 CUDA 或需要静默稳定运行的用户。")
    print("="*60)

    choice = None
    while choice not in ["1", "2", "3", "4"]:
        try:
            choice = input("\n请选择序号 [1-4]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[!] 取消选择，默认选择 1")
            return "1"
    save_choice(choice)
    return choice

def get_engine_choice():
    # Check command line args
    for arg in sys.argv:
        if arg.startswith("--model="):
            c = arg.split("=")[1].strip()
            if c in ["1", "2", "3", "4"]:
                return c

    saved = get_saved_choice()
    
    # Countdown logic (Only interactive if run directly and on Windows)
    is_windows = os.name == 'nt'
    if is_windows and sys.stdin.isatty():
        import msvcrt
        print("\n" + "="*60)
        print("      AetherLens 本地高精度 OCR 服务启动器")
        print("="*60)
        print(f"[*] 历史配置: 选项 [{saved}]")
        print("[*] 正在倒计时 3 秒自动启动（按任意键进入修改菜单）...")
        sys.stdout.write("倒计时: ")
        sys.stdout.flush()
        
        endtime = time.time() + 3
        pressed = False
        last_sec = 4
        while time.time() < endtime:
            remaining = int(endtime - time.time()) + 1
            if remaining < last_sec:
                sys.stdout.write(f"{remaining}.. ")
                sys.stdout.flush()
                last_sec = remaining
            
            if msvcrt.kbhit():
                # Flush keypress buffer
                while msvcrt.kbhit():
                    msvcrt.getch()
                pressed = True
                break
            time.sleep(0.05)
        print()
        if pressed:
            return select_menu_interactive()
        else:
            print(f"[+] 自动启动选项 [{saved}]")
            return saved
    else:
        print(f"[+] 直接使用选项 [{saved}] 启动服务...")
        return saved

# Get user choice
choice = get_engine_choice()

# Initialize selected engine
if choice in ["1", "2", "4"]:
    # PaddleOCR path
    try:
        from paddleocr import PaddleOCR
        import paddle
        print("[*] paddle module path:", paddle.__file__, flush=True)
        try:
            paddle.set_flags({"FLAGS_use_onednn": False})
            paddle.set_num_threads(4)
        except Exception:
            pass
        
        has_cuda = paddle.device.is_compiled_with_cuda()
        
        if choice == "4":
            device_str = "cpu"
            gpu_info = "CPU (已强制使用 CPU)"
        elif has_cuda:
            try:
                gpu_name = paddle.device.cuda.get_device_properties(0).name
                gpu_info = f"GPU ({gpu_name})"
                device_str = "gpu"
            except Exception:
                gpu_info = "CPU (获取GPU设备失败，回退到CPU)"
                device_str = "cpu"
        else:
            gpu_info = "CPU"
            device_str = "cpu"

        if choice == "1":
            print(f"[*] 正在初始化 PaddleOCR (默认高精度模型)，运行设备: {gpu_info}...")

            # Warning for older cards on default model
            if has_cuda:
                try:
                    gpu_name_lower = paddle.device.cuda.get_device_properties(0).name.lower()
                    is_old_gpu = any(x in gpu_name_lower for x in ["1080", "1070", "1060", "1050", "titan xp", "p40", "p100", "980", "970", "960"])
                    if is_old_gpu:
                        print("\n[⚠️ 警告] 检测到 Pascal/Maxwell 架构老显卡（GTX 10/9 系列等）。")
                        print("         默认高精度模型在老卡+CUDA 12 下可能有推理算力Bug，可能会导致漫画文本识别出乱码！")
                        print("         如果您遇到识别全乱码的情况，请重启服务并选择选项 [2] (PP-OCRv3) 或 [3] (EasyOCR)。\n")
                except Exception:
                    pass

            paddle_ocr = PaddleOCR(
                lang="japan", 
                device=device_str, 
                use_textline_orientation=True, 
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
            ocr_ver = "Default"
        else:
            ocr_ver = "PP-OCRv3"
            print(f"[*] 正在初始化 PaddleOCR ({ocr_ver})，运行设备: {gpu_info}...")
            paddle_ocr = PaddleOCR(
                lang="japan", 
                device=device_str, 
                ocr_version=ocr_ver,
                use_textline_orientation=True, 
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
        active_engine = "paddle"
        print(f"[+] PaddleOCR ({ocr_ver}) 引擎加载成功！设备: {device_str.upper()}")
    except Exception as e:
        print(f"[-] PaddleOCR 初始化失败: {e}")
        sys.exit(1)

elif choice == "3":
    # EasyOCR path
    print("[*] 正在加载 EasyOCR (基于 PyTorch)...")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info = f"GPU ({gpu_name})"
            device_str = "gpu"
        else:
            gpu_info = "CPU"
            device_str = "cpu"
    except ImportError:
        has_cuda = False
        device_str = "cpu"
        gpu_info = "CPU"

    try:
        import easyocr
        print(f"[*] 正在初始化 EasyOCR，运行设备: {gpu_info}...")
        easyocr_reader = easyocr.Reader(['ja', 'en'], gpu=has_cuda)
        active_engine = "easyocr"
        print(f"[+] EasyOCR (ja, en) 引擎加载成功！运行设备: {gpu_info}")
    except ImportError:
        print("\n[❌ 错误] 检测到您选择使用 EasyOCR，但本地未安装该包，请先执行以下命令安装:")
        print("  pip install easyocr")
        print("或者安装具有 GPU 加速的 PyTorch (根据您的 CUDA 版本，例如 cu121):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\n正在回退到选项 [4] (纯 CPU 运行 PaddleOCR)...")
        # Fallback to CPU PaddleOCR
        try:
            from paddleocr import PaddleOCR
            paddle_ocr = PaddleOCR(
                lang="japan", 
                device="cpu", 
                use_textline_orientation=True, 
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False
            )
            active_engine = "paddle"
            device_str = "cpu"
            gpu_info = "CPU (已自动回退)"
            print("[+] 回退到 CPU 运行 PaddleOCR (PP-OCRv4) 成功！")
        except Exception as fallback_err:
            print(f"[-] 回退失败: {fallback_err}")
            sys.exit(1)

device = gpu_info
print(f"[*] 当前活动引擎: {active_engine.upper()}，运行后端: {device}")

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

    if active_engine == "paddle" and paddle_ocr:
        try:
            try:
                results = paddle_ocr.ocr(img)
            except TypeError:
                results = paddle_ocr.ocr(img, cls=True)

            if results and results[0]:
                if isinstance(results[0], dict):
                    res_dict = results[0]
                    rec_texts = res_dict.get('rec_texts', [])
                    rec_scores = res_dict.get('rec_scores', [])
                    rec_polys = res_dict.get('rec_polys', [])
                    for i in range(len(rec_texts)):
                        text = rec_texts[i]
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        poly = rec_polys[i] if i < len(rec_polys) else None
                        if poly is not None:
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

    elif active_engine == "easyocr" and easyocr_reader:
        try:
            results = easyocr_reader.readtext(img)
            for line in results:
                bbox = line[0]
                text = line[1]
                confidence = line[2]
                box = [[int(pt[0]), int(pt[1])] for pt in bbox]
                formatted_blocks.append({
                    "text": text,
                    "box": box,
                    "confidence": float(confidence)
                })
        except Exception as e:
            print(f"[-] EasyOCR 运行异常: {e}")
            return jsonify({"error": f"EasyOCR processing error: {e}"}), 500

    print(f"[+] OCR 引擎({active_engine.upper()})成功处理，提取出 {len(formatted_blocks)} 个文本区域", flush=True)
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
# Helper function to check if inpaint server is running
def check_inpaint_server_active():
    import requests
    try:
        res = requests.get('http://127.0.0.1:5001/health', timeout=1.0)
        if res.ok:
            data = res.json()
            return data.get("status") == "healthy"
    except Exception:
        pass
    return False

@app.route('/inpaint', methods=['POST'])
def inpaint_image():
    import requests
    from flask import make_response
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
        
    file = request.files['image']
    blocks_str = request.form.get('blocks', '[]')
    style_str = request.form.get('style', '{}')
    
    # Check if inpaint_server is active on port 5001
    inpaint_active = check_inpaint_server_active()
    
    if inpaint_active:
        # Proxy request to port 5001
        try:
            filename = file.filename or 'image.png'
            file.seek(0)
            files = {'image': (filename, file.read(), file.content_type)}
            data = {'blocks': blocks_str, 'style': style_str}
            
            res = requests.post('http://127.0.0.1:5001/inpaint', files=files, data=data, timeout=60.0)
            if res.ok:
                response = make_response(res.content)
                response.headers.set('Content-Type', 'image/png')
                return response
            else:
                return jsonify({"error": f"Inpaint server returned error: {res.status_code}"}), res.status_code
        except Exception as e:
            print(f"[-] Failed to communicate with inpaint server: {e}")
            # Fallback to local OpenCV if connection fails
            file.seek(0)
            
    # Local OpenCV-only inpainting fallback if port 5001 is offline or failed
    try:
        import json
        blocks = json.loads(blocks_str)
        style = json.loads(style_str)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON parameters: {e}"}), 400
        
    file.seek(0)
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400
        
    h_img, w_img = img.shape[:2]
    erased_img = img.copy()
    
    # Perform OpenCV fast speech bubble flat fills
    base_dim = max(h_img, w_img)
    bubble_dilation = max(1, int(base_dim * 0.002))
    
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
        if block_type == 'onomatopoeia' and style.get('onomatopoeiaMode') == 'ignore':
            continue
            
        crop = erased_img[ymin:ymax, xmin:xmax]
        bg_color = get_background_color(crop)
        text_mask = get_text_mask(crop, bg_color)
        
        # Simple OpenCV fill
        dilated = dilate_mask(text_mask, bubble_dilation)
        crop[dilated == 255] = bg_color
        erased_img[ymin:ymax, xmin:xmax] = crop
        
    _, buffer = cv2.imencode('.png', erased_img)
    response = make_response(buffer.tobytes())
    response.headers.set('Content-Type', 'image/png')
    return response

@app.route('/health', methods=['GET'])
def health():
    inpaint_active = check_inpaint_server_active()
    
    # Get exact ocr engine name
    detailed_engine = "PaddleOCR"
    glbs = globals()
    active_eng = glbs.get("active_engine", "paddle")
    chc = glbs.get("choice", "1")
    ver = glbs.get("ocr_ver", "Default")
    
    if active_eng == "paddle":
        if ver == "PP-OCRv3":
            detailed_engine = "PaddleOCR (PP-OCRv3)"
        elif chc == "4":
            detailed_engine = "PaddleOCR (PP-OCRv4 CPU)"
        else:
            detailed_engine = "PaddleOCR (PP-OCRv4)"
    elif active_eng == "easyocr":
        detailed_engine = "EasyOCR"
        
    return jsonify({
        "status": "healthy",
        "engine": "PaddleOCR" if active_eng == "paddle" else "EasyOCR",
        "engine_detail": detailed_engine,
        "device": device,
        "cuda_available": has_cuda,
        "lama_available": inpaint_active
    })

if __name__ == '__main__':
    print('[*] 正在 127.0.0.1:5000 启动本地 OCR 服务...')
    app.run(host='127.0.0.1', port=5000, debug=False)
