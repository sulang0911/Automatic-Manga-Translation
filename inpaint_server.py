import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from flask import Flask, request, jsonify, make_response
    from flask_cors import CORS
    import numpy as np
    import cv2
    import json
except ImportError:
    print("\n[Error] Missing dependencies for inpaint server.")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

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
    import torch
    
    # Enable CUDA for PyTorch if available
    has_cuda = torch.cuda.is_available()
    gpu_info = "GPU" if has_cuda else "CPU"
    print(f"[*] PyTorch CUDA support: {has_cuda}. Loading LaMa image inpainting model on {gpu_info}...")
    
    lama_inpainter = SimpleLama()
    USE_LAMA = True
    print(f"[+] LaMa image inpainting model loaded successfully on {gpu_info}!")
except Exception as e:
    print(f"[*] Info: Failed to load LaMa: {e}. Will fallback to OpenCV inpainting.")

@app.route('/inpaint', methods=['POST'])
def inpaint_image():
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
        
        # Calculate background uniformity
        h_c, w_c = crop.shape[:2]
        border_pixels = []
        if h_c > 0 and w_c > 0:
            border_pixels.extend(crop[0, :])
            border_pixels.extend(crop[h_c-1, :])
            if h_c > 2:
                border_pixels.extend(crop[1:h_c-1, 0])
                border_pixels.extend(crop[1:h_c-1, w_c-1])
        
        border_pixels = np.array(border_pixels)
        is_uniform = True
        if len(border_pixels) > 0:
            border_gray = 0.299 * border_pixels[:, 2] + 0.587 * border_pixels[:, 1] + 0.114 * border_pixels[:, 0]
            is_uniform = np.std(border_gray) < 15.0
        
        if block_type == 'bubble' and is_uniform:
            dilated = dilate_mask(text_mask, bubble_dilation)
            crop[dilated == 255] = bg_color
            erased_img[ymin:ymax, xmin:xmax] = crop
        else:
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
                
                if inpainted.shape[0] != h_img or inpainted.shape[1] != w_img:
                    inpainted = inpainted[0:h_img, 0:w_img]
                    
                print("[+] Inpaint server: LaMa inpainting completed.")
            except Exception as e:
                print(f"[-] Inpaint server: LaMa error: {e}, falling back to OpenCV.")
                inpainted = None
                
        if inpainted is None:
            inpainted = cv2.inpaint(erased_img, inpaint_mask, 3, cv2.INPAINT_TELEA)
            
        inpainted_filtered = cv2.bilateralFilter(inpainted, 5, 50, 50)
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
        "device": "gpu" if (USE_LAMA and torch.cuda.is_available()) else "cpu"
    })

if __name__ == '__main__':
    print("[*] Inpaint server starting on 127.0.0.1:5001...")
    app.run(host='127.0.0.1', port=5001, debug=False)
