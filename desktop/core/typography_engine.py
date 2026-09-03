import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple

# Common standard font lookup on Windows
WINDOWS_FONTS = [
    "msyh.ttc",       # Microsoft YaHei
    "simhei.ttf",     # SimHei
    "simsun.ttc",     # SimSun
    "msgothic.ttc",   # MS Gothic (Japanese)
    "meiryo.ttc",     # Meiryo
    "arial.ttf",      # Arial
    "seguiemj.ttf"    # Segoe UI
]

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (0, 0, 0)

FONT_NAME_MAP = {
    "霞鹜文楷": "LXGWWenKaiLite-Regular.ttf",
    "lxgw wenkai": "LXGWWenKaiLite-Regular.ttf",
    "得意黑": "SmileySans-Oblique.ttf",
    "smiley sans": "SmileySans-Oblique.ttf",
    "幼圆": "SIMYOU.TTF",
    "youyuan": "SIMYOU.TTF",
    "comic sans ms": "comic.ttf",
    "comic sans": "comic.ttf",
    "segoe print": "segoepr.ttf",
    "ink free": "Inkfree.ttf",
    "华文楷体": "STKAITI.TTF",
    "楷体": "simkai.ttf",
    "kaiti": "simkai.ttf",
    "microsoft yahei": "msyh.ttc",
    "微软雅黑": "msyh.ttc",
    "simhei": "simhei.ttf",
    "黑体": "simhei.ttf",
}

class TypographyEngine:
    def __init__(self):
        self._font_cache = {}

    def _get_font(self, font_family: str, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        cache_key = (font_family, size, bold)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        font_dirs = [
            os.path.join(repo_root, "assets", "fonts"),
            os.path.join(os.getcwd(), "assets", "fonts"),
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
            "C:\\Windows\\Fonts",
            os.path.dirname(os.path.abspath(__file__))
        ]

        # Clean name, e.g. "霞鹜文楷 (日漫萌系)" -> "霞鹜文楷"
        clean_name = font_family.split("(")[0].split("（")[0].strip()
        mapped_file = FONT_NAME_MAP.get(clean_name.lower(), f"{clean_name}.ttf")

        # Try to match font_family directly or fall back to standard fonts
        chosen_font = None
        for font_dir in font_dirs:
            if not os.path.exists(font_dir):
                continue
            cand = os.path.join(font_dir, mapped_file)
            if os.path.exists(cand):
                chosen_font = cand
                break
            candidate = os.path.join(font_dir, f"{clean_name}.ttf")
            if os.path.exists(candidate):
                chosen_font = candidate
                break
            candidate = os.path.join(font_dir, f"{clean_name}.ttc")
            if os.path.exists(candidate):
                chosen_font = candidate
                break

        if not chosen_font:
            for fname in ["LXGWWenKaiLite-Regular.ttf", "msyh.ttc", "simhei.ttf", "arial.ttf"]:
                for font_dir in font_dirs:
                    cand = os.path.join(font_dir, fname)
                    if os.path.exists(cand):
                        chosen_font = cand
                        break
                if chosen_font:
                    break

        try:
            if chosen_font:
                font = ImageFont.truetype(chosen_font, size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def render_translations(self, base_image: np.ndarray, blocks: List[Dict[str, Any]], style_config: Dict[str, Any]) -> np.ndarray:
        if base_image is None or base_image.size == 0:
            return base_image

        h_img, w_img = base_image.shape[:2]
        # Convert BGR OpenCV image to PIL Image (RGBA)
        img_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        font_family = style_config.get("font_family", "Microsoft YaHei")
        font_size_scale = style_config.get("font_size_scale", 1.0)
        auto_fit = style_config.get("auto_fit_font_size", True)
        font_bold = style_config.get("font_bold", False)
        bg_opacity = style_config.get("bg_opacity", 0.95)
        bg_mode = style_config.get("bg_color_mode", "original") # original | custom | none
        custom_bg_hex = style_config.get("bg_color", "#FFFFFF")
        stroke_mode = style_config.get("stroke_mode", "auto")
        custom_stroke_hex = style_config.get("stroke_color", "#FFFFFF")
        stroke_width = int(style_config.get("stroke_width", 2))
        text_color_mode = style_config.get("text_color_mode", "custom")
        custom_text_hex = style_config.get("text_color", "#000000")

        for block in blocks:
            text = block.get("translated_text", "").strip()
            if not text:
                continue

            # Convert percentage coords to pixels
            xmin = int((block.get("xmin", 0) / 100.0) * w_img)
            ymin = int((block.get("ymin", 0) / 100.0) * h_img)
            xmax = int((block.get("xmax", 0) / 100.0) * w_img)
            ymax = int((block.get("ymax", 0) / 100.0) * h_img)

            box_w = max(10, xmax - xmin)
            box_h = max(10, ymax - ymin)

            # Colors
            if bg_mode == "custom":
                bg_rgb = hex_to_rgb(custom_bg_hex)
            elif bg_mode == "original":
                bg_rgb = hex_to_rgb(block.get("bg_color", "#FFFFFF"))
            else: # none
                bg_rgb = None

            if text_color_mode == "custom":
                text_rgb = hex_to_rgb(custom_text_hex)
            else:
                text_rgb = hex_to_rgb(block.get("text_color", "#000000"))

            # 1. Draw bubble background pill/rounded rectangle if opacity > 0 and bg_rgb is not None
            if bg_rgb and bg_opacity > 0:
                bg_alpha = int(bg_opacity * 255)
                # Draw rounded rectangle for speech bubble with slight padding
                padding = max(2, int(min(box_w, box_h) * 0.05))
                rect_coords = [xmin - padding, ymin - padding, xmax + padding, ymax + padding]
                draw.rounded_rectangle(rect_coords, radius=max(4, int(min(box_w, box_h) * 0.15)), 
                                       fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], bg_alpha))

            # 2. Determine best font size
            lines, font, font_size = self._fit_text_to_box(
                text, box_w, box_h, font_family, font_size_scale, auto_fit, font_bold
            )

            # 3. Calculate text stroke / outline color
            stroke_rgb = None
            calc_stroke_w = 0
            if stroke_mode == "manual":
                stroke_rgb = hex_to_rgb(custom_stroke_hex)
                calc_stroke_w = max(1, stroke_width)
            elif stroke_mode == "auto":
                # Auto contrast: if text is dark, stroke is white; if text is light, stroke is black
                luminance = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
                stroke_rgb = (255, 255, 255) if luminance < 128 else (0, 0, 0)
                calc_stroke_w = max(1, int(font_size * 0.08))

            # 4. Render centered text lines
            total_text_h = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines)
            line_spacing = max(2, int(font_size * 0.2))
            total_h = total_text_h + (len(lines) - 1) * line_spacing

            start_y = ymin + (box_h - total_h) / 2.0
            cur_y = start_y

            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                cur_x = xmin + (box_w - lw) / 2.0

                if stroke_rgb and calc_stroke_w > 0:
                    draw.text(
                        (cur_x, cur_y),
                        line,
                        font=font,
                        fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255),
                        stroke_width=calc_stroke_w,
                        stroke_fill=(stroke_rgb[0], stroke_rgb[1], stroke_rgb[2], 255)
                    )
                else:
                    draw.text(
                        (cur_x, cur_y),
                        line,
                        font=font,
                        fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255)
                    )
                cur_y += lh + line_spacing

        # Merge overlay with original
        merged_pil = Image.alpha_composite(pil_img, overlay)
        res_rgb = np.array(merged_pil.convert("RGB"))
        return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

    def _fit_text_to_box(self, text: str, box_w: int, box_h: int, font_family: str, 
                          font_size_scale: float, auto_fit: bool, bold: bool):
        # Base size estimate
        char_count = max(1, len(text.replace("\n", "")))
        aspect = box_h / max(1, box_w)
        
        # Binary search or heuristic font size fitting
        min_sz = 9
        max_sz = min(72, int(box_h * 0.65 * font_size_scale))
        best_sz = min_sz
        best_lines = [text]

        if not auto_fit:
            target_sz = max(10, int(20 * font_size_scale))
            font = self._get_font(font_family, target_sz, bold)
            lines = self._wrap_text(text, box_w, font)
            return lines, font, target_sz

        low = min_sz
        high = max(min_sz, max_sz)

        while low <= high:
            mid = (low + high) // 2
            font = self._get_font(font_family, mid, bold)
            lines = self._wrap_text(text, box_w, font)
            
            # Measure height
            dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            line_spacing = max(2, int(mid * 0.2))
            total_h = 0
            all_fit_w = True
            for line in lines:
                bbox = dummy_draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                if lw > box_w * 1.05:
                    all_fit_w = False
                    break
                total_h += lh + line_spacing

            if all_fit_w and total_h <= box_h * 1.05:
                best_sz = mid
                best_lines = lines
                low = mid + 1
            else:
                high = mid - 1

        scaled_sz = max(min_sz, int(best_sz * font_size_scale))
        final_font = self._get_font(font_family, scaled_sz, bold)
        final_lines = self._wrap_text(text, box_w, final_font)
        return final_lines, final_font, scaled_sz

    def _wrap_text(self, text: str, max_w: int, font: ImageFont.FreeTypeFont) -> List[str]:
        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        paragraphs = text.split("\n")
        all_lines = []

        for p in paragraphs:
            if not p:
                continue
            cur_line = ""
            for char in p:
                test_line = cur_line + char
                bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
                w = bbox[2] - bbox[0]
                if w > max_w and cur_line:
                    all_lines.append(cur_line)
                    cur_line = char
                else:
                    cur_line = test_line
            if cur_line:
                all_lines.append(cur_line)

        return all_lines or [text]
