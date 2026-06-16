import Tesseract from 'tesseract.js';
import type { TranslationBlock } from '../types';
import { getImageDimensions } from './translator';

// Estimates text and background colors inside a bounding box of an image
export const analyzeBlockColors = (
  img: HTMLImageElement,
  ymin: number,
  xmin: number,
  ymax: number,
  xmax: number
): { text_color: string; bg_color: string } => {
  try {
    const tempCanvas = document.createElement('canvas');
    const ctx = tempCanvas.getContext('2d');
    if (!ctx) return { text_color: '#000000', bg_color: '#FFFFFF' };

    const x = (xmin / 100) * img.naturalWidth;
    const y = (ymin / 100) * img.naturalHeight;
    const w = ((xmax - xmin) / 100) * img.naturalWidth;
    const h = ((ymax - ymin) / 100) * img.naturalHeight;

    if (w <= 0 || h <= 0) return { text_color: '#000000', bg_color: '#FFFFFF' };

    tempCanvas.width = Math.min(w, 150); // Scale down for speed
    tempCanvas.height = Math.min(h, 80);
    ctx.drawImage(img, x, y, w, h, 0, 0, tempCanvas.width, tempCanvas.height);

    const imgData = ctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imgData.data;

    // 1. Estimate background color using median of border pixels
    const borderPixels: [number, number, number][] = [];
    const tw = tempCanvas.width;
    const th = tempCanvas.height;
    
    // Top and bottom borders
    for (let col = 0; col < tw; col++) {
      const idxTop = col * 4;
      borderPixels.push([data[idxTop], data[idxTop + 1], data[idxTop + 2]]);
      const idxBot = ((th - 1) * tw + col) * 4;
      borderPixels.push([data[idxBot], data[idxBot + 1], data[idxBot + 2]]);
    }
    // Left and right borders
    for (let row = 1; row < th - 1; row++) {
      const idxLeft = row * tw * 4;
      borderPixels.push([data[idxLeft], data[idxLeft + 1], data[idxLeft + 2]]);
      const idxRight = (row * tw + (tw - 1)) * 4;
      borderPixels.push([data[idxRight], data[idxRight + 1], data[idxRight + 2]]);
    }

    if (borderPixels.length === 0) return { text_color: '#000000', bg_color: '#FFFFFF' };

    const getMedianColor = (pixels: [number, number, number][]): [number, number, number] => {
      const r = pixels.map(p => p[0]).sort((a, b) => a - b);
      const g = pixels.map(p => p[1]).sort((a, b) => a - b);
      const b = pixels.map(p => p[2]).sort((a, b) => a - b);
      const mid = Math.floor(pixels.length / 2);
      return [r[mid], g[mid], b[mid]];
    };

    const bgColorRGB = getMedianColor(borderPixels);

    // 2. Scan all pixels inside the block to find distinct text pixels (far from background color)
    let sumR = 0, sumG = 0, sumB = 0;
    let textPixelCount = 0;
    
    let minBrightness = 255;
    let maxBrightness = 0;
    let minColor = [0, 0, 0];
    let maxColor = [255, 255, 255];

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const brightness = (r * 299 + g * 587 + b * 114) / 1000;

      if (brightness < minBrightness) {
        minBrightness = brightness;
        minColor = [r, g, b];
      }
      if (brightness > maxBrightness) {
        maxBrightness = brightness;
        maxColor = [r, g, b];
      }

      // Manhattan distance in RGB space
      const dist = Math.abs(r - bgColorRGB[0]) + Math.abs(g - bgColorRGB[1]) + Math.abs(b - bgColorRGB[2]);
      
      // If the pixel is significantly different from background, count it as text
      if (dist > 60) {
        sumR += r;
        sumG += g;
        sumB += b;
        textPixelCount++;
      }
    }

    const rgbToHex = (r: number, g: number, b: number) =>
      '#' + [r, g, b].map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0')).join('');

    let textColorHex = '';
    if (textPixelCount > 8) {
      // Compute the average color of all non-background text pixels
      const avgR = sumR / textPixelCount;
      const avgG = sumG / textPixelCount;
      const avgB = sumB / textPixelCount;
      textColorHex = rgbToHex(avgR, avgG, avgB);
    } else {
      // Fallback: use classic contrast brightness extraction
      const avgBgBrightness = (bgColorRGB[0] * 299 + bgColorRGB[1] * 587 + bgColorRGB[2] * 114) / 1000;
      if (avgBgBrightness > 127) {
        textColorHex = rgbToHex(minColor[0], minColor[1], minColor[2]);
      } else {
        textColorHex = rgbToHex(maxColor[0], maxColor[1], maxColor[2]);
      }
    }

    return {
      text_color: textColorHex,
      bg_color: rgbToHex(bgColorRGB[0], bgColorRGB[1], bgColorRGB[2]),
    };
  } catch (e) {
    console.error('Local color analysis failed', e);
    return { text_color: '#000000', bg_color: '#FFFFFF' };
  }
};

// Helper to ping local GPU OCR server
const checkLocalOcrServer = async (): Promise<boolean> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5000ms generous ping for GPU startup
    const res = await fetch('http://127.0.0.1:5000/health', { signal: controller.signal });
    clearTimeout(timeoutId);
    if (res.ok) {
      const data = await res.json();
      return data.status === 'healthy';
    }
  } catch (e) {
    // Local server is not running
  }
  return false;
};

// Main local OCR runner
export const performLocalOCR = async (
  file: File,
  sourceLang: string,
  onProgress?: (progress: number) => void
): Promise<TranslationBlock[]> => {
  onProgress?.(15);
  const dims = await getImageDimensions(file);
  onProgress?.(30);

  let imgUrl: string | undefined = undefined;
  try {
    // Load image element to perform color analysis
    imgUrl = URL.createObjectURL(file);
    const imgElement = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.src = imgUrl!;
      img.onload = () => {
        resolve(img);
      };
      img.onerror = (e) => reject(e);
    });

    onProgress?.(40);

    // 1. Try local GPU OCR Server (EasyOCR backend) if alive
    const isLocalServerActive = await checkLocalOcrServer();
    if (isLocalServerActive) {
      console.log('[OCR] 本地 GPU 识别服务已激活，正在进行 GPU 级 OCR 分析...');
      try {
        const formData = new FormData();
        formData.append('image', file);
        
        const res = await fetch('http://127.0.0.1:5000/ocr', {
          method: 'POST',
          body: formData
        });
        
        if (!res.ok) {
          throw new Error(`Local OCR server returned ${res.status}`);
        }
        
        const data = await res.json();
        const rawBlocks = data.blocks || [];
        
        const blocks: TranslationBlock[] = rawBlocks.map((block: any, index: number) => {
          // box format: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
          const box = block.box;
          const xcoords = box.map((pt: any) => pt[0]);
          const ycoords = box.map((pt: any) => pt[1]);
          
          const x0 = Math.min(...xcoords);
          const y0 = Math.min(...ycoords);
          const x1 = Math.max(...xcoords);
          const y1 = Math.max(...ycoords);
          
          const xmin = (x0 / dims.width) * 100;
          const ymin = (y0 / dims.height) * 100;
          const xmax = (x1 / dims.width) * 100;
          const ymax = (y1 / dims.height) * 100;
          
          const boxWidth = x1 - x0;
          const boxHeight = y1 - y0;
          const isVertical = boxHeight > boxWidth * 1.15;
          const fontHeightPx = isVertical ? boxWidth : boxHeight;
          const font_size_pct = (fontHeightPx / dims.height) * 100;
          
          // Sample colors from local image
          const { text_color, bg_color } = analyzeBlockColors(imgElement, ymin, xmin, ymax, xmax);
          
          return {
            id: `block_gpu_ocr_${Date.now()}_${index}`,
            original_text: block.text.trim(),
            translated_text: '',
            ymin: Math.min(Math.max(ymin, 0), 100),
            xmin: Math.min(Math.max(xmin, 0), 100),
            ymax: Math.min(Math.max(ymax, 0), 100),
            xmax: Math.min(Math.max(xmax, 0), 100),
            text_color,
            bg_color,
            font_size_pct: Math.max(1.2, font_size_pct)
          };
        });
        
        onProgress?.(100);
        return blocks;
      } catch (err) {
        console.warn('[OCR] 本地 GPU 识别服务请求失败，正在自动降级为浏览器 WASM 识别...', err);
      }
    }

    // 2. Fallback to local Tesseract.js WASM
    // Map sourceLang to Tesseract language codes
    let tesseractLang = 'eng+chi_sim';
    if (sourceLang === 'ja') {
      // Load both horizontal Japanese (jpn) and vertical Japanese (jpn_vert) along with English (eng)
      tesseractLang = 'jpn+jpn_vert+eng';
    } else if (sourceLang === 'zh') {
      tesseractLang = 'chi_sim+eng';
    } else if (sourceLang === 'en') {
      tesseractLang = 'eng';
    } else if (sourceLang === 'ko') {
      tesseractLang = 'kor+eng';
    } else if (sourceLang === 'auto') {
      tesseractLang = 'eng+chi_sim+jpn+jpn_vert';
    }

    // Run Tesseract OCR on the image
    const result = await Tesseract.recognize(
      file,
      tesseractLang,
      {
        logger: (m) => {
          if (m.status === 'recognizing text' && onProgress) {
            // Progress goes from 40% to 90%
            onProgress(40 + Math.round(m.progress * 50));
          }
        },
      }
    );

    const lines = (result.data as any).lines || [];
    const blocks: TranslationBlock[] = lines
      .filter((line: any) => line.text && line.text.trim().length > 0)
      .map((line: any, index: number) => {
        const { x0, y0, x1, y1 } = line.bbox;

        const xmin = (x0 / dims.width) * 100;
        const ymin = (y0 / dims.height) * 100;
        const xmax = (x1 / dims.width) * 100;
        const ymax = (y1 / dims.height) * 100;

        const boxWidth = x1 - x0;
        const boxHeight = y1 - y0;
        const isVertical = boxHeight > boxWidth * 1.15;
        const fontHeightPx = isVertical ? boxWidth : boxHeight;
        const font_size_pct = (fontHeightPx / dims.height) * 100;

        // Extract colors locally using the loaded image element
        const { text_color, bg_color } = analyzeBlockColors(imgElement, ymin, xmin, ymax, xmax);

        return {
          id: `block_ocr_${Date.now()}_${index}`,
          original_text: line.text.trim(),
          translated_text: '',
          ymin: Math.min(Math.max(ymin, 0), 100),
          xmin: Math.min(Math.max(xmin, 0), 100),
          ymax: Math.min(Math.max(ymax, 0), 100),
          xmax: Math.min(Math.max(xmax, 0), 100),
          text_color,
          bg_color,
          font_size_pct: Math.max(1.2, font_size_pct),
        };
      });

    onProgress?.(95);
    return blocks;
  } finally {
    if (imgUrl) {
      URL.revokeObjectURL(imgUrl);
    }
  }
};
