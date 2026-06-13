import type { TranslationBlock, StyleConfig } from '../types';


// Wrap text utility for canvas drawing
export interface WrappedLine {
  text: string;
  width: number;
}

export const wrapText = (
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  isCJK: boolean
): WrappedLine[] => {
  const lines: WrappedLine[] = [];
  
  if (isCJK) {
    // For Chinese/Japanese/Korean, we can wrap at character level
    let currentLine = '';
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const testLine = currentLine + char;
      const metrics = ctx.measureText(testLine);
      
      if (metrics.width > maxWidth && currentLine.length > 0) {
        lines.push({ text: currentLine, width: ctx.measureText(currentLine).width });
        currentLine = char;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine.length > 0) {
      lines.push({ text: currentLine, width: ctx.measureText(currentLine).width });
    }
  } else {
    // For English and others, wrap at word level
    const words = text.split(/\s+/);
    let currentLine = '';
    
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      const metrics = ctx.measureText(testLine);
      
      if (metrics.width > maxWidth && currentLine.length > 0) {
        lines.push({ text: currentLine, width: ctx.measureText(currentLine).width });
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine.length > 0) {
      lines.push({ text: currentLine, width: ctx.measureText(currentLine).width });
    }
  }
  
  return lines;
};

// Main function to draw image and translation overlays to a Canvas
export const renderTranslatedCanvas = async (
  originalImageSrc: string,
  blocks: TranslationBlock[],
  style: StyleConfig
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous'; // Avoid tainted canvas
    img.src = originalImageSrc;
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Failed to get 2D context'));
        return;
      }
      
      // 1. Draw original image
      ctx.drawImage(img, 0, 0);
      
      // 2a. First pass: Draw all background masks
      blocks.forEach((block) => {
        const xmin = (block.xmin / 100) * canvas.width;
        const ymin = (block.ymin / 100) * canvas.height;
        const xmax = (block.xmax / 100) * canvas.width;
        const ymax = (block.ymax / 100) * canvas.height;
        
        const w = xmax - xmin;
        const h = ymax - ymin;
        
        // Skip invalid blocks
        if (w <= 0 || h <= 0) return;
        
        // Check onomatopoeia mode for background mask
        if (block.type === 'onomatopoeia' && (style.onomatopoeiaMode === 'ignore' || style.onomatopoeiaMode === 'transparent')) {
          return;
        }
        
        // 2a. Draw Background Mask
        let bgColor = '#000000';
        let useBg = true;
        
        if (style.bgColorMode === 'original') {
          bgColor = block.bg_color;
        } else if (style.bgColorMode === 'custom') {
          bgColor = style.customBgColor;
        } else {
          useBg = false;
        }
        
        if (useBg) {
          ctx.save();
          ctx.fillStyle = bgColor;
          // Apply opacity
          ctx.globalAlpha = style.bgOpacity / 100;
          ctx.fillRect(xmin, ymin, w, h);
          ctx.restore();
        }
      });
      
      // 2b. Second pass: Draw all text overlays
      blocks.forEach((block) => {
        const xmin = (block.xmin / 100) * canvas.width;
        const ymin = (block.ymin / 100) * canvas.height;
        const xmax = (block.xmax / 100) * canvas.width;
        const ymax = (block.ymax / 100) * canvas.height;
        
        const w = xmax - xmin;
        const h = ymax - ymin;
        
        // Skip invalid blocks
        if (w <= 0 || h <= 0) return;
        
        // Check onomatopoeia mode for text overlay
        if (block.type === 'onomatopoeia' && style.onomatopoeiaMode === 'ignore') {
          return;
        }
        
        // 2b. Draw Text
        ctx.save();
        
        // Define text color
        let textColor = '#FFFFFF';
        if (style.textColorMode === 'original') {
          textColor = block.text_color;
        } else {
          textColor = style.customTextColor;
        }
        
        // Font size calculation (based on height percentage)
        const baseFontSize = (block.font_size_pct || 2.0) * canvas.height / 100;
        let fontSize = Math.max(8, baseFontSize * style.fontSizeScale);
        
        const isCJK = /[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/.test(block.translated_text);
        const isVertical = h > w * 1.15 && isCJK;
        
        // Define font styling helper
        const buildFontStyle = (sz: number) => [
          style.fontItalic ? 'italic' : '',
          style.fontBold ? 'bold' : '',
          `${sz}px`,
          style.fontFamily || 'sans-serif'
        ].filter(Boolean).join(' ');

        // Setup drawing styles
        ctx.fillStyle = textColor;
        
        if (style.textShadow) {
          ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
          ctx.shadowBlur = 4;
          ctx.shadowOffsetX = 2;
          ctx.shadowOffsetY = 2;
        } else {
          ctx.shadowColor = 'transparent';
          ctx.shadowBlur = 0;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;
        }
        
        if (style.textStroke) {
          ctx.strokeStyle = style.strokeColor || '#000000';
          ctx.lineWidth = style.strokeWidth || 2;
          ctx.lineJoin = 'round';
        }

        if (isVertical) {
          // --- Vertical text layout ---
          // Helper for vertical wrapping
          const wrapTextVertical = (fs: number, textStr: string, maxH: number): string[] => {
            const cols: string[] = [];
            let current = '';
            const charH = fs * 1.1;
            for (let i = 0; i < textStr.length; i++) {
              const char = textStr[i];
              const testH = (current.length + 1) * charH;
              if (testH > maxH && current.length > 0) {
                cols.push(current);
                current = char;
              } else {
                current += char;
              }
            }
            if (current.length > 0) cols.push(current);
            return cols;
          };

          // Vertical shrink-to-fit using binary search
          if (style.autoFitFontSize) {
            const minFontSize = 7;
            const checkFitVertical = (fs: number) => {
              const pY = fs * 0.4;
              const maxH = h - pY * 2;
              const cW = fs * 1.1;
              const cols = wrapTextVertical(fs, block.translated_text, maxH);
              const totW = cols.length * cW;
              return (totW <= (w - fs * 0.4)) && !cols.some(c => c.length * fs * 1.1 > maxH);
            };

            if (!checkFitVertical(fontSize)) {
              let low = minFontSize;
              let high = fontSize;
              let bestSize = minFontSize;

              for (let i = 0; i < 10; i++) {
                const mid = (low + high) / 2;
                if (checkFitVertical(mid)) {
                  bestSize = mid;
                  low = mid;
                } else {
                  high = mid;
                }
              }
              fontSize = bestSize;
            }
          }

          let paddingY = fontSize * 0.4;
          let maxColHeight = h - paddingY * 2;
          let colWidth = fontSize * 1.1;
          ctx.font = buildFontStyle(fontSize);
          let columns = wrapTextVertical(fontSize, block.translated_text, maxColHeight);
          let totalTextWidth = columns.length * colWidth;

          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          // CJK vertical lines go from right to left
          const startX = xmin + (w / 2) + (totalTextWidth / 2) - (colWidth / 2);
          
          columns.forEach((colText, colIdx) => {
            const x = startX - colIdx * colWidth;
            const colHeight = colText.length * fontSize * 1.1;
            // Center characters vertically in column
            const startY = ymin + (h / 2) - (colHeight / 2) + (fontSize * 1.1 / 2);
            
            for (let charIdx = 0; charIdx < colText.length; charIdx++) {
              const char = colText[charIdx];
              const y = startY + charIdx * fontSize * 1.1;
              
              if (style.textStroke) {
                ctx.strokeText(char, x, y);
              }
              ctx.fillText(char, x, y);
            }
          });

        } else {
          // --- Horizontal text layout (default) ---
          // Horizontal shrink-to-fit using binary search
          if (style.autoFitFontSize) {
            const minFontSize = 7;
            const checkFitHorizontal = (fs: number) => {
              const p = fs * 0.4;
              const maxW = w - p * 2;
              ctx.font = buildFontStyle(fs);
              const lns = wrapText(ctx, block.translated_text, maxW, isCJK);
              const lH = fs * 1.2;
              const totH = lns.length * lH;
              return totH <= (h - p * 1.5);
            };

            if (!checkFitHorizontal(fontSize)) {
              let low = minFontSize;
              let high = fontSize;
              let bestSize = minFontSize;

              for (let i = 0; i < 10; i++) {
                const mid = (low + high) / 2;
                if (checkFitHorizontal(mid)) {
                  bestSize = mid;
                  low = mid;
                } else {
                  high = mid;
                }
              }
              fontSize = bestSize;
            }
          }

          let padding = fontSize * 0.4;
          let maxTextWidth = w - padding * 2;
          ctx.font = buildFontStyle(fontSize);
          let lines = wrapText(ctx, block.translated_text, maxTextWidth, isCJK);
          let lineHeight = fontSize * 1.2;
          let totalTextHeight = lines.length * lineHeight;
          
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          
          // Center text vertically in the bounding box
          let startY = ymin + (h / 2) - (totalTextHeight / 2) + (lineHeight / 2);
          const centerX = xmin + (w / 2);
          
          lines.forEach((line) => {
            if (startY > ymin + h - padding) return;
            
            if (style.textStroke) {
              ctx.strokeText(line.text, centerX, startY);
            }
            ctx.fillText(line.text, centerX, startY);
            startY += lineHeight;
          });
        }
        
        ctx.restore();
      });
      
      try {
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error('Canvas export failed'));
            return;
          }
          const url = URL.createObjectURL(blob);
          resolve(url);
        }, 'image/png');
      } catch (err) {
        reject(err);
      }
    };
    
    img.onerror = (err) => {
      reject(err);
    };
  });
};

export const renderErasedCanvas = async (
  originalImageSrc: string,
  blocks: TranslationBlock[],
  style: StyleConfig
): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = originalImageSrc;
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Failed to get 2D context'));
        return;
      }
      
      // 1. Draw original image
      ctx.drawImage(img, 0, 0);
      
      // 2. Draw all background masks (erasing the text)
      blocks.forEach((block) => {
        const xmin = (block.xmin / 100) * canvas.width;
        const ymin = (block.ymin / 100) * canvas.height;
        const xmax = (block.xmax / 100) * canvas.width;
        const ymax = (block.ymax / 100) * canvas.height;
        
        const w = xmax - xmin;
        const h = ymax - ymin;
        
        if (w <= 0 || h <= 0) return;
        
        // Check onomatopoeia mode for background mask
        if (block.type === 'onomatopoeia' && (style.onomatopoeiaMode === 'ignore' || style.onomatopoeiaMode === 'transparent')) {
          return;
        }
        
        let bgColor = '#000000';
        let useBg = true;
        
        if (style.bgColorMode === 'original') {
          bgColor = block.bg_color;
        } else if (style.bgColorMode === 'custom') {
          bgColor = style.customBgColor;
        } else {
          useBg = false;
        }
        
        if (useBg) {
          ctx.save();
          ctx.fillStyle = bgColor;
          ctx.globalAlpha = style.bgOpacity / 100;
          ctx.fillRect(xmin, ymin, w, h);
          ctx.restore();
        }
      });
      
      try {
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error('Canvas export failed'));
            return;
          }
          resolve(blob);
        }, 'image/png');
      } catch (err) {
        reject(err);
      }
    };
    
    img.onerror = (err) => {
      reject(err);
    };
  });
};
