import React, { useState, useEffect, useRef } from 'react';
import type { ImageItem, TranslationBlock, StyleConfig } from '../types';
import { renderTranslatedCanvas, wrapText } from '../utils/canvasExporter';
import { Download, Edit3, Image as ImageIcon, Eye, RefreshCw } from 'lucide-react';

const getOptimizedFontSize = (
  text: string,
  boxWidthPct: number,
  boxHeightPct: number,
  originalWidth: number,
  originalHeight: number,
  renderedHeight: number,
  baseFontSize: number,
  style: StyleConfig
): number => {
  const initialFontSize = Math.max(8, baseFontSize * style.fontSizeScale);
  if (!style.autoFitFontSize) return initialFontSize;

  const refHeight = originalHeight || 1000;
  const refWidth = originalWidth || 1000;
  
  // Calculate scaled box bounds aligned with current rendered image size
  const scale = renderedHeight / refHeight;
  const w = ((boxWidthPct / 100) * refWidth) * scale;
  const h = ((boxHeightPct / 100) * refHeight) * scale;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return initialFontSize;

  let fontSize = initialFontSize;
  const isCJK = /[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/.test(text);
  const isVertical = h > w * 1.15 && isCJK;

  const buildFontStyle = (sz: number) => [
    style.fontItalic ? 'italic' : '',
    style.fontBold ? 'bold' : '',
    `${sz}px`,
    style.fontFamily || 'sans-serif'
  ].filter(Boolean).join(' ');

  const minFontSize = 6;

  if (isVertical) {
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

    const checkFitVertical = (fs: number) => {
      const paddingY = fs * 0.4;
      const maxColHeight = h - paddingY * 2;
      const colWidth = fs * 1.1;
      ctx.font = buildFontStyle(fs);
      const columns = wrapTextVertical(fs, text, maxColHeight);
      const totalTextWidth = columns.length * colWidth;
      return (totalTextWidth <= (w - fs * 0.4)) && !columns.some(c => c.length * fs * 1.1 > maxColHeight);
    };

    if (checkFitVertical(fontSize)) {
      return fontSize;
    }

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
    return bestSize;

  } else {
    const checkFitHorizontal = (fs: number) => {
      const padding = fs * 0.4;
      const maxTextWidth = w - padding * 2;
      ctx.font = buildFontStyle(fs);
      const lines = wrapText(ctx, text, maxTextWidth, isCJK);
      const lineHeight = fs * 1.2;
      const totalTextHeight = lines.length * lineHeight;
      return totalTextHeight <= (h - padding * 1.5);
    };

    if (checkFitHorizontal(fontSize)) {
      return fontSize;
    }

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
    return bestSize;
  }
};

interface ImageViewerProps {
  image: ImageItem;
  styleConfig: StyleConfig;
  onUpdateBlocks: (imageId: string, blocks: TranslationBlock[]) => void;
  onTranslateSingle: (imageId: string) => Promise<void>;
  isProcessing: boolean;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  image,
  styleConfig,
  onUpdateBlocks,
  onTranslateSingle,
  isProcessing
}) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'original'>('overlay');
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
  const [imgHeight, setImgHeight] = useState(500);
  const [isExporting, setIsExporting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  const wrapperRef = useRef<HTMLDivElement>(null);
  const blockRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

  const handleForceRender = () => {
    setIsRefreshing(true);
    if (image.blocks) {
      // Re-trigger blocks state to force a render refresh across all preview endpoints
      onUpdateBlocks(image.id, [...image.blocks]);
    }
    setTimeout(() => {
      setIsRefreshing(false);
    }, 450);
  };

  // ResizeObserver to calculate real image height for responsive font scaling
  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        // Find the img tag inside to get its actual rendered height
        const img = element.querySelector('img');
        if (img) {
          setImgHeight(img.clientHeight || 500);
        } else {
          setImgHeight(entry.contentRect.height || 500);
        }
      }
    });

    observer.observe(element);
    
    // Also trigger on image load
    const img = element.querySelector('img');
    const handleLoad = () => {
      if (img) setImgHeight(img.clientHeight || 500);
    };
    img?.addEventListener('load', handleLoad);

    return () => {
      observer.disconnect();
      img?.removeEventListener('load', handleLoad);
    };
  }, [image.id, viewMode]);

  const handleBlockTextChange = (blockId: string, newText: string) => {
    if (!image.blocks) return;
    const updated = image.blocks.map((b) =>
      b.id === blockId ? { ...b, translated_text: newText } : b
    );
    onUpdateBlocks(image.id, updated);
  };

  const handleBlockTypeChange = (blockId: string, newType: 'bubble' | 'onomatopoeia' | 'other') => {
    if (!image.blocks) return;
    const updated = image.blocks.map((b) =>
      b.id === blockId ? { ...b, type: newType } : b
    );
    onUpdateBlocks(image.id, updated);
  };

  const handleBlockColorChange = (blockId: string, type: 'text' | 'bg', value: string) => {
    if (!image.blocks) return;
    const updated = image.blocks.map((b) =>
      b.id === blockId ? (type === 'text' ? { ...b, text_color: value } : { ...b, bg_color: value }) : b
    );
    onUpdateBlocks(image.id, updated);
  };

  const selectBlock = (blockId: string) => {
    setSelectedBlockId(blockId);
    // Scroll editor item into view
    const ref = blockRefs.current[blockId];
    if (ref) {
      ref.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleDownload = async () => {
    if (!image.blocks) return;
    setIsExporting(true);
    try {
      const dataUrl = await renderTranslatedCanvas(image.previewUrl, image.blocks, styleConfig);
      const link = document.createElement('a');
      link.download = `translated_${image.name}`;
      link.href = dataUrl;
      link.click();
      setTimeout(() => {
        URL.revokeObjectURL(dataUrl);
      }, 100);
    } catch (err) {
      console.error('Failed to export canvas', err);
      alert('导出图片失败: ' + (err as Error).message);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="editor-workspace">
      {/* Left Pane: Image View */}
      <div className="viewer-pane">
        <div className="glass-card viewer-card">
          <div className="viewer-header">
            <h3 style={{ fontSize: '1rem', fontWeight: 600, maxWidth: '250px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {image.name}
            </h3>
            
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              <div className="viewer-tabs">
                <button
                  className={`viewer-tab ${viewMode === 'overlay' ? 'active' : ''}`}
                  onClick={() => setViewMode('overlay')}
                >
                  <Eye size={14} /> 译文视图
                </button>
                <button
                  className={`viewer-tab ${viewMode === 'original' ? 'active' : ''}`}
                  onClick={() => setViewMode('original')}
                >
                  <ImageIcon size={14} /> 原图
                </button>
              </div>

              {image.blocks && image.blocks.length > 0 && (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="btn btn-secondary"
                    onClick={handleForceRender}
                    disabled={isRefreshing || isExporting}
                    style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                    title="应用最新修改的字体样式或文本进行重绘"
                  >
                    <RefreshCw 
                      size={14} 
                      className={isRefreshing ? "pulse" : ""} 
                      style={{ 
                        animation: isRefreshing ? 'spin 1.5s linear infinite' : 'none',
                        marginRight: '4px' 
                      }} 
                    />
                    {isRefreshing ? '重绘中...' : '重绘排版'}
                  </button>
                  <button
                    className="btn btn-primary"
                    onClick={handleDownload}
                    disabled={isExporting || isRefreshing}
                    style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                  >
                    {isExporting ? <RefreshCw size={14} className="pulse" /> : <Download size={14} />}
                    {isExporting ? '导出中...' : '下载译图'}
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="image-canvas-container">
            <div
              ref={wrapperRef}
              className="workspace-image-wrapper"
              style={{ 
                position: 'relative', 
                display: 'inline-block',
                '--img-height': `${imgHeight}px`
              } as React.CSSProperties}
            >
              <img src={image.previewUrl} alt="Workspace Image" className="workspace-image" />
              
              {viewMode === 'overlay' && image.blocks && (
                <div className="translation-overlay-layer">
                  {/* Pass 1: Background Masks (Drawn at the bottom) */}
                  {image.blocks.map((block) => {
                    const w = block.xmax - block.xmin;
                    const h = block.ymax - block.ymin;
                    
                    // Skip background mask if it's onomatopoeia and mode is ignore or transparent
                    if (block.type === 'onomatopoeia' && (styleConfig.onomatopoeiaMode === 'ignore' || styleConfig.onomatopoeiaMode === 'transparent')) {
                      return null;
                    }
                    
                    const bgColor = styleConfig.bgColorMode === 'original'
                      ? block.bg_color
                      : styleConfig.bgColorMode === 'custom'
                        ? styleConfig.customBgColor
                        : 'transparent';

                    if (styleConfig.bgColorMode === 'none' || bgColor === 'transparent') {
                      return null;
                    }

                    const bgStyle: React.CSSProperties = {
                      position: 'absolute',
                      top: `${block.ymin}%`,
                      left: `${block.xmin}%`,
                      width: `${w}%`,
                      height: `${h}%`,
                      backgroundColor: bgColor,
                      opacity: styleConfig.bgOpacity / 100,
                      pointerEvents: 'none',
                    };

                    return (
                      <div
                        key={`bg-${block.id}`}
                        style={bgStyle}
                      />
                    );
                  })}

                  {/* Pass 2: Text Overlays (Drawn on top) */}
                  {image.blocks.map((block) => {
                    const w = block.xmax - block.xmin;
                    const h = block.ymax - block.ymin;
                    
                    // Skip text layer if it's onomatopoeia and mode is ignore
                    if (block.type === 'onomatopoeia' && styleConfig.onomatopoeiaMode === 'ignore') {
                      return null;
                    }
                    
                    // Style variables
                    const textColor = styleConfig.textColorMode === 'original' 
                      ? block.text_color 
                      : styleConfig.customTextColor;

                    const baseFontSize = (block.font_size_pct || 2.0) * imgHeight / 100;
                    const fontSize = getOptimizedFontSize(
                      block.translated_text,
                      w,
                      h,
                      image.width || 0,
                      image.height || 0,
                      imgHeight,
                      baseFontSize,
                      styleConfig
                    );

                    const isCJK = /[\u4e00-\u9fa5\u3040-\u30ff\uac00-\ud7af]/.test(block.translated_text);
                    const isVertical = h > w * 1.15 && isCJK;

                    const blockStyle: React.CSSProperties = {
                      top: `${block.ymin}%`,
                      left: `${block.xmin}%`,
                      width: `${w}%`,
                      height: `${h}%`,
                      color: textColor,
                      backgroundColor: 'transparent', // No background on text layer so it doesn't cover adjacent text
                      fontFamily: styleConfig.fontFamily,
                      fontSize: `${fontSize}px`,
                      fontWeight: styleConfig.fontBold ? 'bold' : 'normal',
                      fontStyle: styleConfig.fontItalic ? 'italic' : 'normal',
                      textShadow: styleConfig.textShadow ? '1px 1px 2px rgba(0,0,0,0.8)' : 'none',
                      WebkitTextStroke: styleConfig.textStroke ? `${styleConfig.strokeWidth}px ${styleConfig.strokeColor}` : 'none',
                      // Support vertical CJK rendering in live HTML preview
                      writingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
                      WebkitWritingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
                      flexDirection: isVertical ? 'row' : 'column',
                      justifyContent: 'center',
                      alignItems: 'center',
                    };

                    return (
                      <div
                        key={block.id}
                        className={`overlay-block ${selectedBlockId === block.id ? 'selected' : ''}`}
                        style={blockStyle}
                        onClick={() => selectBlock(block.id)}
                      >
                        <span 
                          className="overlay-text-span"
                          style={isVertical ? { width: 'auto', height: 'auto', display: 'block', lineHeight: 1.3 } : undefined}
                        >
                          {block.translated_text}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Right Pane: Blocks Editor */}
      <div className="viewer-pane">
        <div className="glass-card blocks-list-card">
          <div className="viewer-header">
            <h3 style={{ fontSize: '1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Edit3 size={16} /> 文字块细修与校对
            </h3>
            {(!image.blocks || image.blocks.length === 0) && (
              <button
                className="btn btn-primary"
                onClick={() => onTranslateSingle(image.id)}
                disabled={isProcessing}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
              >
                {isProcessing ? '翻译中...' : '单张翻译'}
              </button>
            )}
          </div>

          <div className="blocks-list-container" style={{ marginTop: '1rem' }}>
            {!image.blocks || image.blocks.length === 0 ? (
              <div className="empty-state" style={{ height: '100%', padding: '2rem 0' }}>
                <ImageIcon size={32} />
                <p style={{ fontSize: '0.9rem' }}>暂无文本框数据</p>
                <p style={{ fontSize: '0.8rem', maxWidth: '200px' }}>点击上方“单张翻译”或侧边栏“开始批量翻译”进行分析</p>
              </div>
            ) : (
              image.blocks.map((block, index) => (
                <div
                  key={block.id}
                  ref={(el) => { blockRefs.current[block.id] = el; }}
                  className={`block-editor-item ${selectedBlockId === block.id ? 'selected' : ''}`}
                  onClick={() => setSelectedBlockId(block.id)}
                >
                  <div className="block-editor-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="block-index"># {index + 1}</span>
                      <select
                        value={block.type || 'bubble'}
                        onChange={(e) => handleBlockTypeChange(block.id, e.target.value as any)}
                        style={{
                          fontSize: '0.75rem',
                          padding: '2px 4px',
                          borderRadius: '4px',
                          border: '1px solid var(--border-color, rgba(255,255,255,0.1))',
                          backgroundColor: 'rgba(0,0,0,0.3)',
                          color: 'var(--text-color, #fff)',
                          cursor: 'pointer',
                          outline: 'none'
                        }}
                      >
                        <option value="bubble">气泡内文字</option>
                        <option value="onomatopoeia">气泡外文字/拟声/旁注</option>
                      </select>
                    </div>
                    
                    <div className="block-colors">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>字</span>
                        <input
                          type="color"
                          value={block.text_color}
                          onChange={(e) => handleBlockColorChange(block.id, 'text', e.target.value)}
                          style={{
                            width: '18px',
                            height: '18px',
                            border: 'none',
                            cursor: 'pointer',
                            borderRadius: '3px',
                            background: 'none'
                          }}
                        />
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>底</span>
                        <input
                          type="color"
                          value={block.bg_color}
                          onChange={(e) => handleBlockColorChange(block.id, 'bg', e.target.value)}
                          style={{
                            width: '18px',
                            height: '18px',
                            border: 'none',
                            cursor: 'pointer',
                            borderRadius: '3px',
                            background: 'none'
                          }}
                        />
                      </div>
                    </div>
                  </div>

                  <p className="block-original-text">
                    {block.original_text || '(空白)'}
                  </p>

                  <textarea
                    className="block-translate-textarea"
                    placeholder="输入译文..."
                    value={block.translated_text}
                    onChange={(e) => handleBlockTextChange(block.id, e.target.value)}
                  />
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
