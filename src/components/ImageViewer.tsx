import React, { useState, useEffect, useRef } from 'react';
import type { ImageItem, TranslationBlock, StyleConfig } from '../types';
import { renderTranslatedCanvas, wrapText } from '../utils/canvasExporter';
import { 
  Download, 
  Edit3, 
  Image as ImageIcon, 
  Eye, 
  RefreshCw, 
  ChevronLeft, 
  ChevronRight, 
  ZoomIn, 
  ZoomOut, 
  Maximize, 
  Trash2, 
  Plus 
} from 'lucide-react';

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
  onNavigate?: (direction: 'prev' | 'next') => void;
  hasPrev?: boolean;
  hasNext?: boolean;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  image,
  styleConfig,
  onUpdateBlocks,
  onTranslateSingle,
  isProcessing,
  onNavigate,
  hasPrev,
  hasNext
}) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'original'>('overlay');
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
  const [imgHeight, setImgHeight] = useState(500);
  const [isExporting, setIsExporting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [zoomScale, setZoomScale] = useState(1.0);
  
  // Local blocks for fast editing/dragging feedback
  const [localBlocks, setLocalBlocks] = useState<TranslationBlock[]>([]);
  
  // Panning & Dragging States
  const [isPanning, setIsPanning] = useState(false);
  const [draggingBlockId, setDraggingBlockId] = useState<string | null>(null);
  const [resizingBlockId, setResizingBlockId] = useState<string | null>(null);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const blockRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});
  const dragStartRef = useRef<{ x: number; y: number; xmin: number; xmax: number; ymin: number; ymax: number } | null>(null);
  const panStartRef = useRef<{ x: number; y: number; scrollLeft: number; scrollTop: number } | null>(null);

  // Sync parent blocks to local blocks
  useEffect(() => {
    setLocalBlocks(image.blocks || []);
  }, [image.blocks, image.id]);

  const handleForceRender = () => {
    setIsRefreshing(true);
    if (localBlocks.length > 0) {
      onUpdateBlocks(image.id, [...localBlocks]);
    }
    setTimeout(() => {
      setIsRefreshing(false);
    }, 450);
  };

  // Smooth scroll target block into view on selection
  useEffect(() => {
    if (!selectedBlockId || localBlocks.length === 0) return;
    const block = localBlocks.find(b => b.id === selectedBlockId);
    if (!block || !canvasContainerRef.current) return;
    
    const container = canvasContainerRef.current;
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    
    const xPct = (block.xmin + block.xmax) / 2;
    const yPct = (block.ymin + block.ymax) / 2;
    
    const targetX = (xPct / 100) * wrapper.clientWidth - container.clientWidth / 2;
    const targetY = (yPct / 100) * wrapper.clientHeight - container.clientHeight / 2;
    
    container.scrollTo({
      left: Math.max(0, targetX),
      top: Math.max(0, targetY),
      behavior: 'smooth'
    });
  }, [selectedBlockId, image.id]);

  // ResizeObserver to calculate real image height for responsive font scaling
  useEffect(() => {
    const element = wrapperRef.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const img = element.querySelector('img');
        if (img) {
          setImgHeight(img.clientHeight || 500);
        } else {
          setImgHeight(entry.contentRect.height || 500);
        }
      }
    });

    observer.observe(element);
    
    const img = element.querySelector('img');
    const handleLoad = () => {
      if (img) setImgHeight(img.clientHeight || 500);
    };
    img?.addEventListener('load', handleLoad);

    return () => {
      observer.disconnect();
      img?.removeEventListener('load', handleLoad);
    };
  }, [image.id, viewMode, zoomScale]);

  // Mouse wheel zoom with Ctrl key
  useEffect(() => {
    const container = canvasContainerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey) {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.1 : -0.1;
        setZoomScale(prev => Math.min(3.0, Math.max(0.5, prev + delta)));
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, []);

  // Keyboard navigation shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      const isInputFocused = activeEl && (
        activeEl.tagName === 'INPUT' || 
        activeEl.tagName === 'TEXTAREA' || 
        activeEl.tagName === 'SELECT'
      );

      if (e.key === 'Escape') {
        setSelectedBlockId(null);
        if (isInputFocused && activeEl instanceof HTMLElement) {
          activeEl.blur();
        }
      }

      if (isInputFocused) return;

      if (e.key === '[' && onNavigate && hasPrev) {
        onNavigate('prev');
      } else if (e.key === ']' && onNavigate && hasNext) {
        onNavigate('next');
      } else if (e.key === 'ArrowUp' && localBlocks.length > 0) {
        e.preventDefault();
        const idx = selectedBlockId ? localBlocks.findIndex(b => b.id === selectedBlockId) : -1;
        if (idx > 0) {
          selectBlock(localBlocks[idx - 1].id);
        } else if (idx === -1) {
          selectBlock(localBlocks[0].id);
        }
      } else if (e.key === 'ArrowDown' && localBlocks.length > 0) {
        e.preventDefault();
        const idx = selectedBlockId ? localBlocks.findIndex(b => b.id === selectedBlockId) : -1;
        if (idx < localBlocks.length - 1) {
          selectBlock(localBlocks[idx + 1].id);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedBlockId, localBlocks, onNavigate, hasPrev, hasNext]);

  // Drag block and resize block event listeners
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragStartRef.current || !wrapperRef.current) return;
      const start = dragStartRef.current;
      const wrapper = wrapperRef.current;
      const wrapperRect = wrapper.getBoundingClientRect();
      
      const deltaX = ((e.clientX - start.x) / wrapperRect.width) * 100;
      const deltaY = ((e.clientY - start.y) / wrapperRect.height) * 100;
      
      if (draggingBlockId) {
        let newXmin = start.xmin + deltaX;
        let newXmax = start.xmax + deltaX;
        let newYmin = start.ymin + deltaY;
        let newYmax = start.ymax + deltaY;
        
        const w = start.xmax - start.xmin;
        const h = start.ymax - start.ymin;
        
        if (newXmin < 0) { newXmin = 0; newXmax = w; }
        if (newXmax > 100) { newXmax = 100; newXmin = 100 - w; }
        if (newYmin < 0) { newYmin = 0; newYmax = h; }
        if (newYmax > 100) { newYmax = 100; newYmin = 100 - h; }
        
        const updated = localBlocks.map(b => 
          b.id === draggingBlockId ? { ...b, xmin: newXmin, xmax: newXmax, ymin: newYmin, ymax: newYmax } : b
        );
        setLocalBlocks(updated);
      } else if (resizingBlockId) {
        let newXmax = Math.min(100, Math.max(start.xmin + 2, start.xmax + deltaX));
        let newYmax = Math.min(100, Math.max(start.ymin + 2, start.ymax + deltaY));
        
        const updated = localBlocks.map(b => 
          b.id === resizingBlockId ? { ...b, xmax: newXmax, ymax: newYmax } : b
        );
        setLocalBlocks(updated);
      }
    };
    
    const handleMouseUp = () => {
      if (draggingBlockId || resizingBlockId) {
        setDraggingBlockId(null);
        setResizingBlockId(null);
        dragStartRef.current = null;
        
        // Save the coordinates to parent state only after user stops dragging
        onUpdateBlocks(image.id, [...localBlocks]);
      }
    };
    
    if (draggingBlockId || resizingBlockId) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [draggingBlockId, resizingBlockId, localBlocks, image.id]);

  // Drag-to-pan event listener
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isPanning || !panStartRef.current || !canvasContainerRef.current) return;
      const start = panStartRef.current;
      const container = canvasContainerRef.current;
      
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      
      container.scrollLeft = start.scrollLeft - dx;
      container.scrollTop = start.scrollTop - dy;
    };

    const handleMouseUp = () => {
      setIsPanning(false);
      panStartRef.current = null;
    };

    if (isPanning) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isPanning]);

  // Mouse handlers inside components
  const handleDragStart = (e: React.MouseEvent, block: TranslationBlock, action: 'drag' | 'resize') => {
    e.stopPropagation();
    e.preventDefault();
    setSelectedBlockId(block.id);
    
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      xmin: block.xmin,
      xmax: block.xmax,
      ymin: block.ymin,
      ymax: block.ymax
    };
    
    if (action === 'drag') {
      setDraggingBlockId(block.id);
    } else {
      setResizingBlockId(block.id);
    }
  };

  const handleContainerMouseDown = (e: React.MouseEvent) => {
    if (e.target === canvasContainerRef.current || (e.target as HTMLElement).tagName === 'IMG') {
      if (draggingBlockId || resizingBlockId) return;
      
      setIsPanning(true);
      const container = canvasContainerRef.current;
      if (!container) return;
      panStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        scrollLeft: container.scrollLeft,
        scrollTop: container.scrollTop
      };
    }
  };

  const handleBlockTextChange = (blockId: string, newText: string) => {
    const updated = localBlocks.map((b) =>
      b.id === blockId ? { ...b, translated_text: newText } : b
    );
    setLocalBlocks(updated);
    onUpdateBlocks(image.id, updated);
  };

  const handleBlockTypeChange = (blockId: string, newType: 'bubble' | 'onomatopoeia') => {
    const updated = localBlocks.map((b) =>
      b.id === blockId ? { ...b, type: newType } : b
    );
    setLocalBlocks(updated);
    onUpdateBlocks(image.id, updated);
  };

  const handleBlockColorChange = (blockId: string, type: 'text' | 'bg', value: string) => {
    const updated = localBlocks.map((b) =>
      b.id === blockId ? (type === 'text' ? { ...b, text_color: value } : { ...b, bg_color: value }) : b
    );
    setLocalBlocks(updated);
    onUpdateBlocks(image.id, updated);
  };

  const handleDeleteBlock = (blockId: string) => {
    const updated = localBlocks.filter(b => b.id !== blockId);
    setLocalBlocks(updated);
    onUpdateBlocks(image.id, updated);
    if (selectedBlockId === blockId) {
      setSelectedBlockId(null);
    }
  };

  const handleCreateBlock = () => {
    const newBlock: TranslationBlock = {
      id: `block_${Date.now()}_${Math.random()}`,
      original_text: '',
      translated_text: '新文字框',
      xmin: 40,
      xmax: 60,
      ymin: 40,
      ymax: 48,
      text_color: '#FFFFFF',
      bg_color: '#000000',
      font_size_pct: 2.2,
      type: 'bubble'
    };
    
    const updated = [...localBlocks, newBlock];
    setLocalBlocks(updated);
    onUpdateBlocks(image.id, updated);
    setSelectedBlockId(newBlock.id);
    
    setTimeout(() => {
      const ref = blockRefs.current[newBlock.id];
      if (ref) {
        ref.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }, 100);
  };

  const selectBlock = (blockId: string) => {
    setSelectedBlockId(blockId);
    const ref = blockRefs.current[blockId];
    if (ref) {
      ref.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  const handleDownload = async () => {
    if (localBlocks.length === 0) return;
    setIsExporting(true);
    try {
      const dataUrl = await renderTranslatedCanvas(image.previewUrl, localBlocks, styleConfig);
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
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {onNavigate && (
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button
                    className="btn btn-secondary"
                    disabled={!hasPrev}
                    onClick={() => onNavigate('prev')}
                    style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem', height: '32px' }}
                    title="上一张 (快捷键 [)"
                  >
                    <ChevronLeft size={14} /> 上一张
                  </button>
                  <button
                    className="btn btn-secondary"
                    disabled={!hasNext}
                    onClick={() => onNavigate('next')}
                    style={{ padding: '0.4rem 0.6rem', fontSize: '0.8rem', height: '32px' }}
                    title="下一张 (快捷键 ])"
                  >
                    下一张 <ChevronRight size={14} />
                  </button>
                </div>
              )}
              <h3 style={{ 
                fontSize: '0.95rem', 
                fontWeight: 600, 
                maxWidth: '180px', 
                overflow: 'hidden', 
                textOverflow: 'ellipsis', 
                whiteSpace: 'nowrap',
                margin: 0,
                color: 'var(--text-main)'
              }}>
                {image.name}
              </h3>
            </div>
            
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

              {localBlocks.length > 0 && (
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

          <div 
            ref={canvasContainerRef} 
            className="image-canvas-container" 
            onMouseDown={handleContainerMouseDown}
            style={{ 
              position: 'relative',
              cursor: isPanning ? 'grabbing' : (zoomScale > 1.05 ? 'grab' : 'default')
            }}
          >
            <div
              ref={wrapperRef}
              className="workspace-image-wrapper"
              style={{ 
                position: 'relative', 
                display: 'inline-block',
                margin: 'auto',
                '--img-height': `${imgHeight}px`
              } as React.CSSProperties}
            >
              <img 
                src={image.previewUrl} 
                alt="Workspace Image" 
                className="workspace-image" 
                style={{ maxHeight: `${500 * zoomScale}px` }} 
                draggable={false}
              />
              
              {viewMode === 'overlay' && localBlocks.length > 0 && (
                <div className="translation-overlay-layer">
                  {/* Pass 1: Background Masks (Drawn at the bottom) */}
                  {localBlocks.map((block) => {
                    const w = block.xmax - block.xmin;
                    const h = block.ymax - block.ymin;
                    
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
                  {localBlocks.map((block) => {
                    const w = block.xmax - block.xmin;
                    const h = block.ymax - block.ymin;
                    
                    if (block.type === 'onomatopoeia' && styleConfig.onomatopoeiaMode === 'ignore') {
                      return null;
                    }
                    
                    const textColor = styleConfig.textColorMode === 'original' 
                      ? block.text_color 
                      : styleConfig.customTextColor;

                    const baseFontSize = (block.font_size_pct || 2.0) * (imgHeight || 500) / 100;
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
                      backgroundColor: 'transparent',
                      fontFamily: styleConfig.fontFamily,
                      fontSize: `${fontSize}px`,
                      fontWeight: styleConfig.fontBold ? 'bold' : 'normal',
                      fontStyle: styleConfig.fontItalic ? 'italic' : 'normal',
                      textShadow: styleConfig.textShadow ? '1px 1px 2px rgba(0,0,0,0.8)' : 'none',
                      WebkitTextStroke: styleConfig.textStroke ? `${styleConfig.strokeWidth}px ${styleConfig.strokeColor}` : 'none',
                      writingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
                      WebkitWritingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
                      flexDirection: isVertical ? 'row' : 'column',
                      justifyContent: 'center',
                      alignItems: 'center',
                      cursor: draggingBlockId === block.id ? 'grabbing' : 'grab',
                    };

                    return (
                      <div
                        key={block.id}
                        className={`overlay-block ${selectedBlockId === block.id ? 'selected' : ''}`}
                        style={blockStyle}
                        onMouseDown={(e) => handleDragStart(e, block, 'drag')}
                      >
                        <span 
                          className="overlay-text-span"
                          style={isVertical ? { width: 'auto', height: 'auto', display: 'block', lineHeight: 1.3 } : undefined}
                        >
                          {block.translated_text}
                        </span>

                        {/* Resize Handle (Bottom Right corner) */}
                        {selectedBlockId === block.id && (
                          <div
                            className="block-resize-handle"
                            onMouseDown={(e) => handleDragStart(e, block, 'resize')}
                            style={{
                              position: 'absolute',
                              bottom: '-4px',
                              right: '-4px',
                              width: '10px',
                              height: '10px',
                              backgroundColor: 'var(--color-primary, #6366f1)',
                              border: '1.5px solid #ffffff',
                              borderRadius: '50%',
                              cursor: 'nwse-resize',
                              zIndex: 50,
                            }}
                            title="拖动调整大小"
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Floating Zoom Controls */}
            <div style={{
              position: 'absolute',
              bottom: '1rem',
              right: '1rem',
              display: 'flex',
              gap: '0.25rem',
              background: 'rgba(11, 15, 26, 0.85)',
              border: '1px solid var(--border-color)',
              padding: '0.25rem',
              borderRadius: 'var(--radius-md)',
              backdropFilter: 'blur(10px)',
              zIndex: 50,
              boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
            }}>
              <button
                className="btn btn-secondary"
                onClick={() => setZoomScale(prev => Math.max(0.5, prev - 0.25))}
                style={{ padding: '0.35rem', minWidth: 'auto', border: 'none', display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                title="缩小 (Ctrl + 滚轮向下)"
              >
                <ZoomOut size={14} />
              </button>
              <span style={{ 
                fontSize: '0.75rem', 
                alignSelf: 'center', 
                padding: '0 0.5rem',
                minWidth: '45px',
                textAlign: 'center',
                color: 'var(--text-main)',
                fontWeight: 600
              }}>
                {Math.round(zoomScale * 100)}%
              </span>
              <button
                className="btn btn-secondary"
                onClick={() => setZoomScale(prev => Math.min(3.0, prev + 0.25))}
                style={{ padding: '0.35rem', minWidth: 'auto', border: 'none', display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                title="放大 (Ctrl + 滚轮向上)"
              >
                <ZoomIn size={14} />
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setZoomScale(1.0)}
                style={{ padding: '0.35rem', minWidth: 'auto', border: 'none', display: 'flex', alignItems: 'center', cursor: 'pointer' }}
                title="适应屏幕 (Reset)"
              >
                <Maximize size={14} />
              </button>
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
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className="btn btn-secondary"
                onClick={handleCreateBlock}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '4px' }}
                title="手动绘制并添加一个新的译文框"
              >
                <Plus size={12} /> 添加文本框
              </button>
              <button
                className="btn btn-primary"
                onClick={() => onTranslateSingle(image.id)}
                disabled={isProcessing}
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
              >
                {isProcessing ? '翻译中...' : (localBlocks.length > 0 ? '重新翻译' : '单张翻译')}
              </button>
            </div>
          </div>

          <div className="blocks-list-container" style={{ marginTop: '1rem' }}>
            {localBlocks.length === 0 ? (
              <div className="empty-state" style={{ height: '100%', padding: '2rem 0' }}>
                <ImageIcon size={32} />
                <p style={{ fontSize: '0.9rem' }}>暂无文本框数据</p>
                <p style={{ fontSize: '0.8rem', maxWidth: '200px' }}>点击上方“单张翻译”或侧边栏“开始批量翻译”进行分析</p>
              </div>
            ) : (
              localBlocks.map((block, index) => (
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
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
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

                      <button
                        className="image-card-delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm('确认删除该文本块吗？')) {
                            handleDeleteBlock(block.id);
                          }
                        }}
                        style={{
                          padding: '4px',
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          opacity: 0.6
                        }}
                        title="删除该文本块"
                      >
                        <Trash2 size={14} />
                      </button>
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
