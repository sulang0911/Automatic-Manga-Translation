import { useState, useEffect, useRef } from 'react';
import type { TranslateConfig, StyleConfig, ImageItem, TranslationBlock } from './types';
import { SettingsPanel } from './components/SettingsPanel';
import { UploadArea } from './components/UploadArea';
import { ImageGrid } from './components/ImageGrid';
import { ImageViewer } from './components/ImageViewer';
import { ToastContainer } from './components/Toast';
import type { ToastMessage } from './components/Toast';
import { translateImage, getImageDimensions, checkIfImageIsSolidColor } from './utils/translator';
import { renderTranslatedCanvas, renderErasedCanvas } from './utils/canvasExporter';
import JSZip from 'jszip';
import { 
  Sparkles, 
  Layers, 
  Trash2, 
  Play, 
  Pause, 
  Download, 
  ChevronLeft, 
  BarChart3, 
  CheckCircle, 
  AlertCircle,
  Clock,
  Grid,
  List,
  Settings,
  Menu
} from 'lucide-react';


const LOCAL_STORAGE_KEY_CONFIG = 'img_trans_config';
const LOCAL_STORAGE_KEY_STYLE = 'img_trans_style';

const DEFAULT_CONFIG: TranslateConfig = {
  provider: 'gemini',
  apiKey: '',
  model: 'gemini-2.5-flash',
  customEndpoint: '',
  targetLang: '简体中文',
  sourceLang: 'auto'
};

const DEFAULT_STYLE: StyleConfig = {
  fontFamily: 'Outfit',
  fontSizeScale: 1.0,
  textColorMode: 'original',
  customTextColor: '#FFFFFF',
  bgColorMode: 'original',
  customBgColor: '#000000',
  bgOpacity: 90,
  textShadow: true,
  textStroke: false,
  strokeColor: '#000000',
  strokeWidth: 2,
  fontBold: true,
  fontItalic: false,
  autoFitFontSize: true,
  onomatopoeiaMode: 'ignore'
};

const getRenderHash = (blocks: TranslationBlock[], style: StyleConfig): string => {
  return `${style.fontFamily}-${style.fontSizeScale}-${style.textColorMode}-${style.customTextColor}-${style.bgColorMode}-${style.customBgColor}-${style.bgOpacity}-${style.textShadow}-${style.textStroke}-${style.strokeColor}-${style.strokeWidth}-${style.fontBold}-${style.fontItalic}-${style.autoFitFontSize}-${style.onomatopoeiaMode}-${blocks.map(b => `${b.translated_text}-${b.text_color}-${b.bg_color}-${b.type || 'bubble'}`).join(',')}`;
};

export default function App() {
  const [config, setConfig] = useState<TranslateConfig>(() => {
    const saved = localStorage.getItem(LOCAL_STORAGE_KEY_CONFIG);
    return saved ? JSON.parse(saved) : DEFAULT_CONFIG;
  });

  const [styleConfig, setStyleConfig] = useState<StyleConfig>(() => {
    const saved = localStorage.getItem(LOCAL_STORAGE_KEY_STYLE);
    return saved ? JSON.parse(saved) : DEFAULT_STYLE;
  });

  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [isTranslatingAll, setIsTranslatingAll] = useState(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [directoryHandle, setDirectoryHandle] = useState<FileSystemDirectoryHandle | null>(null);
  const [queueViewMode, setQueueViewMode] = useState<'grid' | 'list'>('list');
  const [showSettings, setShowSettings] = useState(true);
  const [showSidebar, setShowSidebar] = useState(() => {
    return window.innerWidth > 900;
  });
  
  const cancelRef = useRef(false);

  // Auto-save configs
  useEffect(() => {
    localStorage.setItem(LOCAL_STORAGE_KEY_CONFIG, JSON.stringify(config));
  }, [config]);

  useEffect(() => {
    localStorage.setItem(LOCAL_STORAGE_KEY_STYLE, JSON.stringify(styleConfig));
  }, [styleConfig]);

  // Background preloader for Grid view to load image details in small non-blocking batches
  useEffect(() => {
    if (queueViewMode !== 'grid') return;

    let active = true;
    const loadUnresolved = async () => {
      const unresolved = images.filter(img => img.fileHandle && !img.previewUrl);
      if (unresolved.length === 0) return;

      const batchSize = 5;
      for (let i = 0; i < unresolved.length; i += batchSize) {
        if (!active) break;
        const batch = unresolved.slice(i, i + batchSize);
        
        const resolvedBatch = await Promise.all(
          batch.map(img => resolveImageFiles(img))
        );

        if (!active) break;

        setImages(prev => {
          const map = new Map(resolvedBatch.map(r => [r.id, r]));
          return prev.map(img => map.has(img.id) ? map.get(img.id)! : img);
        });

        await new Promise(resolve => setTimeout(resolve, 100));
      }
    };

    loadUnresolved();

    return () => {
      active = false;
    };
  }, [queueViewMode, images.length]);

  // Update translated previews when style changes or blocks text changes in real-time
  useEffect(() => {
    const updatePreviews = async () => {
      let updated = false;
      const newImages = await Promise.all(images.map(async (img) => {
        if (img.status === 'completed' && img.previewUrl && img.blocks && img.blocks.length > 0) {
          const currentHash = getRenderHash(img.blocks, styleConfig);
          if (img.lastRenderedHash !== currentHash) {
            try {
              if (img.translatedPreviewUrl) {
                URL.revokeObjectURL(img.translatedPreviewUrl);
              }
              const url = await renderTranslatedCanvas(img.previewUrl, img.blocks, styleConfig);
              updated = true;
              return { ...img, translatedPreviewUrl: url, lastRenderedHash: currentHash };
            } catch (e) {
              console.error('Failed to update preview thumbnail', e);
            }
          }
        }
        return img;
      }));
      if (updated) {
        setImages(newImages);
      }
    };
    updatePreviews();
  }, [
    styleConfig, 
    images.map(img => `${img.id}-${img.status}-${img.previewUrl ? 'has_orig' : 'no_orig'}-${img.translatedPreviewUrl ? 'has_preview' : 'no_preview'}-${img.blocks?.map(b => b.translated_text + b.text_color + b.bg_color).join('') || ''}`).join(',')
  ]);

  const addToast = (text: string, type: ToastMessage['type'] = 'info') => {
    const newToast: ToastMessage = {
      id: `toast_${Date.now()}_${Math.random()}`,
      text,
      type
    };
    setToasts(prev => [...prev, newToast]);
  };

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  const checkFileCache = async (
    fileName: string,
    dirHandle: FileSystemDirectoryHandle
  ): Promise<{
    blocks?: TranslationBlock[];
    hasOcrCache: boolean;
    hasErasedCache: boolean;
  }> => {
    let blocks: TranslationBlock[] = [];
    let hasOcrCache = false;
    let hasErasedCache = false;

    let cacheDirHandle: FileSystemDirectoryHandle | null = null;
    try {
      cacheDirHandle = await dirHandle.getDirectoryHandle('translation_cache', { create: false });
    } catch (e) {
      // translation_cache folder does not exist
    }

    // Try loading JSON blocks cache
    try {
      let jsonHandle;
      if (cacheDirHandle) {
        try {
          jsonHandle = await cacheDirHandle.getFileHandle(`${fileName}_blocks.json`);
        } catch (err) {
          // Fallback to root directory
          jsonHandle = await dirHandle.getFileHandle(`${fileName}_blocks.json`);
        }
      } else {
        jsonHandle = await dirHandle.getFileHandle(`${fileName}_blocks.json`);
      }
      const jsonFile = await jsonHandle.getFile();
      const text = await jsonFile.text();
      blocks = JSON.parse(text);
      hasOcrCache = true;
    } catch (e) {
      // No JSON cache
    }

    // Try checking if translated or erased preview image cache exists (WITHOUT reading file data)
    try {
      if (cacheDirHandle) {
        try {
          await cacheDirHandle.getFileHandle(`${fileName}_translated.png`);
          hasErasedCache = true;
        } catch (err) {
          try {
            await dirHandle.getFileHandle(`${fileName}_translated.png`);
            hasErasedCache = true;
          } catch (err2) {
            try {
              await cacheDirHandle.getFileHandle(`${fileName}_erased.png`);
              hasErasedCache = true;
            } catch (err3) {
              await dirHandle.getFileHandle(`${fileName}_erased.png`);
              hasErasedCache = true;
            }
          }
        }
      } else {
        try {
          await dirHandle.getFileHandle(`${fileName}_translated.png`);
          hasErasedCache = true;
        } catch (err) {
          await dirHandle.getFileHandle(`${fileName}_erased.png`);
          hasErasedCache = true;
        }
      }
    } catch (e) {
      // No image cache
    }

    return {
      blocks: hasOcrCache ? blocks : undefined,
      hasOcrCache,
      hasErasedCache
    };
  };

  const verifyWritePermission = async (handle: FileSystemDirectoryHandle): Promise<boolean> => {
    // @ts-ignore
    if (typeof handle.queryPermission === 'function') {
      // @ts-ignore
      const status = await handle.queryPermission({ mode: 'readwrite' });
      if (status === 'granted') {
        return true;
      }
    }
    // @ts-ignore
    if (typeof handle.requestPermission === 'function') {
      // @ts-ignore
      const status = await handle.requestPermission({ mode: 'readwrite' });
      return status === 'granted';
    }
    return true; // Fallback
  };

  const resolveImageFiles = async (img: ImageItem): Promise<ImageItem> => {
    if (img.file && img.previewUrl) {
      return img;
    }

    if (img.fileHandle) {
      try {
        const file = await img.fileHandle.getFile();
        const previewUrl = URL.createObjectURL(file);
        
        let translatedPreviewUrl = img.translatedPreviewUrl;
        if (!translatedPreviewUrl && img.status === 'completed' && directoryHandle) {
          try {
            const cacheDirHandle = await directoryHandle.getDirectoryHandle('translation_cache', { create: false });
            let translatedHandle;
            try {
              translatedHandle = await cacheDirHandle.getFileHandle(`${img.name}_translated.png`);
            } catch (err) {
              translatedHandle = await directoryHandle.getFileHandle(`${img.name}_translated.png`);
            }
            const translatedFile = await translatedHandle.getFile();
            translatedPreviewUrl = URL.createObjectURL(translatedFile);
          } catch (e) {
            try {
              const cacheDirHandle = await directoryHandle.getDirectoryHandle('translation_cache', { create: false });
              let erasedHandle;
              try {
                erasedHandle = await cacheDirHandle.getFileHandle(`${img.name}_erased.png`);
              } catch (err) {
                erasedHandle = await directoryHandle.getFileHandle(`${img.name}_erased.png`);
              }
              const erasedFile = await erasedHandle.getFile();
              translatedPreviewUrl = URL.createObjectURL(erasedFile);
            } catch (e2) {
              // No cache image
            }
          }
        }

        return {
          ...img,
          file,
          previewUrl,
          translatedPreviewUrl
        };
      } catch (err) {
        console.error(`Failed to resolve files for ${img.name}`, err);
      }
    }

    return img;
  };

  const handleFilesSelected = async (files: FileList) => {
    const newItems: ImageItem[] = [];
    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));

    if (imageFiles.length === 0) {
      addToast('请选择有效的图片文件！', 'error');
      return;
    }

    // Sort naturally by name (1, 2, ..., 10)
    const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
    imageFiles.sort((a, b) => collator.compare(a.name, b.name));

    addToast(`正在导入 ${imageFiles.length} 张图片...`, 'info');

    for (const file of imageFiles) {
      let cache: {
        blocks?: TranslationBlock[];
        hasOcrCache: boolean;
        hasErasedCache: boolean;
      } = {
        hasOcrCache: false,
        hasErasedCache: false
      };

      if (directoryHandle) {
        try {
          cache = await checkFileCache(file.name, directoryHandle);
        } catch (e) {
          console.error(`Failed to check cache for manually selected file: ${file.name}`, e);
        }
      }

      newItems.push({
        id: `img_${Date.now()}_${Math.random()}`,
        name: file.name,
        file,
        previewUrl: URL.createObjectURL(file),
        translatedPreviewUrl: undefined,
        status: cache.hasOcrCache ? 'completed' : 'idle',
        progress: cache.hasOcrCache ? 100 : 0,
        blocks: cache.blocks,
        hasOcrCache: cache.hasOcrCache,
        hasErasedCache: cache.hasErasedCache
      });
    }

    setImages(prev => [...prev, ...newItems]);
    addToast('导入完成', 'success');
  };

  const handleOpenLocalDirectory = async () => {
    try {
      // @ts-ignore
      const handle = await window.showDirectoryPicker();
      
      // Request write permission under user gesture
      const hasPermission = await verifyWritePermission(handle);
      if (!hasPermission) {
        addToast('未获得文件夹的写入权限，翻译缓存将无法写入本地！', 'error');
      }
      
      setDirectoryHandle(handle);
      addToast('正在扫描工作文件夹中的图片与缓存...', 'info');
      
      const entries: FileSystemFileHandle[] = [];
      
      for await (const entry of handle.values()) {
        if (entry.kind === 'file' && /\.(jpe?g|png|webp|bmp)$/i.test(entry.name)) {
          // Exclude generated helper files
          if (entry.name.endsWith('_erased.png') || entry.name.endsWith('_translated.png') || entry.name.endsWith('_blocks.json')) {
            continue;
          }
          entries.push(entry as FileSystemFileHandle);
        }
      }

      // Sort naturally by name (1, 2, ..., 10)
      const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
      entries.sort((a, b) => collator.compare(a.name, b.name));

      const newItems: ImageItem[] = [];
      for (const entry of entries) {
        const cache = await checkFileCache(entry.name, handle);
        
        newItems.push({
          id: `img_${Date.now()}_${Math.random()}`,
          name: entry.name,
          file: undefined,
          previewUrl: '',
          translatedPreviewUrl: undefined,
          status: cache.hasOcrCache ? 'completed' : 'idle',
          progress: cache.hasOcrCache ? 100 : 0,
          blocks: cache.blocks,
          hasOcrCache: cache.hasOcrCache,
          hasErasedCache: cache.hasErasedCache,
          fileHandle: entry
        });
      }
      
      setImages(prev => {
        prev.forEach(img => {
          if (img.previewUrl) URL.revokeObjectURL(img.previewUrl);
          if (img.translatedPreviewUrl) URL.revokeObjectURL(img.translatedPreviewUrl);
        });
        return newItems;
      });
      setSelectedImageId(null);
      addToast(`导入成功！共载入 ${newItems.length} 张图片，其中 ${newItems.filter(i => i.hasOcrCache).length} 张已自动载入本地翻译缓存。`, 'success');
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        console.error(err);
        addToast(`无法读取本地文件夹: ${(err as Error).message}`, 'error');
      }
    }
  };

  const handleSelectImage = async (id: string) => {
    const target = images.find(img => img.id === id);
    if (!target) return;
    
    const resolved = await resolveImageFiles(target);
    setImages(prev => prev.map(img => img.id === id ? resolved : img));
    setSelectedImageId(id);
  };

  const handleRemoveImage = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setImages(prev => {
      const target = prev.find(item => item.id === id);
      if (target) {
        if (target.previewUrl) {
          URL.revokeObjectURL(target.previewUrl);
        }
        if (target.translatedPreviewUrl) {
          URL.revokeObjectURL(target.translatedPreviewUrl);
        }
      }
      return prev.filter(item => item.id !== id);
    });

    if (selectedImageId === id) {
      setSelectedImageId(null);
    }
  };

  const handleClearQueue = () => {
    images.forEach(img => {
      if (img.previewUrl) {
        URL.revokeObjectURL(img.previewUrl);
      }
      if (img.translatedPreviewUrl) {
        URL.revokeObjectURL(img.translatedPreviewUrl);
      }
    });
    setImages([]);
    setSelectedImageId(null);
    addToast('已清空图片队列', 'info');
  };

  const handleUpdateBlocks = async (imageId: string, blocks: TranslationBlock[]) => {
    setImages(prev => prev.map(img => {
      if (img.id === imageId) {
        if (img.translatedPreviewUrl) {
          URL.revokeObjectURL(img.translatedPreviewUrl);
        }
        return { ...img, blocks, translatedPreviewUrl: undefined };
      }
      return img;
    }));

    const item = images.find(img => img.id === imageId);
    if (item) {
      try {
        const url = await renderTranslatedCanvas(item.previewUrl, blocks, styleConfig);
        
        if (directoryHandle) {
          (async () => {
            try {
              const hasPermission = await verifyWritePermission(directoryHandle);
              if (hasPermission) {
                const cacheDirHandle = await directoryHandle.getDirectoryHandle('translation_cache', { create: true });

                // Update JSON blocks cache
                const jsonHandle = await cacheDirHandle.getFileHandle(`${item.name}_blocks.json`, { create: true });
                const jsonWritable = await jsonHandle.createWritable();
                await jsonWritable.write(JSON.stringify(blocks));
                await jsonWritable.close();

                // Update translated image cache
                const res = await fetch(url);
                const blob = await res.blob();
                const translatedHandle = await cacheDirHandle.getFileHandle(`${item.name}_translated.png`, { create: true });
                const translatedWritable = await translatedHandle.createWritable();
                await translatedWritable.write(blob);
                await translatedWritable.close();
              }
            } catch (fsErr) {
              console.error('Failed to update local cache files on block edit in background', fsErr);
            }
          })();
        }

        const currentHash = getRenderHash(blocks, styleConfig);
        setImages(prev => prev.map(img => img.id === imageId ? { 
          ...img, 
          translatedPreviewUrl: url, 
          lastRenderedHash: currentHash,
          hasOcrCache: true
        } : img));
      } catch (e) {
        console.error('Failed to update blocks preview', e);
      }
    }
  };

  const handleTranslateSingle = async (imageId: string): Promise<void> => {
    let item = images.find(img => img.id === imageId);
    if (!item) return;

    if (!config.apiKey && config.provider !== 'custom') {
      addToast('请先在右侧面板配置大模型 API Key', 'error');
      return;
    }

    if (directoryHandle) {
      const hasPermission = await verifyWritePermission(directoryHandle);
      if (!hasPermission) {
        addToast('未获得文件夹的写入权限，翻译将不会同步到本地磁盘。', 'error');
      }
    }

    // Ensure files are resolved
    const resolvedItem = await resolveImageFiles(item);
    item = resolvedItem;
    setImages(prev => prev.map(img => img.id === imageId ? { ...resolvedItem, status: 'processing', progress: 10 } : img));

    try {
      let dims = { width: item.width || 0, height: item.height || 0 };
      if (!item.width || !item.height) {
        if (item.file) {
          dims = await getImageDimensions(item.file);
        }
      }

      // Check if image is solid color
      const isSolid = await checkIfImageIsSolidColor(item.previewUrl);
      if (isSolid) {
        setImages(prev => prev.map(img => img.id === imageId ? {
          ...img,
          status: 'completed',
          progress: 100,
          blocks: [],
          translatedPreviewUrl: img.previewUrl,
          width: dims.width,
          height: dims.height,
          hasOcrCache: false,
          hasErasedCache: false
        } : img));
        addToast(`图片 "${item.name}" 为纯色（空白）图片，已直接标记为成功。`, 'success');
        return;
      }

      const blocks = await translateImage(item, config, (progress) => {
        setImages(prev => prev.map(img => img.id === imageId ? { ...img, progress } : img));
      });

      // Render and cache translated preview image
      let translatedPreviewUrl: string | undefined = undefined;
      let lastRenderedHash: string | undefined = undefined;
      let hasOcrCache = false;
      let hasErasedCache = false;

      if (blocks.length > 0) {
        try {
          if (item.translatedPreviewUrl) {
            URL.revokeObjectURL(item.translatedPreviewUrl);
          }
          translatedPreviewUrl = await renderTranslatedCanvas(item.previewUrl, blocks, styleConfig);
          lastRenderedHash = getRenderHash(blocks, styleConfig);

          // Write back local caches if directory mode active
          if (directoryHandle) {
            hasOcrCache = true;
            hasErasedCache = true;
            
            // Asynchronous file cache updates
            const currentItem = item;
            const currentBlocks = blocks;
            const currentUrl = translatedPreviewUrl;
            (async () => {
              try {
                const cacheDirHandle = await directoryHandle.getDirectoryHandle('translation_cache', { create: true });

                // 1. Write blocks JSON
                const jsonHandle = await cacheDirHandle.getFileHandle(`${currentItem.name}_blocks.json`, { create: true });
                const jsonWritable = await jsonHandle.createWritable();
                await jsonWritable.write(JSON.stringify(currentBlocks));
                await jsonWritable.close();

                // 2. Write erased background image
                const erasedBlob = await renderErasedCanvas(currentItem.previewUrl, currentBlocks, styleConfig);
                const erasedHandle = await cacheDirHandle.getFileHandle(`${currentItem.name}_erased.png`, { create: true });
                const erasedWritable = await erasedHandle.createWritable();
                await erasedWritable.write(erasedBlob);
                await erasedWritable.close();

                // 3. Write final translated image
                if (currentUrl) {
                  const translatedRes = await fetch(currentUrl);
                  const translatedBlob = await translatedRes.blob();
                  const translatedHandle = await cacheDirHandle.getFileHandle(`${currentItem.name}_translated.png`, { create: true });
                  const translatedWritable = await translatedHandle.createWritable();
                  await translatedWritable.write(translatedBlob);
                  await translatedWritable.close();
                }
              } catch (fsErr) {
                console.error('Failed to write local cache files in background', fsErr);
              }
            })();
          }
        } catch (e) {
          console.error('Failed to pre-render translated preview', e);
        }
      }

      setImages(prev => prev.map(img => img.id === imageId ? { 
        ...img, 
        status: 'completed', 
        progress: 100, 
        blocks,
        translatedPreviewUrl,
        lastRenderedHash,
        width: dims.width,
        height: dims.height,
        hasOcrCache: hasOcrCache || img.hasOcrCache,
        hasErasedCache: hasErasedCache || img.hasErasedCache
      } : img));

      addToast(`图片 "${item.name}" 翻译完成`, 'success');
    } catch (err) {
      console.error(err);
      const errMsg = (err as Error).message || '未知错误';
      setImages(prev => prev.map(img => img.id === imageId ? { ...img, status: 'failed', error: errMsg } : img));
      addToast(`图片 "${item.name}" 翻译失败: ${errMsg}`, 'error');
    }
  };

  const startBatchTranslation = async () => {
    if (!config.apiKey && config.provider !== 'custom') {
      addToast('请先配置并填写大模型的 API Key', 'error');
      return;
    }

    const pending = images.filter(img => img.status === 'idle' || img.status === 'failed');
    if (pending.length === 0) {
      addToast('没有等待翻译的图片 (可点击清空后重新上传)', 'info');
      return;
    }

    if (directoryHandle) {
      const hasPermission = await verifyWritePermission(directoryHandle);
      if (!hasPermission) {
        addToast('未获得文件夹的写入权限，翻译将不会同步到本地磁盘。', 'error');
      }
    }

    setIsTranslatingAll(true);
    cancelRef.current = false;
    addToast(`开始批量处理 ${pending.length} 张图片...`, 'info');

    for (const img of pending) {
      if (cancelRef.current) {
        addToast('批量翻译已暂停', 'info');
        break;
      }

      // Ensure files are resolved before processing
      let currentImg = img;
      const resolved = await resolveImageFiles(img);
      currentImg = resolved;

      setImages(prev => prev.map(item => item.id === img.id ? { ...resolved, status: 'processing', progress: 0 } : item));

      try {
        let dims = { width: currentImg.width || 0, height: currentImg.height || 0 };
        if (!currentImg.width || !currentImg.height) {
          if (currentImg.file) {
            dims = await getImageDimensions(currentImg.file);
          }
        }

        // Check if image is solid color
        const isSolid = await checkIfImageIsSolidColor(currentImg.previewUrl);
        if (isSolid) {
          setImages(prev => prev.map(item => item.id === img.id ? {
            ...item,
            status: 'completed',
            progress: 100,
            blocks: [],
            translatedPreviewUrl: item.previewUrl,
            width: dims.width,
            height: dims.height,
            hasOcrCache: false,
            hasErasedCache: false
          } : item));
          addToast(`图片 "${currentImg.name}" 为纯色（空白）图片，已直接跳过并处理下一张。`, 'success');
          continue;
        }

        const blocks = await translateImage(currentImg, config, (p) => {
          setImages(prev => prev.map(item => item.id === img.id ? { ...item, progress: p } : item));
        });

        // Render and cache translated preview image
        let translatedPreviewUrl: string | undefined = undefined;
        let lastRenderedHash: string | undefined = undefined;
        let hasOcrCache = false;
        let hasErasedCache = false;

        if (blocks.length > 0) {
          try {
            if (currentImg.translatedPreviewUrl) {
              URL.revokeObjectURL(currentImg.translatedPreviewUrl);
            }
            translatedPreviewUrl = await renderTranslatedCanvas(currentImg.previewUrl, blocks, styleConfig);
            lastRenderedHash = getRenderHash(blocks, styleConfig);

            // Write back local caches if directory mode active
            if (directoryHandle) {
              hasOcrCache = true;
              hasErasedCache = true;
              
              // Asynchronous cache file writes
              const item = currentImg;
              const currentBlocks = blocks;
              const currentUrl = translatedPreviewUrl;
              (async () => {
                try {
                  const cacheDirHandle = await directoryHandle.getDirectoryHandle('translation_cache', { create: true });

                  // 1. Write blocks JSON
                  const jsonHandle = await cacheDirHandle.getFileHandle(`${item.name}_blocks.json`, { create: true });
                  const jsonWritable = await jsonHandle.createWritable();
                  await jsonWritable.write(JSON.stringify(currentBlocks));
                  await jsonWritable.close();

                  // 2. Write erased background image
                  const erasedBlob = await renderErasedCanvas(item.previewUrl, currentBlocks, styleConfig);
                  const erasedHandle = await cacheDirHandle.getFileHandle(`${item.name}_erased.png`, { create: true });
                  const erasedWritable = await erasedHandle.createWritable();
                  await erasedWritable.write(erasedBlob);
                  await erasedWritable.close();

                  // 3. Write final translated image
                  if (currentUrl) {
                    const translatedRes = await fetch(currentUrl);
                    const translatedBlob = await translatedRes.blob();
                    const translatedHandle = await cacheDirHandle.getFileHandle(`${item.name}_translated.png`, { create: true });
                    const translatedWritable = await translatedHandle.createWritable();
                    await translatedWritable.write(translatedBlob);
                    await translatedWritable.close();
                  }
                } catch (fsErr) {
                  console.error('Failed to write local cache files in background', fsErr);
                }
              })();
            }
          } catch (e) {
            console.error('Failed to pre-render translated preview', e);
          }
        }

        setImages(prev => prev.map(item => item.id === img.id ? { 
          ...item, 
          status: 'completed', 
          progress: 100, 
          blocks,
          translatedPreviewUrl,
          lastRenderedHash,
          width: dims.width,
          height: dims.height,
          hasOcrCache: hasOcrCache || item.hasOcrCache,
          hasErasedCache: hasErasedCache || item.hasErasedCache
        } : item));

      } catch (err) {
        console.error(err);
        const errMsg = (err as Error).message || '未知错误';
        setImages(prev => prev.map(item => item.id === img.id ? { ...item, status: 'failed', error: errMsg } : item));
        addToast(`图片 "${img.name}" 翻译失败: ${errMsg}`, 'error');
      }
    }

    setIsTranslatingAll(false);
  };

  const pauseBatchTranslation = () => {
    cancelRef.current = true;
    setIsTranslatingAll(false);
  };

  const handleExportZip = async () => {
    const completed = images.filter(img => img.status === 'completed' && img.blocks);
    if (completed.length === 0) {
      addToast('当前没有翻译完成的图片可供打包', 'error');
      return;
    }

    addToast('正在生成图片压缩包，这可能需要一些时间...', 'info');
    const zip = new JSZip();

    try {
      for (const img of completed) {
        if (!img.blocks) continue;
        const resolved = await resolveImageFiles(img);
        if (!resolved.previewUrl) continue;
        const blobUrl = await renderTranslatedCanvas(resolved.previewUrl, resolved.blocks || img.blocks, styleConfig);
        const res = await fetch(blobUrl);
        const blob = await res.blob();
        zip.file(`translated_${img.name}`, blob);
        URL.revokeObjectURL(blobUrl);

        // Revoke temporary Object URLs if they weren't loaded before
        if (!img.previewUrl) {
          if (resolved.previewUrl) URL.revokeObjectURL(resolved.previewUrl);
          if (resolved.translatedPreviewUrl) URL.revokeObjectURL(resolved.translatedPreviewUrl);
        }
      }

      const content = await zip.generateAsync({ type: 'blob' });
      const link = document.createElement('a');
      link.download = `translated_images_${Date.now()}.zip`;
      link.href = URL.createObjectURL(content);
      link.click();
      setTimeout(() => {
        URL.revokeObjectURL(link.href);
      }, 500);
      addToast('批量 ZIP 导出成功！', 'success');
    } catch (err) {
      console.error(err);
      addToast(`打包失败: ${(err as Error).message}`, 'error');
    }
  };

  // Get statistics
  const stats = {
    total: images.length,
    idle: images.filter(img => img.status === 'idle').length,
    processing: images.filter(img => img.status === 'processing').length,
    completed: images.filter(img => img.status === 'completed').length,
    failed: images.filter(img => img.status === 'failed').length,
  };

  const selectedImage = images.find(img => img.id === selectedImageId);

  return (
    <div className={`app-container ${!showSidebar ? 'no-sidebar' : ''}`}>
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div className="brand-icon">
              <Sparkles size={22} />
            </div>
            <span className="brand-name">AetherLens Trans</span>
          </div>
          <button
            onClick={() => setShowSidebar(false)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              padding: '0.25rem',
              borderRadius: '4px',
              transition: 'var(--transition-fast)'
            }}
            title="折叠导航栏"
          >
            <ChevronLeft size={18} />
          </button>
        </div>

        <ul className="nav-menu" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flexGrow: 1, overflow: 'hidden' }}>
          <li className="nav-item">
            <a 
              className={`nav-link ${!selectedImageId ? 'active' : ''}`}
              onClick={() => setSelectedImageId(null)}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <Layers size={16} />
              {selectedImageId ? '返回批处理工作区' : '批处理工作区'}
            </a>
          </li>
          
          {selectedImageId && (
            <>
              <div style={{ height: '1px', background: 'var(--border-color)', margin: '0.5rem 0' }} />
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, padding: '0 0.5rem 0.25rem 0.5rem' }}>
                图片队列 ({images.length})
              </div>
              <div 
                className="sidebar-image-queue"
                style={{ 
                  flexGrow: 1, 
                  overflowY: 'auto', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '0.25rem',
                  paddingRight: '4px' 
                }}
              >
                {images.map((img) => {
                  const isActive = img.id === selectedImageId;
                  
                  let dotColor = 'var(--text-muted)';
                  if (img.status === 'processing') dotColor = 'var(--color-info)';
                  else if (img.status === 'completed') dotColor = 'var(--color-success)';
                  else if (img.status === 'failed') dotColor = 'var(--color-danger)';
                  else if (img.status === 'pending') dotColor = 'var(--color-warning)';
                  
                  return (
                    <a
                      key={img.id}
                      onClick={() => handleSelectImage(img.id)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.5rem 0.75rem',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.8rem',
                        color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                        background: isActive ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
                        borderLeft: isActive ? '3px solid var(--color-primary)' : '3px solid transparent',
                        cursor: 'pointer',
                        textDecoration: 'none',
                        transition: 'var(--transition-fast)'
                      }}
                      onMouseEnter={(e) => {
                        if (!isActive) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) e.currentTarget.style.background = 'transparent';
                      }}
                      title={img.name}
                    >
                      <div 
                        style={{ 
                          width: '6px', 
                          height: '6px', 
                          borderRadius: '50%', 
                          backgroundColor: dotColor,
                          flexShrink: 0
                        }} 
                      />
                      <span 
                        style={{ 
                          overflow: 'hidden', 
                          textOverflow: 'ellipsis', 
                          whiteSpace: 'nowrap',
                          flexGrow: 1,
                          fontWeight: isActive ? 600 : 400
                        }}
                      >
                        {img.name}
                      </span>
                    </a>
                  );
                })}
              </div>
            </>
          )}
        </ul>

        <div className="sidebar-footer">
          <div className="status-badge">
            <div className={`status-dot ${config.apiKey || config.provider === 'custom' ? 'active' : 'inactive'}`} />
            <span>
              {config.apiKey || config.provider === 'custom' 
                ? `${config.provider.toUpperCase()} API 已就绪` 
                : 'API Key 未配置'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="main-content">
        <header className="page-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            {!showSidebar && (
              <button
                className="btn btn-secondary"
                onClick={() => setShowSidebar(true)}
                style={{
                  padding: '0.6rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minWidth: 'auto',
                  borderRadius: 'var(--radius-md)',
                  background: 'rgba(255, 255, 255, 0.04)',
                  borderColor: 'var(--border-color)',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                }}
                title="展开导航栏"
              >
                <Menu size={18} />
              </button>
            )}
            <div>
              <h1 className="page-title">
                {selectedImageId ? '译文精细化校对' : 'AI 批量图片翻译'}
              </h1>
              <p className="page-description">
                {selectedImageId 
                  ? '双击文本框进行二次修改，调整字号及背景遮罩颜色，实时保存并下载。' 
                  : '拖入图片、文件夹，使用大模型多模态视觉能力自动 OCR、翻译并还原排版。'}
              </p>
            </div>
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => setShowSettings(!showSettings)}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '0.5rem 1rem', fontSize: '0.85rem' }}
            title={showSettings ? "收起右侧设置面板，为修图区域腾出更多空间" : "展开右侧设置面板"}
          >
            <Settings size={14} />
            {showSettings ? '折叠设置栏' : '展开设置栏'}
          </button>
        </header>

        {/* Dashboard Grid Layout */}
        <div className={`dashboard-grid ${!showSettings ? 'full-width' : ''}`}>
          {/* Main workspace center */}
          <div className="workspace-main-area" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            
            {/* Batch stats panel (Only shown in main list) */}
            {!selectedImageId && (
              <div className="stats-grid">
                <div className="glass-card" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem', borderRadius: '8px' }}>
                    <BarChart3 size={20} className="text-primary" style={{ color: 'var(--color-primary)' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>图片总数</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{stats.total}</div>
                  </div>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ background: 'rgba(245,158,11,0.1)', padding: '0.5rem', borderRadius: '8px' }}>
                    <Clock size={20} style={{ color: 'var(--color-warning)' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>等待中</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{stats.idle}</div>
                  </div>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ background: 'rgba(16,185,129,0.1)', padding: '0.5rem', borderRadius: '8px' }}>
                    <CheckCircle size={20} style={{ color: 'var(--color-success)' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>已完成</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{stats.completed}</div>
                  </div>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ background: 'rgba(239,68,68,0.1)', padding: '0.5rem', borderRadius: '8px' }}>
                    <AlertCircle size={20} style={{ color: 'var(--color-danger)' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>已失败</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{stats.failed}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Selected image viewer vs upload zone */}
            {selectedImageId && selectedImage ? (
              <div>
                <button 
                  className="btn btn-secondary"
                  onClick={() => setSelectedImageId(null)}
                  style={{ marginBottom: '1.5rem', padding: '0.5rem 1rem' }}
                >
                  <ChevronLeft size={16} /> 返回批处理列表
                </button>
                <ImageViewer
                  image={selectedImage}
                  styleConfig={styleConfig}
                  setStyleConfig={setStyleConfig}
                  onUpdateBlocks={handleUpdateBlocks}
                  onTranslateSingle={handleTranslateSingle}
                  isProcessing={selectedImage.status === 'processing'}
                  onNavigate={(dir) => {
                    const idx = images.findIndex(img => img.id === selectedImageId);
                    if (dir === 'prev' && idx > 0) {
                      handleSelectImage(images[idx - 1].id);
                    } else if (dir === 'next' && idx < images.length - 1) {
                      handleSelectImage(images[idx + 1].id);
                    }
                  }}
                  hasPrev={images.findIndex(img => img.id === selectedImageId) > 0}
                  hasNext={images.findIndex(img => img.id === selectedImageId) < images.length - 1}
                />
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', gap: '1rem', flexDirection: 'column' }}>
                  <UploadArea 
                    onFilesSelected={handleFilesSelected} 
                    onSelectLocalDirectory={handleOpenLocalDirectory}
                    selectedDirectoryName={directoryHandle ? directoryHandle.name : null}
                  />
                </div>

                {/* Queue list card */}
                <div className="glass-card">
                  <div className="queue-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <h3 className="card-title" style={{ margin: 0 }}>
                        <Layers size={18} className="text-primary" />
                        待处理图片队列
                      </h3>
                      
                      {/* View mode toggle */}
                      {images.length > 0 && (
                        <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.03)', padding: '2px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                          <button
                            className="provider-btn"
                            style={{ 
                              padding: '2px 8px', 
                              borderRadius: '4px',
                              background: queueViewMode === 'list' ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
                              color: queueViewMode === 'list' ? 'var(--text-main)' : 'var(--text-muted)',
                              border: 'none',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                              fontSize: '0.75rem',
                              fontWeight: 500
                            }}
                            onClick={() => setQueueViewMode('list')}
                            title="列表视图 (节省内存, 适合1000+图片)"
                          >
                            <List size={12} /> 列表
                          </button>
                          <button
                            className="provider-btn"
                            style={{ 
                              padding: '2px 8px', 
                              borderRadius: '4px',
                              background: queueViewMode === 'grid' ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
                              color: queueViewMode === 'grid' ? 'var(--text-main)' : 'var(--text-muted)',
                              border: 'none',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                              fontSize: '0.75rem',
                              fontWeight: 500
                            }}
                            onClick={() => setQueueViewMode('grid')}
                            title="网格缩略图"
                          >
                            <Grid size={12} /> 网格
                          </button>
                        </div>
                      )}
                    </div>
                    
                    <div className="queue-actions">
                      {images.length > 0 && (
                        <>
                          <button
                            className="btn btn-danger"
                            onClick={handleClearQueue}
                            disabled={isTranslatingAll}
                          >
                            <Trash2 size={16} /> 清空队列
                          </button>
                          
                          <button
                            className="btn btn-secondary"
                            onClick={handleExportZip}
                            disabled={stats.completed === 0}
                          >
                            <Download size={16} /> 导出已完成 ZIP
                          </button>
                          
                          {isTranslatingAll ? (
                            <button
                              className="btn btn-secondary"
                              onClick={pauseBatchTranslation}
                              style={{ borderColor: 'var(--color-warning)', color: 'var(--color-warning)' }}
                            >
                              <Pause size={16} /> 暂停翻译
                            </button>
                          ) : (
                            <button
                              className="btn btn-primary"
                              onClick={startBatchTranslation}
                              disabled={stats.idle === 0 && stats.failed === 0}
                            >
                              <Play size={16} /> 开始批量翻译
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>

                  <ImageGrid
                    images={images}
                    selectedId={selectedImageId}
                    onSelect={handleSelectImage}
                    onRemove={handleRemoveImage}
                    viewMode={queueViewMode}
                  />
                </div>
              </>
            )}
          </div>

          {/* Right settings sidebar */}
          {showSettings && (
            <div className="workspace-sidebar-area">
              <SettingsPanel
                config={config}
                setConfig={setConfig}
                styleConfig={styleConfig}
                setStyleConfig={setStyleConfig}
              />
            </div>
          )}
        </div>
      </main>

      {/* Floating Notifications */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  );
}
