export interface TranslationBlock {
  id: string;
  original_text: string;
  translated_text: string;
  ymin: number; // 0-100
  xmin: number; // 0-100
  ymax: number; // 0-100
  xmax: number; // 0-100
  text_color: string; // Hex color code
  bg_color: string; // Hex color code
  font_size_pct?: number; // suggested font size as pct of image height
  type?: 'bubble' | 'onomatopoeia' | 'other';
}

export type APIProvider = 'gemini' | 'openai' | 'deepseek' | 'custom';

export interface TranslateConfig {
  provider: APIProvider;
  apiKey: string;
  model: string;
  customEndpoint: string;
  targetLang: string;
  sourceLang: string;
}

export type OverlayStyle = 'cover' | 'mask' | 'comparison' | 'original' | 'subtitle';

export interface StyleConfig {
  fontFamily: string;
  fontSizeScale: number; // e.g. 1.0 (default), 1.2, 0.8
  textColorMode: 'original' | 'custom';
  customTextColor: string;
  bgColorMode: 'original' | 'custom' | 'none';
  customBgColor: string;
  bgOpacity: number; // 0 to 100
  textShadow: boolean;
  textStroke: boolean;
  strokeColor: string;
  strokeWidth: number; // px
  fontBold: boolean;
  fontItalic: boolean;
  autoFitFontSize: boolean;
  onomatopoeiaMode: 'ignore' | 'transparent' | 'normal';
  exportCompressed: boolean;
}

export interface ImageItem {
  id: string;
  name: string;
  file?: File;
  previewUrl: string;
  translatedPreviewUrl?: string; // Cache the generated canvas translation
  erasedPreviewUrl?: string; // Cache the pre-inpainted background image
  status: 'idle' | 'pending' | 'processing' | 'completed' | 'failed';
  progress: number; // 0 to 100
  error?: string;
  width?: number;
  height?: number;
  blocks?: TranslationBlock[];
  lastRenderedHash?: string;
  hasOcrCache?: boolean;
  hasErasedCache?: boolean;
  fileHandle?: FileSystemFileHandle;
}
