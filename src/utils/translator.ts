import type { ImageItem, TranslateConfig, TranslationBlock } from '../types';
import { performLocalOCR } from './ocr';

// Helper to extract and parse JSON from a response string robustly
const extractJson = (text: string): any => {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch (e) {
    let cleaned = trimmed;
    if (cleaned.startsWith('```json')) {
      cleaned = cleaned.replace(/^```json/, '').replace(/```$/, '').trim();
    } else if (cleaned.startsWith('```')) {
      cleaned = cleaned.replace(/^```/, '').replace(/```$/, '').trim();
    }
    try {
      return JSON.parse(cleaned);
    } catch (err) {
      const firstBrace = cleaned.indexOf('{');
      const lastBrace = cleaned.lastIndexOf('}');
      if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
        try {
          return JSON.parse(cleaned.substring(firstBrace, lastBrace + 1));
        } catch (jsonErr) {
          // Try array brackets fallback
          const firstBracket = cleaned.indexOf('[');
          const lastBracket = cleaned.lastIndexOf(']');
          if (firstBracket !== -1 && lastBracket !== -1 && lastBracket > firstBracket) {
            try {
              return JSON.parse(cleaned.substring(firstBracket, lastBracket + 1));
            } catch (bracketErr) {
              // ignore
            }
          }
        }
      }
      throw new Error(`JSON 解析失败: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
};



// Helper to convert File to Base64
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const base64String = reader.result as string;
      const base64Data = base64String.split(',')[1];
      resolve(base64Data);
    };
    reader.onerror = (error) => reject(error);
  });
};

// Helper to get image dimensions
export const getImageDimensions = (file: File): Promise<{ width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
      const dims = { width: img.width, height: img.height };
      URL.revokeObjectURL(img.src);
      resolve(dims);
    };
    img.onerror = (err) => reject(err);
  });
};

// JSON Schema for structured output
const translationSchema = {
  type: "object",
  properties: {
    blocks: {
      type: "array",
      description: "List of all text blocks detected and translated in the image.",
      items: {
        type: "object",
        properties: {
          original_text: { type: "string", description: "The original text detected in the image." },
          translated_text: { type: "string", description: "The translated text in the target language." },
          ymin: { type: "number", description: "Top boundary of the text block as a percentage of image height (0-100)." },
          xmin: { type: "number", description: "Left boundary of the text block as a percentage of image width (0-100)." },
          ymax: { type: "number", description: "Bottom boundary of the text block as a percentage of image height (0-100)." },
          xmax: { type: "number", description: "Right boundary of the text block as a percentage of image width (0-100)." },
          text_color: { type: "string", description: "Approximate hex color of the text, e.g. #FFFFFF" },
          bg_color: { type: "string", description: "Approximate average hex color of the background immediately behind the text, e.g. #000000" },
          font_size_pct: { type: "number", description: "Suggested font size as a percentage of the image height, e.g., 2.0" },
          type: { 
            type: "string", 
            enum: ["bubble", "onomatopoeia"],
            description: "Classification of this text block: 'bubble' (dialogue, narration, or any text/onomatopoeia INSIDE standard speech bubbles), or 'onomatopoeia' (hand-drawn sound effects SFX, mood words, side notes, credits, title text, or page numbers OUTSIDE speech bubbles)." 
          }
        },
        required: ["original_text", "translated_text", "ymin", "xmin", "ymax", "xmax", "text_color", "bg_color", "type"]
      }
    }
  },
  required: ["blocks"]
};

// Gemini API Implementation
const translateWithGemini = async (
  base64Image: string,
  mimeType: string,
  config: TranslateConfig
): Promise<TranslationBlock[]> => {
  const { apiKey, model, customEndpoint, targetLang } = config;
  
  // Build URL: support custom proxy endpoint if provided, otherwise standard Google API
  let baseUrl = 'https://generativelanguage.googleapis.com';
  if (customEndpoint && customEndpoint.trim() !== '') {
    baseUrl = customEndpoint.replace(/\/$/, ''); // Remove trailing slash
  }
  
  // Clean endpoint path mapping
  let url = `${baseUrl}/v1beta/models/${model}:generateContent`;
  if (!customEndpoint || customEndpoint.includes('generativelanguage.googleapis.com')) {
    url += `?key=${apiKey}`;
  }

  const prompt = `You are a high-precision image translation and OCR engine, specialized in parsing manga, webtoons, and complex image layouts.
Task: Detect all text blocks in the image, translate them accurately into "${targetLang}", and classify each block type.

Classification Rules for "type" (CRITICAL):
- "bubble": Dialogue text, conversational lines, narration, or any text (including exclamations/onomatopoeia/SFX) that is located INSIDE a speech bubble.
- "onomatopoeia": Sound effects (SFX), mood words, screams, sighs, side notes, page numbers, title text, translator credits, or hand-drawn action descriptions OUTSIDE speech bubbles.

Note: Japanese manga typically contains vertical text inside speech bubbles, read from right to left. Be extremely careful to detect all vertical text regions as complete blocks and get their coordinates correct.
Provide coordinates for each text block as floating-point percentage values (0.0 to 100.0) relative to the image's overall height and width.
Detect the original text color (as hex code) and background color (as hex code) immediately behind each block. This is critical to covering the original text accurately.
Return your results in a structured JSON object strictly conforming to the requested schema.`;

  const requestBody = {
    contents: [
      {
        parts: [
          { text: prompt },
          {
            inlineData: {
              mimeType: mimeType,
              data: base64Image
            }
          }
        ]
      }
    ],
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: translationSchema,
      temperature: 0.1
    }
  };

  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  // Add auth header if custom endpoint is used and requires it
  if (customEndpoint && !customEndpoint.includes('generativelanguage.googleapis.com')) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API Error (${response.status}): ${errorText || response.statusText}`);
  }

  const data = await response.json();
  const textResponse = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!textResponse) {
    throw new Error("Gemini returned an empty response.");
  }

  const parsed = extractJson(textResponse);
  if (!parsed.blocks || !Array.isArray(parsed.blocks)) {
    throw new Error("Invalid response structure from Gemini API");
  }

  return parsed.blocks.map((block: any, index: number) => ({
    id: `block_${Date.now()}_${index}`,
    original_text: block.original_text || '',
    translated_text: block.translated_text || '',
    ymin: clamp(block.ymin, 0, 100),
    xmin: clamp(block.xmin, 0, 100),
    ymax: clamp(block.ymax, 0, 100),
    xmax: clamp(block.xmax, 0, 100),
    text_color: block.text_color || '#FFFFFF',
    bg_color: block.bg_color || '#000000',
    font_size_pct: block.font_size_pct || 2.0,
    type: block.type || 'bubble'
  }));
};

// OpenAI API Implementation
const translateWithOpenAI = async (
  base64Image: string,
  mimeType: string,
  config: TranslateConfig
): Promise<TranslationBlock[]> => {
  const { apiKey, model, customEndpoint, targetLang } = config;

  let url = 'https://api.openai.com/v1/chat/completions';
  if (customEndpoint && customEndpoint.trim() !== '') {
    url = customEndpoint.replace(/\/$/, '');
    if (!url.endsWith('/chat/completions') && !url.includes('/chat/completions?')) {
      // If user provided base URL, append chat/completions
      url = `${url}/chat/completions`;
    }
  }

  const systemPrompt = `You are a high-precision image translation and OCR engine, specialized in parsing manga, webtoons, and complex image layouts.
    Detect all text blocks in the image. Note that the image may contain vertical text layout (especially common in Japanese manga speech bubbles) and horizontal text. Detect all vertical and horizontal texts as complete blocks and output their coordinates.
    Translate the text accurately into "${targetLang}".
    Provide coordinates for each text block as percentage values (0.0 to 100.0) relative to the image's height and width (e.g. ymin=10.5, xmin=20.0, ymax=15.2, xmax=45.0).
    Detect the original text color and background color as hex codes.
    Return your results in a JSON object with a single root property "blocks" containing an array of these blocks. Each block must have: original_text, translated_text, ymin, xmin, ymax, xmax, text_color, bg_color, font_size_pct, type.
    
    Classification Rules for "type" (CRITICAL):
    - "bubble": Dialogue text, conversational lines, narration, or any text (including exclamations/onomatopoeia/SFX) that is located INSIDE a speech bubble.
    - "onomatopoeia": Sound effects (SFX), mood words, screams, sighs, side notes, page numbers, title text, translator credits, or hand-drawn action descriptions OUTSIDE speech bubbles.`;

  const requestBody = {
    model: model || 'gpt-4o-mini',
    messages: [
      {
        role: 'system',
        content: systemPrompt
      },
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Detect all text in this image, translate it to ${targetLang}, and output the coordinates/colors JSON schema.`
          },
          {
            type: 'image_url',
            image_url: {
              url: `data:${mimeType};base64,${base64Image}`
            }
          }
        ]
      }
    ],
    response_format: { 
      type: "json_object" 
    },
    temperature: 0.1
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI API Error (${response.status}): ${errorText || response.statusText}`);
  }

  const data = await response.json();
  const textResponse = data.choices?.[0]?.message?.content;
  if (!textResponse) {
    throw new Error("OpenAI returned an empty response.");
  }

  const parsed = extractJson(textResponse);
  if (!parsed.blocks || !Array.isArray(parsed.blocks)) {
    throw new Error("Invalid response structure from OpenAI API");
  }

  return parsed.blocks.map((block: any, index: number) => ({
    id: `block_${Date.now()}_${index}`,
    original_text: block.original_text || '',
    translated_text: block.translated_text || '',
    ymin: clamp(block.ymin, 0, 100),
    xmin: clamp(block.xmin, 0, 100),
    ymax: clamp(block.ymax, 0, 100),
    xmax: clamp(block.xmax, 0, 100),
    text_color: block.text_color || '#FFFFFF',
    bg_color: block.bg_color || '#000000',
    font_size_pct: block.font_size_pct || 2.0,
    type: block.type || 'bubble'
  }));
};

// DeepSeek API Implementation with local OCR fallback
const translateTextsWithDeepSeek = async (
  blocks: TranslationBlock[],
  config: TranslateConfig
): Promise<TranslationBlock[]> => {
  const { apiKey, model, customEndpoint, targetLang } = config;
  
  let endpoint = 'https://api.deepseek.com/chat/completions';
  if (customEndpoint && customEndpoint.trim() !== '') {
    endpoint = customEndpoint.replace(/\/$/, '');
    if (!endpoint.endsWith('/chat/completions') && !endpoint.includes('/chat/completions?')) {
      endpoint = `${endpoint}/chat/completions`;
    }
  }

  const prompt = `You are a high-quality translation and text classification engine.
Translate the following array of text blocks extracted from a manga/webtoon image into target language: "${targetLang}".
Also classify each block's type based on its text and coordinates.

Classification Guidelines (Important: text-only context):
- "bubble": Regular conversational dialogue, narration, sentences, questions, or any text (including exclamations/onomatopoeia/SFX) that is spoken or located INSIDE a character's speech bubble.
- "onomatopoeia": Sound effects (SFX), exclamations, single-character cries, sighs, side notes, page numbers, title text, credits, or mood expressions written OUTSIDE speech bubbles.

Return a JSON object containing a single root property "translations" containing an array of objects. Each object must have:
- "id": The matching block ID.
- "translated_text": The translated text.
- "type": The classified type ("bubble" or "onomatopoeia").

Input JSON:
${JSON.stringify(blocks.map(b => ({ 
  id: b.id, 
  text: b.original_text, 
  xmin: Math.round(b.xmin), 
  ymin: Math.round(b.ymin), 
  xmax: Math.round(b.xmax), 
  ymax: Math.round(b.ymax) 
})))}

Return ONLY the raw JSON object. Do not include markdown code block syntax.`;

  const requestBody = {
    model: model || 'deepseek-chat',
    messages: [
      { role: 'user', content: prompt }
    ],
    response_format: { type: "json_object" },
    temperature: 0.1
  };

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`DeepSeek API Error (${response.status}): ${errorText || response.statusText}`);
  }

  const data = await response.json();
  let textResponse = data.choices?.[0]?.message?.content;
  if (!textResponse) {
    throw new Error("DeepSeek returned an empty response.");
  }

  const parsed = extractJson(textResponse);
  if (!parsed.translations || !Array.isArray(parsed.translations)) {
    throw new Error("Invalid response structure from DeepSeek API");
  }

  const translationMap = new Map<string, { translated_text: string, type: 'bubble' | 'onomatopoeia' | 'other' }>();
  parsed.translations.forEach((item: any) => {
    translationMap.set(item.id, {
      translated_text: item.translated_text || '',
      type: item.type || 'bubble'
    });
  });

  return blocks.map(block => {
    const res = translationMap.get(block.id);
    return {
      ...block,
      translated_text: res ? res.translated_text : block.original_text,
      type: res ? res.type : (block.type || 'bubble')
    };
  });
};

// Helper to check if local OCR server is active
const checkLocalOcrServerActive = async (): Promise<boolean> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600); // 600ms quick ping
    const res = await fetch('http://127.0.0.1:5000/health', { signal: controller.signal });
    clearTimeout(timeoutId);
    if (res.ok) {
      const data = await res.json();
      return data.status === 'healthy';
    }
  } catch (e) {
    // Server is offline
  }
  return false;
};

// Translate text blocks using Gemini Text API
const translateTextsWithGeminiContent = async (
  blocks: TranslationBlock[],
  config: TranslateConfig
): Promise<TranslationBlock[]> => {
  const { apiKey, model, customEndpoint, targetLang } = config;
  let baseUrl = 'https://generativelanguage.googleapis.com';
  if (customEndpoint && customEndpoint.trim() !== '') {
    baseUrl = customEndpoint.replace(/\/$/, '');
  }
  let url = `${baseUrl}/v1beta/models/${model}:generateContent`;
  if (!customEndpoint || customEndpoint.includes('generativelanguage.googleapis.com')) {
    url += `?key=${apiKey}`;
  }
  
  const prompt = `You are a high-quality translation and text classification engine.
Translate the following array of text blocks extracted from a manga/webtoon image into target language: "${targetLang}".
Also classify each block's type based on its text and coordinates.

Classification Guidelines (Important: text-only context):
- "bubble": Regular conversational dialogue, narration, sentences, questions, or any text (including exclamations/onomatopoeia/SFX) that is spoken or located INSIDE a character's speech bubble.
- "onomatopoeia": Sound effects (SFX), exclamations, single-character cries, sighs, side notes, page numbers, title text, credits, or mood expressions written OUTSIDE speech bubbles.

Return a JSON object containing a single root property "translations" containing an array of objects. Each object must have:
- "id": The matching block ID.
- "translated_text": The translated text.
- "type": The classified type ("bubble" or "onomatopoeia").

Input JSON:
${JSON.stringify(blocks.map(b => ({ 
  id: b.id, 
  text: b.original_text, 
  xmin: Math.round(b.xmin), 
  ymin: Math.round(b.ymin), 
  xmax: Math.round(b.xmax), 
  ymax: Math.round(b.ymax) 
})))}${""}`;

  const requestBody = {
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: {
      responseMimeType: "application/json",
      responseSchema: {
        type: "object",
        properties: {
          translations: {
            type: "array",
            items: {
              type: "object",
              properties: {
                id: { type: "string" },
                translated_text: { type: "string" },
                type: { 
                  type: "string", 
                  enum: ["bubble", "onomatopoeia"],
                  description: "The classified type of the text block."
                }
              },
              required: ["id", "translated_text", "type"]
            }
          }
        },
        required: ["translations"]
      },
      temperature: 0.1
    }
  };

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (customEndpoint && !customEndpoint.includes('generativelanguage.googleapis.com')) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini Text Translation Error: ${errorText}`);
  }

  const data = await response.json();
  const textResponse = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!textResponse) throw new Error("Gemini returned empty text translation.");
  
  const parsed = extractJson(textResponse);
  const translationMap = new Map<string, { translated_text: string, type: 'bubble' | 'onomatopoeia' | 'other' }>();
  parsed.translations.forEach((item: any) => {
    translationMap.set(item.id, {
      translated_text: item.translated_text || '',
      type: item.type || 'bubble'
    });
  });

  return blocks.map(block => {
    const res = translationMap.get(block.id);
    return {
      ...block,
      translated_text: res ? res.translated_text : block.original_text,
      type: res ? res.type : (block.type || 'bubble')
    };
  });
};

// Translate text blocks using OpenAI Chat API
const translateTextsWithOpenAIContent = async (
  blocks: TranslationBlock[],
  config: TranslateConfig
): Promise<TranslationBlock[]> => {
  const { apiKey, model, customEndpoint, targetLang } = config;
  let url = 'https://api.openai.com/v1/chat/completions';
  if (customEndpoint && customEndpoint.trim() !== '') {
    url = customEndpoint.replace(/\/$/, '');
    if (!url.endsWith('/chat/completions') && !url.includes('/chat/completions?')) {
      url = `${url}/chat/completions`;
    }
  }
  
  const prompt = `You are a high-quality translation and text classification engine.
Translate the following array of text blocks extracted from a manga/webtoon image into target language: "${targetLang}".
Also classify each block's type based on its text and coordinates.

Classification Guidelines (Important: text-only context):
- "bubble": Regular conversational dialogue, narration, sentences, questions, or any text (including exclamations/onomatopoeia/SFX) that is spoken or located INSIDE a character's speech bubble.
- "onomatopoeia": Sound effects (SFX), exclamations, single-character cries, sighs, side notes, page numbers, title text, credits, or mood expressions written OUTSIDE speech bubbles.

Return a JSON object containing a single root property "translations" containing an array of objects. Each object must have:
- "id": The matching block ID.
- "translated_text": The translated text.
- "type": The classified type ("bubble" or "onomatopoeia").

Input JSON:
${JSON.stringify(blocks.map(b => ({ 
  id: b.id, 
  text: b.original_text, 
  xmin: Math.round(b.xmin), 
  ymin: Math.round(b.ymin), 
  xmax: Math.round(b.xmax), 
  ymax: Math.round(b.ymax) 
})))}

Return ONLY the raw JSON object. Do not include markdown code block syntax.`;

  const requestBody = {
    model: model || 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    response_format: { type: "json_object" },
    temperature: 0.1
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenAI Text Translation Error: ${errorText}`);
  }

  const data = await response.json();
  const textResponse = data.choices?.[0]?.message?.content;
  if (!textResponse) throw new Error("OpenAI returned empty text translation.");

  const parsed = extractJson(textResponse);
  const translationMap = new Map<string, { translated_text: string, type: 'bubble' | 'onomatopoeia' | 'other' }>();
  parsed.translations.forEach((item: any) => {
    translationMap.set(item.id, {
      translated_text: item.translated_text || '',
      type: item.type || 'bubble'
    });
  });

  return blocks.map(block => {
    const res = translationMap.get(block.id);
    return {
      ...block,
      translated_text: res ? res.translated_text : block.original_text,
      type: res ? res.type : (block.type || 'bubble')
    };
  });
};

// Main translate function routing
export const translateImage = async (
  item: ImageItem,
  config: TranslateConfig,
  onProgress?: (progress: number) => void
): Promise<TranslationBlock[]> => {
  onProgress?.(10);
  
  const mimeType = item.file.type || 'image/jpeg';
  let blocks: TranslationBlock[] = [];

  const isLocalOcrActive = await checkLocalOcrServerActive();

  if (isLocalOcrActive) {
    console.log('[Translator] 检测到本地高精度 OCR 服务在线，将优先使用 PaddleOCR 进行文本提取以确保排版位置完美...');
    
    // 1. Run local PaddleOCR first (10% to 70%)
    const ocrBlocks = await performLocalOCR(item.file, config.sourceLang, (p) => {
      onProgress?.(10 + Math.round((p / 100) * 60));
    });

    if (ocrBlocks.length === 0) {
      onProgress?.(100);
      return [];
    }

    onProgress?.(75);
    
    // 2. Translate text blocks using the selected LLM provider (75% to 100%)
    if (config.provider === 'gemini') {
      blocks = await translateTextsWithGeminiContent(ocrBlocks, config);
    } else if (config.provider === 'openai' || config.provider === 'custom') {
      blocks = await translateTextsWithOpenAIContent(ocrBlocks, config);
    } else if (config.provider === 'deepseek') {
      blocks = await translateTextsWithDeepSeek(ocrBlocks, config);
    } else {
      throw new Error(`Unsupported provider: ${config.provider}`);
    }
  } else {
    console.log('[Translator] 本地 OCR 服务未在线，使用备用云端视觉/WASM识别方案...');
    
    if (config.provider === 'gemini') {
      const base64Image = await fileToBase64(item.file);
      onProgress?.(30);
      blocks = await translateWithGemini(base64Image, mimeType, config);
    } else if (config.provider === 'openai' || config.provider === 'custom') {
      const base64Image = await fileToBase64(item.file);
      onProgress?.(30);
      blocks = await translateWithOpenAI(base64Image, mimeType, config);
    } else if (config.provider === 'deepseek') {
      // DeepSeek has no vision API, so we run WASM Tesseract.js OCR
      const ocrBlocks = await performLocalOCR(item.file, config.sourceLang, (p) => {
        onProgress?.(10 + Math.round((p / 100) * 60));
      });
      if (ocrBlocks.length === 0) {
        onProgress?.(100);
        return [];
      }
      onProgress?.(75);
      blocks = await translateTextsWithDeepSeek(ocrBlocks, config);
    } else {
      throw new Error(`Unsupported provider: ${config.provider}`);
    }
  }

  onProgress?.(100);
  return blocks;
};

// Test connection endpoint helper
export const testApiConnection = async (config: TranslateConfig): Promise<string> => {
  const { provider, apiKey, model, customEndpoint } = config;
  
  if (provider === 'gemini') {
    let baseUrl = 'https://generativelanguage.googleapis.com';
    if (customEndpoint && customEndpoint.trim() !== '') {
      baseUrl = customEndpoint.replace(/\/$/, '');
    }
    let url = `${baseUrl}/v1beta/models/${model || 'gemini-2.5-flash'}:generateContent`;
    if (!customEndpoint || customEndpoint.includes('generativelanguage.googleapis.com')) {
      url += `?key=${apiKey}`;
    }
    
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (customEndpoint && !customEndpoint.includes('generativelanguage.googleapis.com')) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        contents: [{ parts: [{ text: "Verify connection. Reply with 'OK' only." }] }]
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`(${response.status}): ${errorText || response.statusText}`);
    }
    
    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '连接成功';
  } else {
    // OpenAI, DeepSeek, Custom
    let url = '';
    let defaultModel = model;
    
    if (provider === 'openai') {
      url = 'https://api.openai.com/v1/chat/completions';
      defaultModel = model || 'gpt-4o-mini';
    } else if (provider === 'deepseek') {
      url = 'https://api.deepseek.com/chat/completions';
      defaultModel = model || 'deepseek-chat';
    } else {
      url = customEndpoint.replace(/\/$/, '');
      if (!url.endsWith('/chat/completions') && !url.includes('/chat/completions?')) {
        url = `${url}/chat/completions`;
      }
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: defaultModel,
        messages: [{ role: 'user', content: "Verify connection. Reply with 'OK' only." }],
        max_tokens: 5
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`(${response.status}): ${errorText || response.statusText}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '连接成功';
  }
};

// Util helper to clamp numbers
const clamp = (val: number, min: number, max: number): number => {
  if (isNaN(val)) return min;
  return Math.min(Math.max(val, min), max);
};
