import React, { useState, useEffect } from 'react';
import type { TranslateConfig, StyleConfig, APIProvider } from '../types';
import { Eye, EyeOff, Settings, Sparkles, Sliders, Type, Palette, Loader2, Activity } from 'lucide-react';
import { testApiConnection } from '../utils/translator';


interface SettingsPanelProps {
  config: TranslateConfig;
  setConfig: React.Dispatch<React.SetStateAction<TranslateConfig>>;
  styleConfig: StyleConfig;
  setStyleConfig: React.Dispatch<React.SetStateAction<StyleConfig>>;
  onRenderAllFonts?: () => Promise<void>;
  isRenderingAllFonts?: boolean;
}

const SOURCE_LANGUAGES = [
  { code: 'auto', name: '自动检测 (大模型推荐)' },
  { code: 'ja', name: '日文 (包含漫画竖排/横排文字)' },
  { code: 'en', name: '英文 (English)' },
  { code: 'zh', name: '中文 (Chinese)' },
  { code: 'ko', name: '韩文 (Korean)' }
];

const LANGUAGES = [
  { code: '简体中文', name: '简体中文 (Simplified Chinese)' },
  { code: '繁體中文', name: '繁體中文 (Traditional Chinese)' },
  { code: 'English', name: 'English (English)' },
  { code: '日本語', name: '日本語 (Japanese)' },
  { code: '한국어', name: '한국어 (Korean)' },
  { code: 'Español', name: 'Español (Spanish)' },
  { code: 'Français', name: 'Français (French)' },
  { code: 'Deutsch', name: 'Deutsch (German)' },
  { code: 'Русский', name: 'Русский (Russian)' },
  { code: 'Português', name: 'Português (Portuguese)' },
  { code: 'Italiano', name: 'Italiano (Italian)' },
  { code: 'العربية', name: 'العربية (Arabic)' },
];

const FONTS = [
  { value: 'system-ui', name: '系统默认 (System Default)' },
  { value: 'Microsoft YaHei', name: '微软雅黑 (YaHei Sans)' },
  { value: 'SimHei', name: '黑体 (SimHei Bold)' },
  { value: 'KaiTi', name: '楷体 (KaiTi Serif)' },
  { value: 'FangSong', name: '仿宋 (FangSong)' },
  { value: 'Inter', name: 'Inter (English Sans)' },
  { value: 'Outfit', name: 'Outfit (English Geometrics)' }
];

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  config,
  setConfig,
  styleConfig,
  setStyleConfig,
  onRenderAllFonts,
  isRenderingAllFonts
}) => {
  const [showKey, setShowKey] = useState(false);
  const [activeTab, setActiveTab] = useState<'api' | 'style'>('api');
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ status: 'success' | 'error'; message: string } | null>(null);
  const [ocrStatus, setOcrStatus] = useState<{
    status: 'online' | 'offline' | 'loading';
    engine: string;
    device: string;
    lama: string;
  }>({
    status: 'loading',
    engine: '加载中...',
    device: '-',
    lama: '-',
  });

  useEffect(() => {
    let active = true;
    const checkStatus = async () => {
      try {
        const res = await fetch('http://127.0.0.1:5000/health');
        if (res.ok) {
          const data = await res.json();
          if (active) {
            setOcrStatus({
              status: 'online',
              engine: data.engine_detail || data.engine || 'PaddleOCR',
              device: data.device || 'CPU',
              lama: data.lama_available ? 'LaMa 高清擦除' : 'OpenCV 基础擦除'
            });
          }
        } else {
          throw new Error();
        }
      } catch (e) {
        if (active) {
          setOcrStatus({
            status: 'offline',
            engine: '未连接',
            device: '未知',
            lama: '未知'
          });
        }
      }
    };
    
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const handleTestConnection = async () => {
    if (!config.apiKey && config.provider !== 'custom') {
      setTestResult({ status: 'error', message: '请填写 API Key 后再进行测试' });
      return;
    }
    
    setIsTesting(true);
    setTestResult(null);
    try {
      const resp = await testApiConnection(config);
      setTestResult({
        status: 'success',
        message: `连接成功！模型响应: "${resp}"`
      });
    } catch (err) {
      setTestResult({
        status: 'error',
        message: `连接失败: ${(err as Error).message}`
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleProviderChange = (provider: APIProvider) => {
    let defaultModel = 'gemini-2.5-flash';
    if (provider === 'openai') defaultModel = 'gpt-4o-mini';
    else if (provider === 'deepseek') defaultModel = 'deepseek-chat';
    else if (provider === 'custom') defaultModel = 'gpt-4o-mini';

    setConfig(prev => ({
      ...prev,
      provider,
      model: defaultModel,
      // Clear endpoint override if switching back to official hosts, or set defaults
      customEndpoint: provider === 'custom' ? '' : prev.customEndpoint
    }));
  };

  const updateConfig = (key: keyof TranslateConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const updateStyle = (key: keyof StyleConfig, value: any) => {
    setStyleConfig(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="glass-card">
      <div className="provider-selector">
        <button
          className={`provider-btn ${activeTab === 'api' ? 'active' : ''}`}
          onClick={() => setActiveTab('api')}
        >
          <Settings size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
          API 与模型设置
        </button>
        <button
          className={`provider-btn ${activeTab === 'style' ? 'active' : ''}`}
          onClick={() => setActiveTab('style')}
        >
          <Sliders size={14} style={{ marginRight: '6px', verticalAlign: 'middle' }} />
          翻译文本样式
        </button>
      </div>

      {activeTab === 'api' ? (
        <div className="api-settings">
          <h3 className="card-title">
            <Sparkles size={18} className="text-primary" />
            AI 翻译引擎配置
          </h3>

          {/* 本地服务状态卡片 */}
          <div style={{
            marginBottom: '1.5rem',
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: '0.85rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <span style={{ fontWeight: '600', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Activity size={14} className="text-primary" />
                本地 OCR 与擦除服务状态
              </span>
              <span style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.8rem',
                fontWeight: 'bold',
                color: ocrStatus.status === 'online' ? 'var(--color-success)' : ocrStatus.status === 'offline' ? '#fca5a5' : 'var(--text-muted)'
              }}>
                <span style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: ocrStatus.status === 'online' ? '#10b981' : ocrStatus.status === 'offline' ? '#ef4444' : '#6b7280',
                  animation: ocrStatus.status === 'online' ? 'pulse 2s infinite' : 'none',
                  display: 'inline-block'
                }}></span>
                {ocrStatus.status === 'online' ? '在线' : ocrStatus.status === 'offline' ? '离线' : '检测中...'}
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <div>
                <span style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', opacity: 0.6 }}>文字识别模型</span>
                <span style={{ fontWeight: '500', color: 'var(--text-main)' }}>{ocrStatus.engine}</span>
              </div>
              <div>
                <span style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', opacity: 0.6 }}>硬件运算后端</span>
                <span style={{ fontWeight: '500', color: 'var(--text-main)' }}>{ocrStatus.device}</span>
              </div>
              <div>
                <span style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', opacity: 0.6 }}>背景擦除算法</span>
                <span style={{ fontWeight: '500', color: 'var(--text-main)' }}>{ocrStatus.lama}</span>
              </div>
              <div>
                <span style={{ display: 'block', fontSize: '0.75rem', color: 'var(--text-muted)', opacity: 0.6 }}>大模型翻译引擎</span>
                <span style={{ fontWeight: '500', color: 'var(--text-main)' }}>{config.provider.toUpperCase()} ({config.model})</span>
              </div>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">服务商 (API Provider)</label>
            <div className="provider-selector" style={{ marginBottom: 0 }}>
              <button
                type="button"
                className={`provider-btn ${config.provider === 'gemini' ? 'active' : ''}`}
                onClick={() => handleProviderChange('gemini')}
              >
                Gemini
              </button>
              <button
                type="button"
                className={`provider-btn ${config.provider === 'openai' ? 'active' : ''}`}
                onClick={() => handleProviderChange('openai')}
              >
                OpenAI
              </button>
              <button
                type="button"
                className={`provider-btn ${config.provider === 'deepseek' ? 'active' : ''}`}
                onClick={() => handleProviderChange('deepseek')}
              >
                DeepSeek
              </button>
              <button
                type="button"
                className={`provider-btn ${config.provider === 'custom' ? 'active' : ''}`}
                onClick={() => handleProviderChange('custom')}
              >
                自定义 / 代理
              </button>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">API Key</label>
            <div style={{ position: 'relative' }}>
              <input
                type={showKey ? 'text' : 'password'}
                className="form-input"
                placeholder={config.provider === 'gemini' ? 'AIzaSy...' : config.provider === 'deepseek' ? 'sk-...' : 'sk-...'}
                value={config.apiKey}
                onChange={e => updateConfig('apiKey', e.target.value)}
                style={{ paddingRight: '2.5rem' }}
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                style={{
                  position: 'absolute',
                  right: '0.75rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-muted)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          {config.provider === 'custom' && (
            <div className="form-group">
              <label className="form-label">API 代理端点 (Endpoint)</label>
              <input
                type="text"
                className="form-input"
                placeholder="https://api.yourproxy.com/v1"
                value={config.customEndpoint}
                onChange={e => updateConfig('customEndpoint', e.target.value)}
              />
              <span className="switch-sublabel" style={{ marginTop: '0.25rem', display: 'block' }}>
                支持 OpenAI 格式的代理或本地大模型接口（如 Ollama, DeepSeek API）。
              </span>
            </div>
          )}

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">翻译模型 (Model)</label>
              {config.provider === 'gemini' ? (
                <select
                  className="form-select"
                  value={config.model}
                  onChange={e => updateConfig('model', e.target.value)}
                >
                  <option value="gemini-2.5-flash">gemini-2.5-flash (推荐 - 快速)</option>
                  <option value="gemini-2.5-pro">gemini-2.5-pro (高质量 - 慢)</option>
                  <option value="gemini-1.5-flash">gemini-1.5-flash (稳定)</option>
                  <option value="gemini-1.5-pro">gemini-1.5-pro (精细)</option>
                </select>
              ) : config.provider === 'openai' ? (
                <select
                  className="form-select"
                  value={config.model}
                  onChange={e => updateConfig('model', e.target.value)}
                >
                  <option value="gpt-4o-mini">gpt-4o-mini (推荐 - 性价比)</option>
                  <option value="gpt-4o">gpt-4o (全能旗舰)</option>
                </select>
              ) : config.provider === 'deepseek' ? (
                <select
                  className="form-select"
                  value={config.model}
                  onChange={e => updateConfig('model', e.target.value)}
                >
                  <option value="deepseek-chat">deepseek-chat (DeepSeek V3 / 极速)</option>
                  <option value="deepseek-reasoner">deepseek-reasoner (DeepSeek R1 / 推理)</option>
                </select>
              ) : (
                <input
                  type="text"
                  className="form-input"
                  placeholder="deepseek-chat"
                  value={config.model}
                  onChange={e => updateConfig('model', e.target.value)}
                />
              )}
            </div>

            <div className="form-group">
              <label className="form-label">图片文字语言 (Source Language)</label>
              <select
                className="form-select"
                value={config.sourceLang}
                onChange={e => updateConfig('sourceLang', e.target.value)}
              >
                {SOURCE_LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">目标翻译语言 (Target Language)</label>
              <select
                className="form-select"
                value={config.targetLang}
                onChange={e => updateConfig('targetLang', e.target.value)}
              >
                {LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group" style={{ marginTop: '2rem' }}>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ width: '100%', gap: '8px' }}
                onClick={handleTestConnection}
                disabled={isTesting}
              >
                {isTesting ? (
                  <Loader2 size={16} className="pulse" style={{ animation: 'spin 1.5s linear infinite' }} />
                ) : null}
                {isTesting ? '正在发送测试请求...' : '测试大模型连接'}
              </button>

              {testResult && (
                <div 
                  style={{ 
                    marginTop: '1rem', 
                    padding: '0.75rem 1rem', 
                    borderRadius: 'var(--radius-md)', 
                    fontSize: '0.85rem',
                    border: '1px solid',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '8px',
                    backgroundColor: testResult.status === 'success' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)',
                    borderColor: testResult.status === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                    color: testResult.status === 'success' ? 'var(--color-success)' : '#fca5a5'
                  }}
                >
                  <div style={{ flexGrow: 1, wordBreak: 'break-all', lineHeight: '1.4' }}>
                    {testResult.message}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="style-settings">
          <h3 className="card-title">
            <Type size={18} className="text-primary" />
            排版与样式渲染
          </h3>

          <div className="form-group">
            <label className="form-label">渲染字体 (Font Family)</label>
            <select
              className="form-select"
              value={styleConfig.fontFamily}
              onChange={e => updateStyle('fontFamily', e.target.value)}
            >
              {FONTS.map(f => (
                <option key={f.value} value={f.value}>
                  {f.name}
                </option>
              ))}
            </select>
          </div>

          <div className="switch-group" style={{ marginBottom: '1.5rem' }}>
            <div className="switch-label-container">
              <span className="form-label" style={{ marginBottom: 0 }}>字号智能自适应 (Auto-Fit)</span>
              <span className="switch-sublabel">当译文超出气泡框高度时自动缩小字号</span>
            </div>
            <label className="switch">
              <input
                type="checkbox"
                checked={styleConfig.autoFitFontSize}
                onChange={e => updateStyle('autoFitFontSize', e.target.checked)}
              />
              <span className="slider"></span>
            </label>
          </div>

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <label className="form-label">字体大小缩放 (Font Scale)</label>
              <span style={{ fontSize: '0.8rem', color: 'var(--color-primary)', fontWeight: 'bold' }}>
                {styleConfig.fontSizeScale.toFixed(1)}x
              </span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={styleConfig.fontSizeScale}
              onChange={e => updateStyle('fontSizeScale', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">文本颜色模式</label>
              <select
                className="form-select"
                value={styleConfig.textColorMode}
                onChange={e => updateStyle('textColorMode', e.target.value)}
              >
                <option value="original">自动检测 (原文颜色)</option>
                <option value="custom">自定义固定颜色</option>
              </select>
            </div>

            {styleConfig.textColorMode === 'custom' && (
              <div className="form-group">
                <label className="form-label">自定义文本颜色</label>
                <div className="color-picker-input-container">
                  <input
                    type="color"
                    className="color-dot-picker"
                    value={styleConfig.customTextColor}
                    onChange={e => updateStyle('customTextColor', e.target.value)}
                  />
                  <input
                    type="text"
                    className="form-input"
                    value={styleConfig.customTextColor}
                    onChange={e => updateStyle('customTextColor', e.target.value)}
                    style={{ textTransform: 'uppercase' }}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">遮罩底色模式 (Mask Background)</label>
              <select
                className="form-select"
                value={styleConfig.bgColorMode}
                onChange={e => updateStyle('bgColorMode', e.target.value)}
              >
                <option value="original">自动检测 (背景克隆)</option>
                <option value="custom">自定义固定遮罩</option>
                <option value="none">无遮罩 (纯文本覆盖)</option>
              </select>
            </div>

            {styleConfig.bgColorMode === 'custom' && (
              <div className="form-group">
                <label className="form-label">自定义遮罩颜色</label>
                <div className="color-picker-input-container">
                  <input
                    type="color"
                    className="color-dot-picker"
                    value={styleConfig.customBgColor}
                    onChange={e => updateStyle('customBgColor', e.target.value)}
                  />
                  <input
                    type="text"
                    className="form-input"
                    value={styleConfig.customBgColor}
                    onChange={e => updateStyle('customBgColor', e.target.value)}
                    style={{ textTransform: 'uppercase' }}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="form-group">
            <label className="form-label">气泡外拟声词/语气词处理 (Onomatopoeia)</label>
            <select
              className="form-select"
              value={styleConfig.onomatopoeiaMode}
              onChange={e => updateStyle('onomatopoeiaMode', e.target.value)}
            >
              <option value="ignore">保持原样 (不翻译且不绘制遮罩)</option>
              <option value="transparent">翻译但去除遮罩 (透明底文本覆盖)</option>
              <option value="normal">正常翻译 (保留常规挡板遮罩)</option>
            </select>
            <span className="switch-sublabel" style={{ marginTop: '0.25rem', display: 'block' }}>
              漫画插画上的拟声词和语气词，推荐“保持原样”以防止矩形掩膜破坏画面美感。
            </span>
          </div>

          {styleConfig.bgColorMode !== 'none' && (
            <div className="form-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                <label className="form-label">遮罩底色不透明度 (Opacity)</label>
                <span style={{ fontSize: '0.8rem', color: 'var(--color-primary)', fontWeight: 'bold' }}>
                  {styleConfig.bgOpacity}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={styleConfig.bgOpacity}
                onChange={e => updateStyle('bgOpacity', parseInt(e.target.value))}
              />
            </div>
          )}

          <div className="form-group">
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Palette size={14} /> Text FX 辅助渲染特效
            </h4>
            
            <div className="switch-group">
              <div className="switch-label-container">
                <span className="form-label" style={{ marginBottom: 0 }}>文字描边 (Text Stroke)</span>
                <span className="switch-sublabel">在边缘生成描边以提高可读性</span>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={styleConfig.textStroke}
                  onChange={e => updateStyle('textStroke', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>

            {styleConfig.textStroke && (
              <div className="form-row" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">描边颜色</label>
                  <input
                    type="color"
                    className="color-dot-picker"
                    style={{ width: '100%' }}
                    value={styleConfig.strokeColor}
                    onChange={e => updateStyle('strokeColor', e.target.value)}
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                    <label className="form-label">描边宽度</label>
                    <span style={{ fontSize: '0.8rem' }}>{styleConfig.strokeWidth}px</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="6"
                    step="1"
                    value={styleConfig.strokeWidth}
                    onChange={e => updateStyle('strokeWidth', parseInt(e.target.value))}
                  />
                </div>
              </div>
            )}

            <div className="switch-group">
              <div className="switch-label-container">
                <span className="form-label" style={{ marginBottom: 0 }}>文字阴影 (Drop Shadow)</span>
                <span className="switch-sublabel">为文字添加立体黑影阴影</span>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={styleConfig.textShadow}
                  onChange={e => updateStyle('textShadow', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>

            <div className="switch-group">
              <div className="switch-label-container">
                <span className="form-label" style={{ marginBottom: 0 }}>粗体字 (Bold)</span>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={styleConfig.fontBold}
                  onChange={e => updateStyle('fontBold', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>

            <div className="switch-group">
              <div className="switch-label-container">
                <span className="form-label" style={{ marginBottom: 0 }}>斜体字 (Italic)</span>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={styleConfig.fontItalic}
                  onChange={e => updateStyle('fontItalic', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>

            <div className="switch-group" style={{ marginTop: '1.25rem' }}>
              <div className="switch-label-container">
                <span className="form-label" style={{ marginBottom: 0 }}>导出压缩版图片 (肉眼无损)</span>
                <span className="switch-sublabel">在导出或打包译图时，使用高质量 WebP 格式替代庞大的 PNG 格式，大幅缩减文件大小</span>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={styleConfig.exportCompressed}
                  onChange={e => updateStyle('exportCompressed', e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>

          <div className="form-group" style={{ marginTop: '1.5rem', marginBottom: 0 }}>
            {onRenderAllFonts && (
              <button
                type="button"
                className="btn btn-primary"
                style={{ width: '100%', gap: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                onClick={onRenderAllFonts}
                disabled={isRenderingAllFonts}
              >
                {isRenderingAllFonts ? (
                  <Loader2 size={16} className="pulse" style={{ animation: 'spin 1.5s linear infinite' }} />
                ) : (
                  <Type size={16} />
                )}
                {isRenderingAllFonts ? '正在重新绘制所有字体...' : '重新渲染所有图片字体'}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
