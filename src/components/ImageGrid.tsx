import React from 'react';
import type { ImageItem } from '../types';
import { Trash2, CheckCircle2, AlertCircle, Loader2, Database, Eraser, Image as ImageIcon } from 'lucide-react';


interface ImageGridProps {
  images: ImageItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string, e: React.MouseEvent) => void;
  viewMode?: 'grid' | 'list';
}

export const ImageGrid: React.FC<ImageGridProps> = ({
  images,
  selectedId,
  onSelect,
  onRemove,
  viewMode = 'list'
}) => {
  if (images.length === 0) {
    return (
      <div className="empty-state">
        <ImageIcon size={48} />
        <p style={{ fontWeight: 500 }}>暂未上传任何图片</p>
        <p style={{ fontSize: '0.85rem' }}>请在上方区域上传图片或选择文件夹开始</p>
      </div>
    );
  }

  const getStatusBadge = (status: ImageItem['status']) => {
    switch (status) {
      case 'pending':
        return (
          <span className="badge badge-status pending">
            <span className="status-dot" style={{ backgroundColor: 'var(--color-warning)' }} />
            排队中
          </span>
        );
      case 'processing':
        return (
          <span className="badge badge-status processing">
            <Loader2 size={10} className="pulse" style={{ animation: 'spin 1.5s linear infinite' }} />
            翻译中
          </span>
        );
      case 'completed':
        return (
          <span className="badge badge-status completed">
            <CheckCircle2 size={10} />
            已完成
          </span>
        );
      case 'failed':
        return (
          <span className="badge badge-status failed">
            <AlertCircle size={10} />
            失败
          </span>
        );
      default:
        return (
          <span className="badge badge-status idle">
            <span className="status-dot" style={{ backgroundColor: 'var(--text-muted)' }} />
            待处理
          </span>
        );
    }
  };

  if (viewMode === 'list') {
    return (
      <div className="queue-list-view" style={{ display: 'flex', flexDirection: 'column', overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', minWidth: '500px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              <th style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>图片名称</th>
              <th style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>处理状态</th>
              <th style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>本地数据缓存</th>
              <th style={{ padding: '0.75rem 1rem', fontWeight: 600, textAlign: 'right' }}>操作</th>
            </tr>
          </thead>
          <tbody>
            {images.map((img) => {
              const isSelected = selectedId === img.id;
              return (
                <tr
                  key={img.id}
                  onClick={() => onSelect(img.id)}
                  style={{
                    borderBottom: '1px solid rgba(255,255,255,0.03)',
                    cursor: 'pointer',
                    background: isSelected ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
                    borderLeft: isSelected ? '3px solid var(--color-primary)' : '3px solid transparent',
                    transition: 'var(--transition-fast)'
                  }}
                  className="list-view-row"
                >
                  <td style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', fontWeight: 500, maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={img.name}>
                    {img.name}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {getStatusBadge(img.status)}
                      {img.status === 'processing' && (
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{img.progress}%</span>
                      )}
                    </div>
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {img.hasOcrCache && (
                        <span 
                          style={{ 
                            display: 'inline-flex', 
                            alignItems: 'center', 
                            gap: '4px', 
                            fontSize: '0.65rem', 
                            background: 'rgba(99, 102, 241, 0.15)', 
                            color: '#a5b4fc', 
                            padding: '2px 6px', 
                            borderRadius: '4px',
                            border: '1px solid rgba(99, 102, 241, 0.25)' 
                          }}
                          title="已加载本地 OCR 及译文数据 JSON"
                        >
                          <Database size={10} /> JSON
                        </span>
                      )}
                      {img.hasErasedCache && (
                        <span 
                          style={{ 
                            display: 'inline-flex', 
                            alignItems: 'center', 
                            gap: '4px', 
                            fontSize: '0.65rem', 
                            background: 'rgba(16, 185, 129, 0.15)', 
                            color: '#6ee7b7', 
                            padding: '2px 6px', 
                            borderRadius: '4px',
                            border: '1px solid rgba(16, 185, 129, 0.25)' 
                          }}
                          title="已关联本地去字后底图缓存"
                        >
                          <Eraser size={10} /> Clean
                        </span>
                      )}
                      {!img.hasOcrCache && !img.hasErasedCache && (
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>无缓存</span>
                      )}
                    </div>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>
                    <button
                      className="image-card-delete"
                      onClick={(e) => onRemove(img.id, e)}
                      style={{ padding: '4px', background: 'none', border: 'none', cursor: 'pointer' }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="images-grid">
      {images.map((img) => (
        <div
          key={img.id}
          className={`image-card ${selectedId === img.id ? 'active' : ''}`}
          onClick={() => onSelect(img.id)}
        >
          <div className="image-thumbnail-container">
            <img src={img.status === 'completed' && img.translatedPreviewUrl ? img.translatedPreviewUrl : img.previewUrl} alt={img.name} className="image-thumbnail" />
            
            {/* Status overlay */}
            <div className="image-overlay-info">
              {getStatusBadge(img.status)}
            </div>
 
            {/* Processing details progress bar */}
            {img.status === 'processing' && (
              <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '4px',
                background: 'rgba(255,255,255,0.1)'
              }}>
                <div style={{
                  height: '100%',
                  width: `${img.progress}%`,
                  background: 'var(--color-primary)',
                  transition: 'width 0.2s ease-out'
                }} />
              </div>
            )}
          </div>
 
          <div className="image-info-bar">
            <span className="image-name" title={img.name}>
              {img.name}
            </span>
            <button
              className="image-card-delete"
              onClick={(e) => onRemove(img.id, e)}
              title="删除图片"
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>
      ))}
 
      {/* Spinner keyframe hack */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
