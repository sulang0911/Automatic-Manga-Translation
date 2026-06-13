import React, { useState, useRef } from 'react';
import { UploadCloud, FolderOpen } from 'lucide-react';

interface UploadAreaProps {
  onFilesSelected: (files: FileList) => void;
  onSelectLocalDirectory: () => void;
  selectedDirectoryName: string | null;
}

export const UploadArea: React.FC<UploadAreaProps> = ({ 
  onFilesSelected, 
  onSelectLocalDirectory,
  selectedDirectoryName
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(e.dataTransfer.files);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
    }
  };

  return (
    <div className="glass-card" style={{ padding: '1rem' }}>
      <div
        className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-icon-container">
          <UploadCloud size={32} />
        </div>
        <div>
          <p className="upload-text">拖拽图片到这里，或者 <span style={{ color: 'var(--color-primary)', textDecoration: 'underline' }}>点击浏览</span></p>
          <p className="upload-subtext" style={{ marginTop: '4px' }}>
            支持 PNG, JPG, JPEG, WEBP 格式图片批量翻译
            {selectedDirectoryName && (
              <span style={{ display: 'block', color: '#10B981', marginTop: '4px', fontWeight: 'bold' }}>
                已关联本地工作文件夹: {selectedDirectoryName} (磁盘缓存已启用)
              </span>
            )}
          </p>
        </div>
        
        {/* Hidden inputs */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          multiple
          accept="image/*"
          onChange={handleFileChange}
        />
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
        <button
          className="btn btn-secondary"
          style={{ flex: 1, padding: '0.5rem 1rem', fontSize: '0.85rem' }}
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
        >
          <UploadCloud size={16} /> 选择多张图片
        </button>
        <button
          className={`btn ${selectedDirectoryName ? 'btn-primary' : 'btn-secondary'}`}
          style={{ flex: 1, padding: '0.5rem 1rem', fontSize: '0.85rem' }}
          onClick={(e) => {
            e.stopPropagation();
            onSelectLocalDirectory();
          }}
        >
          <FolderOpen size={16} /> {selectedDirectoryName ? '切换本地文件夹' : '选择本地文件夹'}
        </button>
      </div>
    </div>
  );
};

