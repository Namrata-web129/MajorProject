import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { detectionService, getErrorMessage } from '@/services/api';
import {
  Upload, Image, Video, X, Loader2, AlertCircle,
  FileVideo, Info
} from 'lucide-react';

const ACCEPTED_IMAGES = ['.jpg', '.jpeg', '.png'];
const ACCEPTED_VIDEOS = ['.mp4', '.avi', '.mov'];
const MAX_SIZE_MB = 500;

export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<string>('');
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const isImage = file ? ACCEPTED_IMAGES.some(e => file.name.toLowerCase().endsWith(e)) : false;
  const isVideo = file ? ACCEPTED_VIDEOS.some(e => file.name.toLowerCase().endsWith(e)) : false;
  const preview = file && isImage ? URL.createObjectURL(file) : null;

  const validateFile = (f: File): string | null => {
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    const allowed = [...ACCEPTED_IMAGES, ...ACCEPTED_VIDEOS];
    if (!allowed.includes(ext)) return `File type not allowed. Accepted: ${allowed.join(', ')}`;
    if (f.size > MAX_SIZE_MB * 1024 * 1024) return `File too large. Max: ${MAX_SIZE_MB} MB`;
    return null;
  };

  const handleFile = (f: File) => {
    const err = validateFile(f);
    if (err) { setError(err); return; }
    setError(null);
    setFile(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProgress('Uploading file...');

    try {
      let result;
      if (isImage) {
        setProgress('Running face detection and analysis...');
        result = await detectionService.analyzeImage(file);
      } else {
        setProgress('Extracting frames and running analysis...');
        result = await detectionService.analyzeVideo(file);
      }
      setProgress('Analysis submitted!');
      navigate(`/results/${result.analysis.id}`);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
      setProgress('');
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="section-title">Deepfake Detection</h1>
        <p className="section-subtitle">Upload an image or video to analyze with the Graph Transformer pipeline</p>
      </div>

      <div className="glass-card p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-white">Just exploring the interface?</p>
          <p className="text-xs text-slate-400 mt-1">
            Open a synthetic demo preview. It does not analyze media or show a prediction.
          </p>
        </div>
        <button
          type="button"
          onClick={() => navigate('/demo')}
          className="btn-secondary text-sm justify-center"
        >
          View Demo Preview
        </button>
      </div>

      {/* Upload Zone */}
      <div
        className={`upload-zone p-12 text-center ${dragging ? 'dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !file && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={[...ACCEPTED_IMAGES, ...ACCEPTED_VIDEOS].join(',')}
          onChange={onInputChange}
        />

        {!file ? (
          <div className="space-y-4">
            <div className="flex justify-center gap-4">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
                   style={{ background: 'rgba(124,58,237,0.15)' }}>
                <Image size={26} className="text-violet-400" />
              </div>
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
                   style={{ background: 'rgba(59,130,246,0.15)' }}>
                <Video size={26} className="text-blue-400" />
              </div>
            </div>
            <div>
              <p className="text-white font-semibold text-lg">Drag & drop or click to upload</p>
              <p className="text-slate-400 text-sm mt-1">Images: JPG, JPEG, PNG · Videos: MP4, AVI, MOV</p>
              <p className="text-slate-600 text-xs mt-1">Maximum file size: {MAX_SIZE_MB} MB</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {preview ? (
              <img src={preview} alt="preview" className="max-h-48 mx-auto rounded-xl object-contain" />
            ) : (
              <div className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto"
                   style={{ background: 'rgba(59,130,246,0.15)' }}>
                <FileVideo size={36} className="text-blue-400" />
              </div>
            )}
            <div>
              <p className="text-white font-semibold">{file.name}</p>
              <p className="text-slate-400 text-sm">{formatSize(file.size)} · {isImage ? 'Image' : 'Video'}</p>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl text-sm"
             style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
          <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Info */}
      {file && (
        <div className="glass-card p-4 flex items-start gap-3 text-sm">
          <Info size={16} className="text-blue-400 mt-0.5 flex-shrink-0" />
          <div className="text-slate-300">
            {isImage ? (
              <p>The image will be analyzed: face detection → 320×320 crop → 256-patch ViT encoding → KNN graph → GCN + Graph Transformer → Real/Fake verdict + XAI heatmap.</p>
            ) : (
              <p>Frames will be sampled at 1 fps, each analyzed independently. Results are aggregated (mean fake probability) into a video-level verdict. Suspicious timestamps will be listed.</p>
            )}
          </div>
        </div>
      )}

      {/* Progress */}
      {loading && progress && (
        <div className="flex items-center gap-3 p-4 rounded-xl text-sm"
             style={{ background: 'rgba(124,58,237,0.1)', border: '1px solid rgba(124,58,237,0.2)', color: '#a78bfa' }}>
          <Loader2 size={16} className="animate-spin flex-shrink-0" />
          <span>{progress}</span>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        {file && (
          <button
            onClick={() => { setFile(null); setError(null); }}
            className="btn-secondary flex items-center gap-2"
            disabled={loading}
          >
            <X size={16} /> Clear
          </button>
        )}
        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="btn-primary flex-1 justify-center text-base py-4"
        >
          {loading ? (
            <><Loader2 size={18} className="animate-spin" /> Analyzing...</>
          ) : (
            <><Upload size={18} /> Analyze {isVideo ? 'Video' : 'Image'}</>
          )}
        </button>
      </div>

      {/* Pipeline Info */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-white mb-3">Processing Pipeline</h3>
        <div className="grid grid-cols-3 gap-2 text-xs text-center">
          {[
            'Face Detection', 'Patch Extraction', 'ViT Encoding',
            'Graph Construction', 'GCN Processing', 'Transformer Classification'
          ].map((step, i) => (
            <div key={i} className="p-2 rounded-lg" style={{ background: 'rgba(124,58,237,0.08)' }}>
              <span className="text-violet-400 font-mono">{String(i + 1).padStart(2, '0')}</span>
              <p className="text-slate-400 mt-0.5">{step}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
