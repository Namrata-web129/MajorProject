import { useEffect, useState, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { detectionService, explainabilityService, getErrorMessage } from '@/services/api';
import { useAuth } from '@/contexts/AuthContext';
import type { FullAnalysisResponse, ExplainabilityData } from '@/types';
import {
  CheckCircle, XCircle, Clock, AlertTriangle, Brain,
  ChevronLeft, Loader2, Eye
} from 'lucide-react';

function ConfidenceBar({ label, value, isReal }: { label: string; value: number; isReal: boolean }) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className={`font-bold font-mono ${isReal ? 'text-emerald-400' : 'text-red-400'}`}>{pct}%</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${pct}%`,
            background: isReal
              ? 'linear-gradient(90deg, #065f46, #10b981)'
              : 'linear-gradient(90deg, #7f1d1d, #ef4444)',
          }}
        />
      </div>
    </div>
  );
}

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const { isAdmin } = useAuth();
  const [result, setResult] = useState<FullAnalysisResponse | null>(null);
  const [xai, setXai] = useState<ExplainabilityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchResult = async () => {
    if (!id) return;
    try {
      const data = await detectionService.getAnalysis(id);
      setResult(data);

      if (data.analysis.status === 'completed') {
        // Stop polling and fetch XAI
        if (pollingRef.current) clearInterval(pollingRef.current);
        try {
          const xaiData = await explainabilityService.getExplainability(id);
          setXai(xaiData);
        } catch { /* XAI may not be ready */ }
      } else if (data.analysis.status === 'failed') {
        if (pollingRef.current) clearInterval(pollingRef.current);
      }
    } catch (err) {
      setError(getErrorMessage(err));
      if (pollingRef.current) clearInterval(pollingRef.current);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResult();
    pollingRef.current = setInterval(fetchResult, 3000);
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-3">
          <Loader2 size={40} className="text-violet-400 animate-spin mx-auto" />
          <p className="text-slate-400">Loading analysis result...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <AlertTriangle size={40} className="text-red-400" />
        <p className="text-red-300">{error}</p>
        <Link to="/detect" className="btn-primary text-sm">Try Again</Link>
      </div>
    );
  }

  if (!result) return null;

  const { analysis, detection, video_frames, suspicious_timestamps } = result;
  const isProcessing = analysis.status === 'processing' || analysis.status === 'pending';
  const isFailed = analysis.status === 'failed';
  const isFake = detection?.prediction === 'FAKE';

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Back */}
      <Link to="/history" className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors">
        <ChevronLeft size={16} /> Back to History
      </Link>

      {/* Status Header */}
      {isProcessing ? (
        <div className="glass-card p-6 flex items-center gap-4"
             style={{ border: '1px solid rgba(245,158,11,0.3)' }}>
          <Loader2 size={32} className="text-amber-400 animate-spin flex-shrink-0" />
          <div>
            <h1 className="text-xl font-bold text-white">Processing...</h1>
            <p className="text-slate-400 text-sm mt-1">
              Running media analysis. The first run may take longer if pretrained weights need downloading.
              This page updates automatically.
            </p>
          </div>
        </div>
      ) : isFailed ? (
        <div className="glass-card p-6" style={{ border: '1px solid rgba(239,68,68,0.3)' }}>
          <div className="flex items-center gap-3 mb-3">
            <XCircle size={28} className="text-red-400" />
            <h1 className="text-xl font-bold text-white">Analysis Failed</h1>
          </div>
          <p className="text-red-300 text-sm bg-red-950/30 rounded-lg p-3 font-mono">
            {analysis.error_message || 'Unknown error occurred.'}
          </p>
          <div className="mt-4 p-3 rounded-lg text-xs text-slate-400"
               style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)' }}>
            <AlertTriangle size={12} className="inline mr-1.5 text-amber-400" />
            No prediction was generated. Check the error above. The first analysis needs an internet connection
            to download the pretrained detector; later runs use the cached model.
          </div>
          <div className="mt-4">
            {isAdmin ? (
              <Link to="/training" className="btn-primary text-sm">
                Open Model Training
              </Link>
            ) : (
              <Link to="/login" className="btn-secondary text-sm">
                Sign in as administrator
              </Link>
            )}
          </div>
        </div>
      ) : detection ? (
        <div className={`glass-card p-6`}
             style={{ border: `1px solid ${isFake ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)'}` }}>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-4">
              {isFake
                ? <XCircle size={40} className="text-red-400 flex-shrink-0" />
                : <CheckCircle size={40} className="text-emerald-400 flex-shrink-0" />
              }
              <div>
                <p className="text-sm text-slate-400 mb-1">Verdict</p>
                <h1 className={`text-4xl font-black ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
                  {detection.prediction}
                </h1>
                <p className="text-sm text-slate-400 mt-1">
                  {(detection.confidence * 100).toFixed(1)}% uncalibrated model score · {detection.face_count} face{detection.face_count !== 1 ? 's' : ''} analyzed
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-500">{analysis.file_name}</p>
              <p className="text-xs text-slate-600">{new Date(analysis.created_at).toLocaleString()}</p>
              {analysis.processing_duration_ms && (
                <p className="text-xs text-slate-600">{(analysis.processing_duration_ms / 1000).toFixed(1)}s</p>
              )}
            </div>
          </div>

          <div className="mt-5 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-200">
            <AlertTriangle size={13} className="mr-1.5 inline" />
            Detector: {detection.detector_name || 'Project Graph Transformer'}.{' '}
            {detection.detector_notice || 'This experimental prediction is not proof of authenticity. Video results summarize sampled face frames.'}
          </div>

          <div className="mt-6 space-y-3">
            <ConfidenceBar label="Real Probability" value={detection.real_probability} isReal={true} />
            <ConfidenceBar label="Fake Probability" value={detection.fake_probability} isReal={false} />
          </div>
        </div>
      ) : null}

      {/* Explainability Heatmap */}
      {xai && (
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <Brain size={20} className="text-violet-400" />
            <h2 className="font-semibold text-white">Explainability — Graph Transformer Relevancy Map</h2>
          </div>
          {xai.heatmap_base64 ? (
            <div className="space-y-4">
              <img
                src={`data:image/jpeg;base64,${xai.heatmap_base64}`}
                alt="Attention heatmap overlay"
                className="w-full max-w-lg mx-auto rounded-xl"
              />
              <div className="p-3 rounded-xl text-xs text-slate-400"
                   style={{ background: 'rgba(124,58,237,0.08)', border: '1px solid rgba(124,58,237,0.15)' }}>
                <Eye size={12} className="inline mr-1.5 text-violet-400" />
                {xai.explanation_text}
              </div>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">Heatmap not available for this analysis.</p>
          )}

          {/* Patch Relevance Grid */}
          {xai.patch_relevance && xai.patch_relevance.length === 256 && (
            <div className="mt-4">
              <p className="text-xs text-slate-400 mb-2">Patch-level Relevance (16×16 grid)</p>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: 'repeat(16, 1fr)' }}>
                {xai.patch_relevance.map((v, i) => {
                  const intensity = Math.min(1, Math.max(0, v));
                  return (
                    <div
                      key={i}
                      className="aspect-square rounded-sm"
                      style={{
                        background: `rgba(${isFake ? '239,68,68' : '16,185,129'}, ${intensity * 0.9 + 0.05})`,
                      }}
                      title={`Patch ${i}: ${(intensity * 100).toFixed(1)}%`}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Video Frame Results */}
      {video_frames && video_frames.length > 0 && (
        <div className="glass-card overflow-hidden">
          <div className="p-5 border-b border-subtle">
            <h2 className="font-semibold text-white flex items-center gap-2">
              <Clock size={16} className="text-blue-400" />
              Frame-Level Analysis ({video_frames.length} frames)
            </h2>
            {suspicious_timestamps && suspicious_timestamps.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                <span className="text-xs text-red-400 font-medium">Suspicious timestamps:</span>
                {suspicious_timestamps.map((t, i) => (
                  <span key={i} className="badge-fake text-xs">
                    {t.toFixed(1)}s
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="max-h-64 overflow-y-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Frame</th>
                  <th>Time</th>
                  <th>Verdict</th>
                  <th>Real %</th>
                  <th>Fake %</th>
                </tr>
              </thead>
              <tbody>
                {video_frames.map(f => (
                  <tr key={f.frame_index}>
                    <td className="font-mono text-xs">#{f.frame_index}</td>
                    <td className="font-mono text-xs">{f.timestamp_seconds.toFixed(2)}s</td>
                    <td>
                      {f.prediction === 'REAL'
                        ? <span className="badge-real text-xs">Real</span>
                        : <span className="badge-fake text-xs">Fake</span>}
                    </td>
                    <td className="text-emerald-400 font-mono text-xs">{(f.real_probability * 100).toFixed(1)}%</td>
                    <td className="text-red-400 font-mono text-xs">{(f.fake_probability * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Actions */}
      {detection && (
        <div className="flex gap-3">
          <Link to={`/explainability/${id}`} className="btn-secondary flex items-center gap-2 text-sm">
            <Brain size={16} /> Full Explainability
          </Link>
          <Link to="/detect" className="btn-primary text-sm">
            New Analysis
          </Link>
        </div>
      )}
    </div>
  );
}
