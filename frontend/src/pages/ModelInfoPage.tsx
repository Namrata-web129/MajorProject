import { useEffect, useState } from 'react';
import { modelService } from '@/services/api';
import type { ModelInfo, ModelStatus } from '@/types';
import { Cpu, CheckCircle, XCircle, Loader2 } from 'lucide-react';

function InfoRow({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-subtle last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <span className={`text-sm font-medium text-white ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  );
}

export default function ModelInfoPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      modelService.getInfo(),
      modelService.getStatus(),
      modelService.getCheckpoints().catch(() => [] as string[]),
    ]).then(([infoRes, statusRes, cpRes]) => {
      if (infoRes.status === 'fulfilled') setInfo(infoRes.value);
      if (statusRes.status === 'fulfilled') setStatus(statusRes.value);
      if (cpRes.status === 'fulfilled') setCheckpoints(cpRes.value as string[]);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 size={36} className="text-violet-400 animate-spin" />
    </div>
  );

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="section-title flex items-center gap-3">
          <Cpu size={24} className="text-violet-400" /> Model Information
        </h1>
        <p className="section-subtitle">Architecture details and active detection model</p>
      </div>

      {/* Status Card */}
      {status && (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-white mb-4">System Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="p-3 rounded-xl" style={{ background: 'rgba(124,58,237,0.08)' }}>
              <p className="text-xs text-slate-500 mb-1">Inference Ready</p>
              <div className="flex items-center gap-2">
                {status.model_loaded || status.pretrained_fallback_available
                  ? <CheckCircle size={16} className="text-emerald-400" />
                  : <XCircle size={16} className="text-red-400" />}
                <span className={`text-sm font-bold ${status.model_loaded || status.pretrained_fallback_available ? 'text-emerald-400' : 'text-red-400'}`}>
                  {status.model_loaded ? 'Graph Transformer ready' : status.pretrained_fallback_available ? 'Pretrained ViT ready' : 'Not ready'}
                </span>
              </div>
            </div>
            <div className="p-3 rounded-xl" style={{ background: 'rgba(124,58,237,0.08)' }}>
              <p className="text-xs text-slate-500 mb-1">Compute Device</p>
              <p className="text-sm font-bold text-white uppercase">{status.device}</p>
            </div>
            <div className="p-3 rounded-xl" style={{ background: 'rgba(124,58,237,0.08)' }}>
              <p className="text-xs text-slate-500 mb-1">Active Detector</p>
              <p className="text-sm font-bold text-white">
                {status.model_loaded ? 'Graph Transformer' : 'Pretrained ViT'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Architecture Details */}
      {info ? (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-white mb-1">{info.name}</h2>
          <p className="text-xs text-slate-500 mb-4">{info.architecture}</p>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">
                Vision Transformer
              </p>
              <InfoRow label="Input Image Size" value={`${info.image_size} × ${info.image_size}`} />
              <InfoRow label="Patch Size" value={`${info.patch_size} × ${info.patch_size}`} mono />
              <InfoRow label="Total Patches" value={Math.pow(info.image_size / info.patch_size, 2)} mono />
              <InfoRow label="Embedding Dimension" value={info.embed_dim} mono />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">
                Graph Network
              </p>
              <InfoRow label="KNN Neighbors (K)" value={info.knn_k} mono />
              <InfoRow label="GCN Layers" value={info.gcn_layers} mono />
              <InfoRow label="Transformer Blocks" value={info.transformer_blocks} mono />
              <InfoRow label="Attention Heads" value={info.attention_heads} mono />
            </div>
          </div>

          <div className="mt-6">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">
              Checkpoints
            </p>
            <InfoRow
              label="Pretrain Checkpoint"
              value={info.pretrain_checkpoint_path
                ? <span className="text-emerald-400 text-xs">✓ {info.pretrain_checkpoint_path}</span>
                : <span className="text-slate-600 text-xs">Not available</span>
              }
            />
            <InfoRow
              label="Classifier Checkpoint"
              value={info.checkpoint_path
                ? <span className="text-emerald-400 text-xs">✓ {info.checkpoint_path}</span>
                : <span className="text-slate-600 text-xs">Not available</span>
              }
            />
          </div>
        </div>
      ) : (
        <div className="glass-card p-8 text-center">
          <p className="text-slate-400">No model version registered yet.</p>
        </div>
      )}

      {/* Research Reference */}
      <div className="glass-card p-5">
        <h2 className="font-semibold text-white mb-3">Research Reference</h2>
        <div className="text-sm text-slate-300 space-y-2">
          <p><span className="text-slate-500">Paper:</span> "Self-Supervised Graph Transformer for Deepfake Detection"</p>
          <p><span className="text-slate-500">Authors:</span> Aminollah Khormali and Jiann-Shiun Yuan</p>
          <p><span className="text-slate-500">Published:</span> IEEE Access 2024</p>
          <p className="text-xs text-slate-500 mt-3">
            This system implements the complete methodology: masked image modeling, student-teacher
            self-distillation, spatial KNN graph construction, GCN feature learning, Graph Transformer
            classification, and gradient-weighted attention relevancy mapping.
          </p>
        </div>
      </div>

      {/* Checkpoints on disk */}
      {checkpoints.length > 0 && (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-white mb-3">Checkpoint Files on Disk</h2>
          <div className="space-y-2">
            {checkpoints.map((cp, i) => (
              <div key={i} className="flex items-center gap-2 text-sm font-mono text-slate-300 p-2 rounded-lg"
                   style={{ background: 'rgba(124,58,237,0.06)' }}>
                <CheckCircle size={14} className="text-emerald-400 flex-shrink-0" />
                {cp}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
