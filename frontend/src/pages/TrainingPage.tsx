import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { trainingService, modelService, getErrorMessage } from '@/services/api';
import type { TrainingStatus, ModelStatus } from '@/types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
  Activity, Play, StopCircle, Loader2, AlertTriangle,
  CheckCircle, XCircle, Cpu, Lock
} from 'lucide-react';

const defaultPretrainConfig = {
  dataset_path: './datasets/unlabeled',
  batch_size: 32,
  epochs: 100,
  teacher_momentum: 0.996,
  mask_ratio: 0.75,
  save_every: 10,
  seed: 42,
  model_name: 'SelfSupervisedViT_v1',
};

const defaultClassifierConfig = {
  dataset_path: './datasets',
  batch_size: 32,
  epochs: 50,
  learning_rate: 0.0001,
  freeze_backbone: false,
  save_every: 5,
  seed: 42,
  model_name: 'GraphTransformer_v1',
};

export default function TrainingPage() {
  const { isAdmin } = useAuth();
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingStatus | null>(null);
  const [jobHistory, setJobHistory] = useState<TrainingStatus[]>([]);
  const [pretrainCfg, setPretrainCfg] = useState(defaultPretrainConfig);
  const [classifierCfg, setClassifierCfg] = useState(defaultClassifierConfig);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'pretrain' | 'classifier'>('pretrain');

  useEffect(() => {
    modelService.getStatus().then(setModelStatus).catch(() => {});
    trainingService.getHistory().then(setJobHistory).catch(() => {});
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') return;
    const iv = setInterval(async () => {
      try {
        const updated = await trainingService.getStatus(activeJob.job_id);
        setActiveJob(updated);
        if (updated.status === 'completed' || updated.status === 'failed' || updated.status === 'stopped') {
          clearInterval(iv);
          modelService.getStatus().then(setModelStatus).catch(() => {});
        }
      } catch {}
    }, 3000);
    return () => clearInterval(iv);
  }, [activeJob?.job_id, activeJob?.status]);

  if (!isAdmin) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <Lock size={40} className="text-slate-600" />
        <p className="text-slate-400">Admin access required to view training controls.</p>
      </div>
    );
  }

  const startPretraining = async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await trainingService.startPretrain(pretrainCfg);
      const status = await trainingService.getStatus(res.job_id);
      setActiveJob(status);
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const startClassifier = async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await trainingService.startClassifier(classifierCfg);
      const status = await trainingService.getStatus(res.job_id);
      setActiveJob(status);
    } catch (e) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const stopJob = async () => {
    if (!activeJob) return;
    try {
      await trainingService.stop(activeJob.job_id);
    } catch {}
  };

  const isRunning = activeJob?.status === 'running' || activeJob?.status === 'queued';

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="section-title flex items-center gap-3">
          <Activity size={24} className="text-violet-400" /> Model Training
        </h1>
        <p className="section-subtitle">Configure and launch self-supervised pretraining or supervised classifier training</p>
      </div>

      {/* Model Status */}
      {modelStatus && (
        <div className="glass-card p-4 flex items-center gap-4">
          <Cpu size={18} className="text-violet-400 flex-shrink-0" />
          <div className="flex-1 grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-slate-500">Device</p>
              <p className="text-sm font-bold text-white uppercase">{modelStatus.device}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Checkpoint</p>
              <p className={`text-sm font-bold ${modelStatus.checkpoint_exists ? 'text-emerald-400' : 'text-slate-500'}`}>
                {modelStatus.checkpoint_exists ? 'Available' : 'None'}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500">Pretrain</p>
              <p className={`text-sm font-bold ${modelStatus.pretrain_checkpoint_exists ? 'text-emerald-400' : 'text-slate-500'}`}>
                {modelStatus.pretrain_checkpoint_exists ? 'Done' : 'None'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl text-sm"
             style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
          <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Active Job Progress */}
      {activeJob && (
        <div className="glass-card p-5" style={{ border: `1px solid ${isRunning ? 'rgba(124,58,237,0.4)' : 'rgba(100,116,139,0.2)'}` }}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              {isRunning && <Loader2 size={18} className="text-violet-400 animate-spin" />}
              <h2 className="font-semibold text-white capitalize">
                {activeJob.job_type} Job — {activeJob.status}
              </h2>
            </div>
            {isRunning && (
              <button onClick={stopJob} className="btn-danger text-xs py-1.5 px-3">
                <StopCircle size={14} /> Stop
              </button>
            )}
          </div>

          <div className="mb-3 text-sm text-slate-400">
            Epoch {activeJob.current_epoch} / {activeJob.total_epochs} ({activeJob.progress_pct}%)
          </div>
          <div className="progress-bar mb-4">
            <div className="progress-fill" style={{ width: `${activeJob.progress_pct}%` }} />
          </div>

          {/* Loss Chart */}
          {activeJob.metrics.length > 0 && (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={activeJob.metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(124,58,237,0.1)" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#0a1628', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 8 }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line type="monotone" dataKey="train_loss" stroke="#7c3aed" strokeWidth={2} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" strokeWidth={2} dot={false} name="Val Loss" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      )}

      {/* Training Configuration Tabs */}
      {!isRunning && (
        <div className="glass-card p-5">
          <div className="flex gap-2 mb-5">
            <button onClick={() => setTab('pretrain')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${tab === 'pretrain' ? 'bg-violet-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              1. Self-Supervised Pretraining
            </button>
            <button onClick={() => setTab('classifier')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${tab === 'classifier' ? 'bg-violet-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              2. Classifier Training
            </button>
          </div>

          {tab === 'pretrain' && (
            <div className="space-y-4">
              <div className="p-3 rounded-xl text-xs text-slate-400"
                   style={{ background: 'rgba(124,58,237,0.08)', border: '1px solid rgba(124,58,237,0.15)' }}>
                Trains the ViT feature extractor using student-teacher self-distillation (L_CLS + L_MIM)
                on unlabeled images. No labels needed. Reference: 800 epochs.
              </div>
              {(Object.keys(defaultPretrainConfig) as (keyof typeof defaultPretrainConfig)[]).map(k => (
                <div key={k}>
                  <label className="block text-xs font-medium text-slate-400 mb-1 capitalize">{k.replace('_', ' ')}</label>
                  <input
                    type={typeof pretrainCfg[k] === 'number' ? 'number' : 'text'}
                    value={String(pretrainCfg[k])}
                    onChange={e => setPretrainCfg(prev => ({
                      ...prev, [k]: typeof prev[k] === 'number' ? Number(e.target.value) : e.target.value
                    }))}
                    className="input-field text-sm"
                  />
                </div>
              ))}
              <button onClick={startPretraining} disabled={loading} className="btn-primary w-full justify-center">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                Start Pretraining
              </button>
            </div>
          )}

          {tab === 'classifier' && (
            <div className="space-y-4">
              <div className="p-3 rounded-xl text-xs text-slate-400"
                   style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)' }}>
                Trains the GCN + Graph Transformer classifier on labeled real/fake data.
                Requires dataset organized as: train/real/, train/fake/, val/real/, val/fake/.
              </div>
              {(Object.keys(defaultClassifierConfig) as (keyof typeof defaultClassifierConfig)[]).map(k => (
                <div key={k}>
                  <label className="block text-xs font-medium text-slate-400 mb-1 capitalize">{k.replace('_', ' ')}</label>
                  {typeof classifierCfg[k] === 'boolean' ? (
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={classifierCfg[k] as boolean}
                        onChange={e => setClassifierCfg(prev => ({ ...prev, [k]: e.target.checked }))}
                        className="w-4 h-4" />
                      <span className="text-sm text-slate-300">Enabled</span>
                    </label>
                  ) : (
                    <input
                      type={typeof classifierCfg[k] === 'number' ? 'number' : 'text'}
                      value={String(classifierCfg[k])}
                      onChange={e => setClassifierCfg(prev => ({
                        ...prev, [k]: typeof prev[k] === 'number' ? Number(e.target.value) : e.target.value
                      }))}
                      className="input-field text-sm"
                    />
                  )}
                </div>
              ))}
              <button onClick={startClassifier} disabled={loading} className="btn-primary w-full justify-center">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                Start Classifier Training
              </button>
            </div>
          )}
        </div>
      )}

      {/* Job History */}
      {jobHistory.length > 0 && (
        <div className="glass-card overflow-hidden">
          <div className="p-4 border-b border-subtle">
            <h2 className="font-semibold text-white text-sm">Training History</h2>
          </div>
          <table className="data-table">
            <thead><tr><th>Type</th><th>Status</th><th>Progress</th><th>Best Val Loss</th></tr></thead>
            <tbody>
              {jobHistory.map(j => (
                <tr key={j.job_id}>
                  <td className="capitalize">{j.job_type}</td>
                  <td>
                    <span className={j.status === 'completed' ? 'text-emerald-400' : j.status === 'failed' ? 'text-red-400' : 'text-amber-400'}>
                      {j.status}
                    </span>
                  </td>
                  <td className="font-mono text-xs">{j.current_epoch}/{j.total_epochs}</td>
                  <td className="font-mono text-xs">{j.best_val_loss?.toFixed(4) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
