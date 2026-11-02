import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { detectionService, modelService } from '@/services/api';
import type { AnalysisSummary, ModelStatus } from '@/types';
import {
  Search, Activity, CheckCircle, XCircle, Clock,
  TrendingUp, Upload, ChevronRight, Cpu, AlertTriangle
} from 'lucide-react';

function StatCard({ icon: Icon, label, value, color, trend }: {
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string;
  value: string | number;
  color: string;
  trend?: string;
}) {
  return (
    <div className="stat-card">
      <div className="flex items-center justify-between">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center`}
             style={{ background: `${color}20` }}>
          <Icon size={20} style={{ color }} />
        </div>
        {trend && <span className="text-xs text-emerald-400">{trend}</span>}
      </div>
      <div>
        <p className="text-3xl font-bold text-white">{value}</p>
        <p className="text-sm text-slate-400">{label}</p>
      </div>
    </div>
  );
}

function PredictionBadge({ prediction }: { prediction?: string }) {
  if (!prediction) return <span className="badge-pending">Pending</span>;
  return prediction === 'REAL'
    ? <span className="badge-real"><CheckCircle size={12} />Real</span>
    : <span className="badge-fake"><XCircle size={12} />Fake</span>;
}

export default function DashboardPage() {
  const { user, isAdmin } = useAuth();
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [hist, status] = await Promise.allSettled([
          detectionService.getHistory(1, 5),
          modelService.getStatus(),
        ]);
        if (hist.status === 'fulfilled') setHistory(hist.value.items);
        if (status.status === 'fulfilled') setModelStatus(status.value);
      } finally {
        setLoadingHistory(false);
      }
    })();
  }, []);

  const total = history.length;
  const realCount = history.filter(h => h.prediction === 'REAL').length;
  const fakeCount = history.filter(h => h.prediction === 'FAKE').length;
  const processing = history.filter(h => h.status === 'processing' || h.status === 'pending').length;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="section-title">Dashboard</h1>
          <p className="section-subtitle">
            Welcome back, <span className="text-violet-400">{user?.username}</span>
          </p>
        </div>
        <Link to="/detect" className="btn-primary text-sm">
          <Upload size={16} /> New Analysis
        </Link>
      </div>

      {/* Model Status Banner */}
      {modelStatus && !modelStatus.model_loaded && (
        <div className="glass-card p-4 flex items-center gap-3"
             style={{ border: '1px solid rgba(245,158,11,0.3)', background: 'rgba(245,158,11,0.05)' }}>
          <AlertTriangle size={18} className="text-amber-400 flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-amber-300">Model Not Trained</p>
            <p className="text-xs text-slate-400 mt-0.5">
              No trained model checkpoint is available. Inference will fail until the model is trained.
              {isAdmin && ' Go to Training to start the training pipeline.'}
            </p>
          </div>
          {isAdmin && (
            <Link to="/training" className="btn-secondary text-xs py-2 px-3">
              Train Model <ChevronRight size={14} />
            </Link>
          )}
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Search} label="Total Analyses" value={total} color="#7c3aed" />
        <StatCard icon={CheckCircle} label="Real Detections" value={realCount} color="#10b981" />
        <StatCard icon={XCircle} label="Fake Detections" value={fakeCount} color="#ef4444" />
        <StatCard icon={Clock} label="Processing" value={processing} color="#f59e0b" />
      </div>

      {/* Model Status Card */}
      {modelStatus && (
        <div className="glass-card p-5">
          <div className="flex items-center gap-3 mb-4">
            <Cpu size={18} className="text-violet-400" />
            <h2 className="font-semibold text-white">Model Status</h2>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-slate-500 mb-1">Status</p>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${modelStatus.model_loaded ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                <span className="text-sm font-medium text-white capitalize">
                  {modelStatus.training_status}
                </span>
              </div>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Device</p>
              <p className="text-sm font-medium text-white uppercase">{modelStatus.device}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Checkpoint</p>
              <span className={`text-sm font-medium ${modelStatus.checkpoint_exists ? 'text-emerald-400' : 'text-red-400'}`}>
                {modelStatus.checkpoint_exists ? 'Available' : 'Missing'}
              </span>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Pretrain</p>
              <span className={`text-sm font-medium ${modelStatus.pretrain_checkpoint_exists ? 'text-emerald-400' : 'text-slate-500'}`}>
                {modelStatus.pretrain_checkpoint_exists ? 'Available' : 'Not done'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Recent History */}
      <div className="glass-card overflow-hidden">
        <div className="flex items-center justify-between p-5 border-b border-subtle">
          <div className="flex items-center gap-3">
            <Activity size={18} className="text-violet-400" />
            <h2 className="font-semibold text-white">Recent Analyses</h2>
          </div>
          <Link to="/history" className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1">
            View all <ChevronRight size={14} />
          </Link>
        </div>
        {loadingHistory ? (
          <div className="p-8 text-center">
            <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto" />
          </div>
        ) : history.length === 0 ? (
          <div className="p-12 text-center">
            <Search size={40} className="text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400 font-medium">No analyses yet</p>
            <p className="text-slate-600 text-sm mt-1">Upload an image or video to get started</p>
            <Link to="/detect" className="btn-primary text-sm mt-4 inline-flex">
              Start Detecting
            </Link>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Type</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Date</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {history.map(item => (
                <tr key={item.id}>
                  <td className="text-white font-medium truncate max-w-[150px]">{item.file_name || '—'}</td>
                  <td className="capitalize">{item.file_type || '—'}</td>
                  <td><PredictionBadge prediction={item.prediction} /></td>
                  <td>
                    {item.confidence != null
                      ? <span className="font-mono text-xs">{(item.confidence * 100).toFixed(1)}%</span>
                      : '—'}
                  </td>
                  <td className="text-xs">
                    {new Date(item.created_at).toLocaleDateString()}
                  </td>
                  <td>
                    <Link to={`/results/${item.id}`}
                      className="text-xs text-violet-400 hover:text-violet-300 flex items-center gap-1">
                      View <ChevronRight size={12} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
