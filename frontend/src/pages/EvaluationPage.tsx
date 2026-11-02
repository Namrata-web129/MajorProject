import { useEffect, useState } from 'react';
import { evaluationService, modelService, getErrorMessage } from '@/services/api';
import type { EvaluationResult, ModelStatus } from '@/types';
import { BarChart3, Play, Loader2, AlertTriangle, Lock } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, Radar
} from 'recharts';

export default function EvaluationPage() {
  const { isAdmin } = useAuth();
  const [results, setResults] = useState<EvaluationResult[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState({
    model_version_id: '',
    eval_type: 'in_dataset',
    dataset_path: './datasets/test',
    dataset_name: 'FaceForensics++',
  });

  useEffect(() => {
    evaluationService.getResults().then(setResults).catch(() => {});
    modelService.getStatus().then(s => {
      setModelStatus(s);
      if (s.active_model_id) setConfig(c => ({ ...c, model_version_id: s.active_model_id! }));
    }).catch(() => {});
  }, []);

  const runEval = async () => {
    setError(null);
    setRunning(true);
    try {
      await evaluationService.runEvaluation(config);
      setTimeout(async () => {
        const r = await evaluationService.getResults();
        setResults(r);
        setRunning(false);
      }, 3000);
    } catch (e) {
      setError(getErrorMessage(e));
      setRunning(false);
    }
  };

  const latest = results[0];
  const barData = results.slice(0, 5).map(r => ({
    name: r.dataset_name.slice(0, 12),
    accuracy: r.accuracy ? +(r.accuracy * 100).toFixed(1) : 0,
    f1: r.f1_score ? +(r.f1_score * 100).toFixed(1) : 0,
    auc: r.auc_roc ? +(r.auc_roc * 100).toFixed(1) : 0,
  }));

  const radarData = latest ? [
    { metric: 'Accuracy', value: latest.accuracy ? +(latest.accuracy * 100).toFixed(1) : 0 },
    { metric: 'Precision', value: latest.precision ? +(latest.precision * 100).toFixed(1) : 0 },
    { metric: 'Recall', value: latest.recall ? +(latest.recall * 100).toFixed(1) : 0 },
    { metric: 'F1', value: latest.f1_score ? +(latest.f1_score * 100).toFixed(1) : 0 },
    { metric: 'AUC', value: latest.auc_roc ? +(latest.auc_roc * 100).toFixed(1) : 0 },
  ] : [];

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="section-title flex items-center gap-3">
          <BarChart3 size={24} className="text-violet-400" /> Model Evaluation
        </h1>
        <p className="section-subtitle">Performance metrics computed from actual model inference on test datasets</p>
      </div>

      {/* Important Disclaimer */}
      <div className="p-4 rounded-xl text-xs text-slate-400"
           style={{ background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.2)' }}>
        <AlertTriangle size={12} className="inline mr-1.5 text-amber-400" />
        All metrics displayed are computed by running the model on real test data. No results are
        hardcoded or taken from the research paper. If no model is trained, evaluation cannot run.
      </div>

      {/* Admin: Run Evaluation */}
      {isAdmin && (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-white mb-4">Run Evaluation</h2>
          {error && (
            <div className="p-3 rounded-xl mb-4 text-xs text-red-300"
                 style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}>
              {error}
            </div>
          )}
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Evaluation Type</label>
              <select
                value={config.eval_type}
                onChange={e => setConfig(c => ({ ...c, eval_type: e.target.value }))}
                className="input-field text-sm"
              >
                <option value="in_dataset">In-Dataset</option>
                <option value="cross_dataset">Cross-Dataset</option>
                <option value="robustness">Robustness</option>
                <option value="ablation">Ablation Study</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Dataset Name</label>
              <input value={config.dataset_name}
                onChange={e => setConfig(c => ({ ...c, dataset_name: e.target.value }))}
                className="input-field text-sm" />
            </div>
            <div className="md:col-span-2">
              <label className="block text-xs text-slate-400 mb-1">Dataset Path</label>
              <input value={config.dataset_path}
                onChange={e => setConfig(c => ({ ...c, dataset_path: e.target.value }))}
                className="input-field text-sm font-mono" />
            </div>
          </div>
          <button onClick={runEval} disabled={running || !modelStatus?.model_loaded} className="btn-primary">
            {running ? <><Loader2 size={16} className="animate-spin" /> Running...</> : <><Play size={16} /> Run Evaluation</>}
          </button>
          {!modelStatus?.model_loaded && (
            <p className="text-xs text-amber-400 mt-2">
              ⚠ Model checkpoint not available. Train the model before running evaluation.
            </p>
          )}
        </div>
      )}

      {/* Results */}
      {results.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <BarChart3 size={40} className="text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No evaluation results yet.</p>
          <p className="text-slate-600 text-sm mt-1">Train a model and run evaluation to see metrics.</p>
        </div>
      ) : (
        <>
          {/* Latest metrics */}
          {latest && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {[
                { label: 'Accuracy', value: latest.accuracy },
                { label: 'Precision', value: latest.precision },
                { label: 'Recall', value: latest.recall },
                { label: 'F1 Score', value: latest.f1_score },
                { label: 'AUC-ROC', value: latest.auc_roc },
              ].map(({ label, value }) => (
                <div key={label} className="glass-card p-4 text-center">
                  <p className="text-2xl font-bold gradient-text">
                    {value != null ? `${(value * 100).toFixed(1)}%` : '—'}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">{label}</p>
                </div>
              ))}
            </div>
          )}

          {/* Radar Chart */}
          {radarData.length > 0 && (
            <div className="glass-card p-5">
              <h2 className="font-semibold text-white mb-4 text-sm">Performance Radar ({latest.dataset_name})</h2>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(124,58,237,0.2)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: '#94a3b8' }} />
                  <Radar name="Score" dataKey="value" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.25} />
                  <Tooltip
                    contentStyle={{ background: '#0a1628', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 8 }}
                    formatter={(v: number) => [`${v}%`, '']}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Bar Chart: cross-dataset */}
          {barData.length > 1 && (
            <div className="glass-card p-5">
              <h2 className="font-semibold text-white mb-4 text-sm">Cross-Dataset Performance</h2>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(124,58,237,0.1)" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: '#0a1628', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 8 }}
                  />
                  <Bar dataKey="accuracy" fill="#7c3aed" radius={[4, 4, 0, 0]} name="Accuracy %" />
                  <Bar dataKey="f1" fill="#3b82f6" radius={[4, 4, 0, 0]} name="F1 %" />
                  <Bar dataKey="auc" fill="#06b6d4" radius={[4, 4, 0, 0]} name="AUC %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* History table */}
          <div className="glass-card overflow-hidden">
            <div className="p-4 border-b border-subtle">
              <h2 className="font-semibold text-white text-sm">Evaluation History</h2>
            </div>
            <table className="data-table">
              <thead><tr><th>Dataset</th><th>Type</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>Date</th></tr></thead>
              <tbody>
                {results.map(r => (
                  <tr key={r.id}>
                    <td className="font-medium">{r.dataset_name}</td>
                    <td className="capitalize text-xs">{r.eval_type}</td>
                    <td className="font-mono text-xs">{r.accuracy ? `${(r.accuracy * 100).toFixed(1)}%` : '—'}</td>
                    <td className="font-mono text-xs">{r.f1_score ? `${(r.f1_score * 100).toFixed(1)}%` : '—'}</td>
                    <td className="font-mono text-xs">{r.auc_roc ? `${(r.auc_roc * 100).toFixed(1)}%` : '—'}</td>
                    <td className="text-xs">{new Date(r.evaluated_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
