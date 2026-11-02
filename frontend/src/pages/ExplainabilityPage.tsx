import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { explainabilityService, getErrorMessage } from '@/services/api';
import type { ExplainabilityData } from '@/types';
import { Brain, Eye, Network, Info, ChevronLeft, Loader2, AlertTriangle } from 'lucide-react';

export default function ExplainabilityPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<ExplainabilityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    explainabilityService.getExplainability(id)
      .then(setData)
      .catch(e => setError(getErrorMessage(e)))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 size={36} className="text-violet-400 animate-spin" />
    </div>
  );

  if (error) return (
    <div className="max-w-2xl mx-auto text-center py-20">
      <AlertTriangle size={40} className="text-amber-400 mx-auto mb-4" />
      <p className="text-slate-300">{error}</p>
      <Link to="/history" className="btn-secondary text-sm mt-4 inline-flex">Back</Link>
    </div>
  );

  if (!data) return null;

  const isFake = data.prediction === 'FAKE';
  const gridSize = 16;

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <Link to={`/results/${id}`} className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white">
        <ChevronLeft size={16} /> Back to Results
      </Link>

      <div>
        <h1 className="section-title flex items-center gap-3">
          <Brain size={24} className="text-violet-400" /> Explainable AI Analysis
        </h1>
        <p className="section-subtitle">
          Verdict: <span className={isFake ? 'text-red-400 font-semibold' : 'text-emerald-400 font-semibold'}>{data.prediction}</span>
          {' '}· {(data.confidence * 100).toFixed(1)}% confidence
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Heatmap */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Eye size={16} className="text-violet-400" />
            <h2 className="font-semibold text-white text-sm">Gradient-Weighted Attention Heatmap</h2>
          </div>
          {data.heatmap_base64 ? (
            <img
              src={`data:image/jpeg;base64,${data.heatmap_base64}`}
              alt="Attention heatmap"
              className="w-full rounded-xl"
            />
          ) : (
            <div className="h-48 rounded-xl flex items-center justify-center"
                 style={{ background: 'rgba(124,58,237,0.05)', border: '1px dashed rgba(124,58,237,0.2)' }}>
              <p className="text-slate-500 text-sm">Heatmap not available</p>
            </div>
          )}
          <p className="text-xs text-slate-500 mt-3">
            Computed from actual Graph Transformer attention weights and gradients.
            Brighter regions indicate higher relevance.
          </p>
        </div>

        {/* Patch Relevance Grid */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Network size={16} className="text-blue-400" />
            <h2 className="font-semibold text-white text-sm">Patch Relevance Map (16×16)</h2>
          </div>
          {data.patch_relevance && data.patch_relevance.length === 256 ? (
            <div className="space-y-2">
              <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
                {data.patch_relevance.map((v, i) => {
                  const intensity = Math.min(1, Math.max(0, v));
                  const r = isFake ? 239 : 16;
                  const g = isFake ? 68 : 185;
                  const b = isFake ? 68 : 129;
                  return (
                    <div
                      key={i}
                      className="aspect-square rounded-sm cursor-help"
                      style={{ background: `rgba(${r},${g},${b}, ${intensity * 0.85 + 0.08})` }}
                      title={`Patch ${i}: relevance ${(intensity * 100).toFixed(1)}%`}
                    />
                  );
                })}
              </div>
              <div className="flex items-center justify-between text-xs text-slate-500 mt-1">
                <span>Low relevance</span>
                <div className="flex-1 mx-3 h-1.5 rounded-full"
                     style={{
                       background: `linear-gradient(90deg, rgba(${isFake ? '239,68,68' : '16,185,129'}, 0.1), rgba(${isFake ? '239,68,68' : '16,185,129'}, 0.9))`
                     }} />
                <span>High relevance</span>
              </div>
            </div>
          ) : (
            <div className="h-48 rounded-xl flex items-center justify-center"
                 style={{ background: 'rgba(59,130,246,0.05)', border: '1px dashed rgba(59,130,246,0.2)' }}>
              <p className="text-slate-500 text-sm">Patch relevance not available</p>
            </div>
          )}
        </div>
      </div>

      {/* Explanation Text */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-3">
          <Info size={16} className="text-cyan-400" />
          <h2 className="font-semibold text-white text-sm">Model Decision Explanation</h2>
        </div>
        <p className="text-slate-300 text-sm leading-relaxed">{data.explanation_text}</p>
        <div className="mt-4 p-3 rounded-xl text-xs"
             style={{ background: 'rgba(6,182,212,0.08)', border: '1px solid rgba(6,182,212,0.15)', color: '#94a3b8' }}>
          <strong className="text-cyan-400">Methodology:</strong> The relevancy map is computed using
          gradient-weighted Transformer attention propagation. Attention weights from all 3 Transformer blocks
          are combined with class-specific gradients, averaged across 8 attention heads, with identity
          connections added to prevent self-inhibition. The resulting node-level scores are mapped back
          to their 20×20 spatial positions on the 320×320 facial image.
        </div>
      </div>

      {/* Attention Weights Visualization */}
      {data.attention_weights && data.attention_weights.length > 0 && (
        <div className="glass-card p-5">
          <h2 className="font-semibold text-white text-sm mb-4">Transformer Attention Pattern</h2>
          <div className="overflow-x-auto">
            <div className="text-xs text-slate-500 mb-2">Attention matrix (first block, first head)</div>
            <div className="grid gap-0.5 w-fit" style={{ gridTemplateColumns: `repeat(${Math.min(32, data.attention_weights[0].length)}, 6px)` }}>
              {data.attention_weights[0].slice(0, 32).map((row, ri) =>
                row.slice(0, 32).map((v, ci) => (
                  <div key={`${ri}-${ci}`} className="w-1.5 h-1.5 rounded-sm"
                       style={{ background: `rgba(124,58,237, ${Math.min(1, v * 5)})` }} />
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
