import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { detectionService } from '@/services/api';
import type { AnalysisSummary } from '@/types';
import { History, CheckCircle, XCircle, Trash2, Eye, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';

function StatusBadge({ status, prediction }: { status: string; prediction?: string }) {
  if (status === 'completed' && prediction === 'REAL') return <span className="badge-real"><CheckCircle size={11} /> Real</span>;
  if (status === 'completed' && prediction === 'FAKE') return <span className="badge-fake"><XCircle size={11} /> Fake</span>;
  if (status === 'processing' || status === 'pending') return <span className="badge-processing"><Loader2 size={11} className="animate-spin" /> Processing</span>;
  if (status === 'failed') return <span style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }} className="badge-fake"><XCircle size={11} /> Failed</span>;
  return <span className="badge-pending">{status}</span>;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [hasNext, setHasNext] = useState(false);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const PAGE_SIZE = 15;

  const loadHistory = async (p: number) => {
    setLoading(true);
    try {
      const data = await detectionService.getHistory(p, PAGE_SIZE);
      setHistory(data.items);
      setTotal(data.total);
      setHasNext(data.has_next);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadHistory(page); }, [page]);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this analysis record?')) return;
    setDeleting(id);
    try {
      await detectionService.deleteAnalysis(id);
      setHistory(prev => prev.filter(h => h.id !== id));
      setTotal(prev => prev - 1);
    } catch {}
    setDeleting(null);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="section-title flex items-center gap-3">
            <History size={24} className="text-violet-400" /> Analysis History
          </h1>
          <p className="section-subtitle">{total} total analyses</p>
        </div>
        <Link to="/detect" className="btn-primary text-sm">New Analysis</Link>
      </div>

      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="p-12 text-center">
            <Loader2 size={32} className="text-violet-400 animate-spin mx-auto" />
          </div>
        ) : history.length === 0 ? (
          <div className="p-16 text-center">
            <History size={48} className="text-slate-700 mx-auto mb-4" />
            <p className="text-slate-400 font-medium">No analyses yet</p>
            <p className="text-slate-600 text-sm mt-1">Upload an image or video to begin detection</p>
            <Link to="/detect" className="btn-primary text-sm mt-5 inline-flex">Start Detecting</Link>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Type</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Faces</th>
                <th>Duration</th>
                <th>Date</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.map(item => (
                <tr key={item.id}>
                  <td className="max-w-[160px]">
                    <span className="text-white text-sm font-medium truncate block" title={item.file_name}>
                      {item.file_name || '—'}
                    </span>
                  </td>
                  <td className="capitalize text-xs">{item.file_type || '—'}</td>
                  <td><StatusBadge status={item.status} prediction={item.prediction} /></td>
                  <td className="font-mono text-xs">
                    {item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td className="text-center">{item.face_count ?? '—'}</td>
                  <td className="font-mono text-xs">
                    {item.processing_duration_ms != null
                      ? `${(item.processing_duration_ms / 1000).toFixed(1)}s`
                      : '—'}
                  </td>
                  <td className="text-xs whitespace-nowrap">
                    {new Date(item.created_at).toLocaleDateString()}<br />
                    <span className="text-slate-600">{new Date(item.created_at).toLocaleTimeString()}</span>
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <Link to={`/results/${item.id}`}
                        className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                        style={{ background: 'rgba(124,58,237,0.15)' }}
                        title="View result">
                        <Eye size={14} className="text-violet-400" />
                      </Link>
                      <button
                        onClick={() => handleDelete(item.id)}
                        disabled={deleting === item.id}
                        className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                        style={{ background: 'rgba(239,68,68,0.1)' }}
                        title="Delete">
                        {deleting === item.id
                          ? <Loader2 size={14} className="animate-spin text-red-400" />
                          : <Trash2 size={14} className="text-red-400" />}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Pagination */}
        {total > PAGE_SIZE && (
          <div className="flex items-center justify-between px-5 py-4 border-t border-subtle">
            <p className="text-xs text-slate-500">
              Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, total)} of {total}
            </p>
            <div className="flex gap-2">
              <button onClick={() => setPage(p => p - 1)} disabled={page === 1}
                className="btn-secondary text-xs py-1.5 px-3 disabled:opacity-40">
                <ChevronLeft size={14} /> Previous
              </button>
              <button onClick={() => setPage(p => p + 1)} disabled={!hasNext}
                className="btn-secondary text-xs py-1.5 px-3 disabled:opacity-40">
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
