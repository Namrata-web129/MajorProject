import { Link } from 'react-router-dom';
import {
  AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2, Image, ShieldAlert,
} from 'lucide-react';

const demoArtwork = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 400">
    <defs>
      <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
        <stop stop-color="#101b38"/>
        <stop offset="1" stop-color="#21143c"/>
      </linearGradient>
      <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
        <path d="M32 0H0V32" fill="none" stroke="#8b5cf6" stroke-opacity=".22"/>
      </pattern>
    </defs>
    <rect width="640" height="400" rx="24" fill="url(#bg)"/>
    <rect width="640" height="400" rx="24" fill="url(#grid)"/>
    <circle cx="320" cy="170" r="92" fill="#334477" fill-opacity=".72"/>
    <path d="M148 370c18-93 77-139 172-139s154 46 172 139" fill="#334477" fill-opacity=".72"/>
    <circle cx="286" cy="160" r="7" fill="#c4b5fd"/>
    <circle cx="354" cy="160" r="7" fill="#c4b5fd"/>
    <path d="M294 204c18 13 34 13 52 0" fill="none" stroke="#c4b5fd" stroke-width="5" stroke-linecap="round"/>
    <text x="320" y="350" fill="#ddd6fe" font-family="Arial,sans-serif" font-size="16" text-anchor="middle">
      SYNTHETIC ILLUSTRATION — NOT A REAL FACE
    </text>
  </svg>
`)}`;

const workflow = [
  'Upload and validation flow',
  'Face preprocessing flow',
  'Patch and graph pipeline overview',
  'Result and explainability layout',
];

export default function DemoPage() {
  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      <Link to="/detect" className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white">
        <ArrowLeft size={16} /> Back to Detection
      </Link>

      <div
        className="glass-card p-5 flex items-start gap-3"
        style={{ border: '1px solid rgba(245,158,11,0.4)', background: 'rgba(245,158,11,0.06)' }}
        role="alert"
      >
        <AlertTriangle size={20} className="text-amber-400 mt-0.5 flex-shrink-0" />
        <div>
          <h1 className="text-lg font-bold text-amber-200">Demo only — not a real analysis</h1>
          <p className="text-sm text-slate-300 mt-1">
            This page uses a generated illustration to preview the interface. No model was run, no media was
            analyzed, and this demo does not produce a REAL/FAKE prediction or confidence score.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <section className="glass-card p-5 space-y-4">
          <div className="flex items-center gap-3">
            <Image size={20} className="text-violet-400" />
            <div>
              <h2 className="font-semibold text-white">Synthetic sample</h2>
              <p className="text-xs text-slate-500">Generated SVG illustration · not a face or dataset item</p>
            </div>
          </div>
          <img
            src={demoArtwork}
            alt="Generated abstract illustration marked as synthetic and not a real face"
            className="w-full rounded-xl border border-subtle"
          />
          <div className="rounded-xl p-4 flex items-center gap-3"
               style={{ background: 'rgba(124,58,237,0.1)', border: '1px solid rgba(124,58,237,0.2)' }}>
            <ShieldAlert size={20} className="text-violet-300" />
            <div>
              <p className="text-sm font-semibold text-white">Model result: Not available</p>
              <p className="text-xs text-slate-400">A trained checkpoint and authorized dataset are required.</p>
            </div>
          </div>
        </section>

        <section className="glass-card p-5 space-y-4">
          <div>
            <h2 className="font-semibold text-white">What this demo previews</h2>
            <p className="text-sm text-slate-400 mt-1">These are interface sections only—not completed AI processing.</p>
          </div>
          <ol className="space-y-3">
            {workflow.map((step, index) => (
              <li key={step} className="flex items-center gap-3 rounded-xl p-3"
                  style={{ background: 'rgba(124,58,237,0.07)' }}>
                <CheckCircle2 size={18} className="text-violet-400 flex-shrink-0" />
                <span className="text-sm text-slate-300">{String(index + 1).padStart(2, '0')} · {step}</span>
              </li>
            ))}
          </ol>
          <div className="rounded-xl border border-subtle p-4 space-y-2">
            <p className="text-xs uppercase tracking-wider text-slate-500">Prediction</p>
            <p className="text-lg font-bold text-slate-300">None — demo mode</p>
            <p className="text-xs text-slate-500">Confidence, heatmap, and evaluation metrics are intentionally omitted.</p>
          </div>
          <div className="flex flex-wrap gap-3 pt-2">
            <Link to="/detect" className="btn-secondary text-sm">Try real upload</Link>
            <Link to="/dashboard" className="btn-primary text-sm">
              Dashboard <ArrowRight size={16} />
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
}
