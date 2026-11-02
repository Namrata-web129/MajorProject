import { Link } from 'react-router-dom';
import {
  Shield, Brain, Network, Zap, Eye, Lock,
  ArrowRight, ChevronRight, Github, ExternalLink
} from 'lucide-react';

const features = [
  {
    icon: Brain,
    title: 'Self-Supervised Vision Transformer',
    desc: 'Student-Teacher distillation with Masked Image Modeling extracts robust facial representations without labels.',
    color: 'text-violet-400',
    bg: 'rgba(124,58,237,0.1)',
  },
  {
    icon: Network,
    title: 'Graph Neural Network',
    desc: 'Spatial K-Nearest Neighbor graph construction connects facial patches. 3-layer GCN learns local manipulation cues.',
    color: 'text-blue-400',
    bg: 'rgba(59,130,246,0.1)',
  },
  {
    icon: Zap,
    title: 'Graph Transformer Classifier',
    desc: '3-block, 8-head Transformer with multi-head attention identifies global manipulation patterns across graph nodes.',
    color: 'text-cyan-400',
    bg: 'rgba(6,182,212,0.1)',
  },
  {
    icon: Eye,
    title: 'Explainable AI Heatmaps',
    desc: 'Gradient-weighted attention relevancy maps reveal exactly which facial regions influenced the model\'s decision.',
    color: 'text-emerald-400',
    bg: 'rgba(16,185,129,0.1)',
  },
  {
    icon: Lock,
    title: 'Robust Evaluation Suite',
    desc: 'Cross-dataset, cross-manipulation, and robustness testing against Gaussian noise, blur, JPEG compression, and more.',
    color: 'text-amber-400',
    bg: 'rgba(245,158,11,0.1)',
  },
  {
    icon: Shield,
    title: 'Multi-Dataset Support',
    desc: 'Designed for FaceForensics++, Celeb-DF, DFDC, WildDeepfake, DeeperForensics, and FaceShifter.',
    color: 'text-pink-400',
    bg: 'rgba(236,72,153,0.1)',
  },
];

const steps = [
  { n: '01', title: 'Upload Media', desc: 'Upload an image or video file for analysis' },
  { n: '02', title: 'Face Detection', desc: 'MTCNN detects and crops all facial regions' },
  { n: '03', title: 'ViT Feature Extraction', desc: 'Self-supervised ViT encodes 256 patch embeddings' },
  { n: '04', title: 'Graph Construction', desc: 'KNN spatial graph connects facial patches (K=8)' },
  { n: '05', title: 'GCN + Transformer', desc: '3-layer GCN + 3-block Graph Transformer classifies' },
  { n: '06', title: 'Result + Explanation', desc: 'Real/Fake verdict with attention heatmap overlay' },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>

      {/* ── Navigation ─────────────────────────────────────────────────── */}
      <nav className="fixed top-0 inset-x-0 z-50 flex items-center justify-between px-8 py-4"
           style={{ background: 'rgba(6,11,24,0.85)', backdropFilter: 'blur(16px)', borderBottom: '1px solid rgba(124,58,237,0.12)' }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg, #7c3aed, #3b82f6)' }}>
            <Shield size={16} className="text-white" />
          </div>
          <span className="font-bold text-white">DeepFake Detection AI</span>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/login" className="btn-secondary text-sm py-2 px-4">Login</Link>
          <Link to="/register" className="btn-primary text-sm py-2 px-4">Get Started</Link>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className="pt-28 pb-20 px-6 text-center relative overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[400px] rounded-full opacity-20"
               style={{ background: 'radial-gradient(ellipse, rgba(124,58,237,0.6) 0%, transparent 70%)', filter: 'blur(60px)' }} />
        </div>

        <div className="relative max-w-4xl mx-auto animate-fade-in-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm mb-6"
               style={{ background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(124,58,237,0.3)', color: '#a78bfa' }}>
            <Zap size={14} />
            IEEE Access 2024 Research Implementation
          </div>

          <h1 className="text-5xl md:text-6xl font-black tracking-tight mb-6 leading-tight">
            <span className="text-white">Self-Supervised</span>
            <br />
            <span className="gradient-text">Graph Transformer</span>
            <br />
            <span className="text-white">Deepfake Detection</span>
          </h1>

          <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            A research-grade deepfake detection platform implementing the complete methodology
            from Khormali &amp; Yuan (IEEE Access 2024) — Student-Teacher ViT pretraining,
            KNN graph construction, GCN, Graph Transformer classification, and gradient-weighted
            explainability heatmaps.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link to="/register" className="btn-primary text-base px-8 py-4">
              Start Analysis <ArrowRight size={18} />
            </Link>
            <Link to="/login" className="btn-secondary text-base px-8 py-4">
              Sign In <ChevronRight size={18} />
            </Link>
          </div>

          <div className="mt-6 flex items-center justify-center gap-6 text-sm text-slate-500">
            <span className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              Real Detection
            </span>
            <span className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              Fake Detection
            </span>
            <span className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              Explainability Heatmaps
            </span>
          </div>
        </div>
      </section>

      {/* ── Architecture Pipeline ─────────────────────────────────────── */}
      <section className="py-20 px-6" style={{ background: 'rgba(10,22,40,0.5)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-3">Detection Pipeline</h2>
            <p className="text-slate-400">Six-stage research architecture from upload to explainable prediction</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {steps.map((step, i) => (
              <div key={i} className="glass-card p-5 relative overflow-hidden">
                <div className="absolute top-3 right-3 text-4xl font-black opacity-10 text-violet-400">
                  {step.n}
                </div>
                <div className="text-xs font-bold text-violet-400 uppercase tracking-widest mb-2">{step.n}</div>
                <h3 className="text-sm font-bold text-white mb-1">{step.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────── */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-3">Research Methodology</h2>
            <p className="text-slate-400">Complete implementation of the IEEE Access 2024 paper</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f, i) => (
              <div key={i} className="glass-card p-6 group">
                <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 transition-all duration-300 group-hover:scale-110"
                     style={{ background: f.bg }}>
                  <f.icon size={22} className={f.color} />
                </div>
                <h3 className="text-sm font-bold text-white mb-2">{f.title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Model Config Reference ──────────────────────────────────── */}
      <section className="py-20 px-6" style={{ background: 'rgba(10,22,40,0.5)' }}>
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-3">Research Architecture Parameters</h2>
          <p className="text-slate-400 mb-10">Reference configuration from Khormali &amp; Yuan (IEEE Access 2024)</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: 'Image Size', value: '320×320' },
              { label: 'Patch Size', value: '20×20' },
              { label: 'Graph Nodes', value: '256' },
              { label: 'KNN K', value: '8' },
              { label: 'GCN Layers', value: '3' },
              { label: 'Transformer Blocks', value: '3' },
              { label: 'Attention Heads', value: '8' },
              { label: 'Embedding Dim', value: '256' },
            ].map((p, i) => (
              <div key={i} className="glass-card p-4 text-center">
                <p className="text-xl font-bold gradient-text">{p.value}</p>
                <p className="text-xs text-slate-400 mt-1">{p.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────── */}
      <section className="py-20 px-6 text-center">
        <div className="max-w-xl mx-auto">
          <h2 className="text-3xl font-bold text-white mb-4">Ready to detect deepfakes?</h2>
          <p className="text-slate-400 mb-8">
            Create an account to start analyzing images and videos with our
            research-grade AI pipeline.
          </p>
          <Link to="/register" className="btn-primary text-base px-10 py-4">
            Create Free Account <ArrowRight size={18} />
          </Link>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────── */}
      <footer className="py-8 px-6 text-center border-t border-subtle">
        <p className="text-xs text-slate-600">
          Based on: "Self-Supervised Graph Transformer for Deepfake Detection" — Khormali &amp; Yuan, IEEE Access 2024.
          Research implementation for educational and research purposes.
        </p>
      </footer>
    </div>
  );
}
