import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Shield, Eye, EyeOff, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';

function PasswordStrength({ password }: { password: string }) {
  const checks = [
    { label: 'At least 8 characters', ok: password.length >= 8 },
    { label: 'Contains uppercase letter', ok: /[A-Z]/.test(password) },
    { label: 'Contains a digit', ok: /\d/.test(password) },
  ];
  if (!password) return null;
  return (
    <ul className="mt-2 space-y-1">
      {checks.map((c, i) => (
        <li key={i} className={`flex items-center gap-2 text-xs ${c.ok ? 'text-emerald-400' : 'text-slate-500'}`}>
          <CheckCircle2 size={12} className={c.ok ? 'text-emerald-400' : 'text-slate-600'} />
          {c.label}
        </li>
      ))}
    </ul>
  );
}

export default function RegisterPage() {
  const [form, setForm] = useState({ email: '', username: '', password: '', full_name: '' });
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const { register, error, clearError } = useAuth();
  const navigate = useNavigate();

  const update = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(prev => ({ ...prev, [k]: e.target.value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    setLoading(true);
    try {
      await register(form);
      navigate('/dashboard', { replace: true });
    } catch {
      // error in context
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-10 relative overflow-hidden"
         style={{ background: 'var(--bg-primary)' }}>
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full opacity-10"
             style={{ background: 'radial-gradient(ellipse, rgba(59,130,246,0.8) 0%, transparent 70%)', filter: 'blur(60px)' }} />
      </div>

      <div className="w-full max-w-md relative animate-fade-in-up">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4"
               style={{ background: 'linear-gradient(135deg, #7c3aed, #3b82f6)', boxShadow: '0 0 30px rgba(59,130,246,0.4)' }}>
            <Shield size={26} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Create account</h1>
          <p className="text-slate-400 text-sm mt-1">Join the DeepFake Detection System</p>
        </div>

        <div className="glass-card p-8">
          {error && (
            <div className="flex items-start gap-3 p-4 rounded-xl mb-6 text-sm"
                 style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)', color: '#fca5a5' }}>
              <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Full Name (optional)</label>
              <input type="text" value={form.full_name} onChange={update('full_name')}
                placeholder="John Doe" className="input-field" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Email *</label>
              <input id="reg-email" type="email" value={form.email} onChange={update('email')}
                placeholder="you@example.com" className="input-field" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Username *</label>
              <input id="reg-username" type="text" value={form.username} onChange={update('username')}
                placeholder="johndoe" className="input-field" required minLength={3} maxLength={50} />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Password *</label>
              <div className="relative">
                <input id="reg-password" type={showPw ? 'text' : 'password'} value={form.password} onChange={update('password')}
                  placeholder="••••••••" className="input-field pr-11" required minLength={8} />
                <button type="button" onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300">
                  {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
              <PasswordStrength password={form.password} />
            </div>

            <button id="register-btn" type="submit" disabled={loading}
              className="btn-primary w-full justify-center py-3.5 text-base mt-2">
              {loading ? <><Loader2 size={18} className="animate-spin" /> Creating account...</> : 'Create Account'}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-subtle text-center">
            <p className="text-sm text-slate-400">
              Already have an account?{' '}
              <Link to="/login" className="text-violet-400 hover:text-violet-300 font-medium transition-colors">
                Sign in
              </Link>
            </p>
          </div>
        </div>
        <p className="text-center text-xs text-slate-600 mt-6">
          <Link to="/" className="hover:text-slate-400 transition-colors">← Back to home</Link>
        </p>
      </div>
    </div>
  );
}
