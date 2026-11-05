import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import type { LucideIcon } from 'lucide-react';
import {
  LayoutDashboard, Search, Activity, Brain, Cpu,
  BarChart3, History, LogOut, Shield
} from 'lucide-react';

interface NavItem {
  to: string;
  icon: LucideIcon;
  label: string;
  adminOnly?: boolean;
}

const navItems: NavItem[] = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/detect', icon: Search, label: 'Detect' },
  { to: '/explainability', icon: Brain, label: 'Explainability' },
  { to: '/history', icon: History, label: 'History' },
  { to: '/model', icon: Cpu, label: 'Model Info' },
  { to: '/training', icon: Activity, label: 'Training', adminOnly: true },
  { to: '/evaluation', icon: BarChart3, label: 'Evaluation', adminOnly: true },
];

export function Sidebar() {
  const { user, isAdmin, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const visibleItems = navItems.filter(item => !item.adminOnly || isAdmin);

  return (
    <div className="sidebar w-64 flex-shrink-0">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-subtle">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg, #7c3aed, #3b82f6)' }}>
            <Shield size={18} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white">DeepFake AI</p>
            <p className="text-xs text-slate-500">Detection System</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto no-scrollbar">
        {/* User section */}
        {isAdmin && (
          <div className="mb-4">
            <p className="px-4 text-xs font-semibold uppercase tracking-widest text-slate-600 mb-2">
              Admin Controls
            </p>
          </div>
        )}

        {visibleItems.map(({ to, icon: Icon, label, adminOnly }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `nav-item ${isActive ? 'active' : ''} ${adminOnly ? 'border-l-violet-400' : ''}`
            }
          >
            <Icon size={18} />
            <span>{label}</span>
            {adminOnly && (
              <span className="ml-auto text-xs px-1.5 py-0.5 rounded bg-violet-900/40 text-violet-400 border border-violet-700/30">
                Admin
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* User Info + Logout */}
      <div className="px-3 py-4 border-t border-subtle">
        <div className="glass-card p-3 flex items-center gap-3 mb-3">
          <div className="w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold text-white"
               style={{ background: 'linear-gradient(135deg, #7c3aed, #3b82f6)' }}>
            {user?.username?.charAt(0).toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white truncate">{user?.username}</p>
            <p className="text-xs text-slate-500 truncate">{user?.email}</p>
          </div>
          {isAdmin && (
            <div className="w-5 h-5 rounded-full bg-violet-600/30 flex items-center justify-center">
              <Shield size={10} className="text-violet-400" />
            </div>
          )}
        </div>
        <button
          onClick={handleLogout}
          className="nav-item w-full text-red-400 hover:text-red-300 hover:bg-red-900/20"
        >
          <LogOut size={18} />
          <span>Logout</span>
        </button>
      </div>
    </div>
  );
}
