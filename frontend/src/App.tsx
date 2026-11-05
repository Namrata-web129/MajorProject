import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from '@/contexts/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { AppLayout } from '@/layouts/AppLayout';

import LandingPage from '@/pages/LandingPage';
import LoginPage from '@/pages/LoginPage';
import RegisterPage from '@/pages/RegisterPage';
import DashboardPage from '@/pages/DashboardPage';
import DetectPage from '@/pages/DetectPage';
import DemoPage from '@/pages/DemoPage';
import ResultsPage from '@/pages/ResultsPage';
import ExplainabilityPage from '@/pages/ExplainabilityPage';
import HistoryPage from '@/pages/HistoryPage';
import ModelInfoPage from '@/pages/ModelInfoPage';
import TrainingPage from '@/pages/TrainingPage';
import EvaluationPage from '@/pages/EvaluationPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            {/* Public routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />

            {/* Protected app routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <AppLayout />
                </ProtectedRoute>
              }
            >
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="detect" element={<DetectPage />} />
              <Route path="demo" element={<DemoPage />} />
              <Route path="results/:id" element={<ResultsPage />} />
              <Route path="explainability/:id" element={<ExplainabilityPage />} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="model" element={<ModelInfoPage />} />
              <Route
                path="training"
                element={
                  <ProtectedRoute requireAdmin>
                    <TrainingPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="evaluation"
                element={
                  <ProtectedRoute requireAdmin>
                    <EvaluationPage />
                  </ProtectedRoute>
                }
              />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
