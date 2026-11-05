import { StrictMode, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

const rootElement = document.getElementById('app');
if (!rootElement) {
  throw new Error('Required app root element was not found.');
}

createRoot(rootElement).render(
  createElement(StrictMode, null, createElement(App)),
);
