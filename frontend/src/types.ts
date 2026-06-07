export interface ApiDemoCase {
  case_id: string;
  name: string;
  mode: string;
  mode_label: string;
  description: string;
  explanation: string;
  drag_coefficient_estimate: number;
  lift_coefficient_estimate: number;
  wake_score: number;
  grid: { nx: number; ny: number; nz: number };
}

export interface ActiveCase {
  id: string;
  backendId: string;
  name: string;
  desc: string;
  drag: number;
  lift: number;
  wake: number;
  mode: string;
  modeLabel: string;
  explanation: string;
  isCustom: boolean;
}

export type DataSource = 'cached' | 'computed' | 'offline_fallback' | 'real_3d_lbm';

export const PROCESSING_STEPS = [
  'Upload file',
  'Detect object & scale',
  'Build simulation grid',
  'Ready for simulation',
] as const;

export const OFFLINE_PREVIEW =
  'data:image/svg+xml,' +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
      <rect width="320" height="180" fill="#0a0b10"/>
      <text x="160" y="78" fill="#64748b" font-family="sans-serif" font-size="13" text-anchor="middle">Offline Fallback</text>
      <text x="160" y="102" fill="#94a3b8" font-family="sans-serif" font-size="11" text-anchor="middle">Using cached sports car template</text>
      <rect x="100" y="115" width="120" height="40" rx="6" fill="none" stroke="#7c3aed" stroke-width="2"/>
    </svg>`
  );