import { Upload, Wind, Info } from 'lucide-react';

interface LandingScreenProps {
  onTryDemo: () => void;
  onUpload: () => void;
}

export function LandingScreen({ onTryDemo, onUpload }: LandingScreenProps) {
  return (
    <div className="landing-screen">
      <div className="landing-content">
        <div className="landing-badge">Prototype · Educational Simulation</div>
        <h1 className="landing-title">AeroVoxel</h1>
        <p className="landing-pitch">
          Turn a smartphone photo or video into an interactive virtual wind tunnel.
          Visualize airflow, pressure, and wake behavior directly in your browser.
        </p>
        <p className="landing-honesty">
          AeroVoxel is a prototype-grade aerodynamic insight tool. It uses precomputed and
          lightweight 2D LBM fields — not certified CFD or wind-tunnel testing.
        </p>
        <div className="landing-actions">
          <button type="button" className="landing-btn primary" onClick={onTryDemo}>
            <Wind size={18} />
            Try Demo Object
          </button>
          <button type="button" className="landing-btn secondary" onClick={onUpload}>
            <Upload size={18} />
            Upload Image / Video
          </button>
        </div>
        <div className="landing-features">
          <div className="landing-feature">
            <Info size={14} />
            <span>3 template cases with cached 2D flow fields</span>
          </div>
          <div className="landing-feature">
            <Info size={14} />
            <span>Upload pipeline with silhouette extraction</span>
          </div>
          <div className="landing-feature">
            <Info size={14} />
            <span>Live 2D LBM solver for custom shapes</span>
          </div>
        </div>
      </div>
    </div>
  );
}