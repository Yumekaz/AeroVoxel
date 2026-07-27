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
          Educational virtual wind tunnel from smartphone imagery using lightweight
          Lattice Boltzmann simulation. Explore airflow, pressure, and wake behavior in the browser.
        </p>
        <p className="landing-honesty">
          Prototype-grade and educational only. Uses OpenCV silhouettes, template matching, and
          CPU Lattice Boltzmann (live 2D + cached demos) — not certified CFD, not AI 3D reconstruction.
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
            <span>Demo cases including circular cylinder validation</span>
          </div>
          <div className="landing-feature">
            <Info size={14} />
            <span>Upload via OpenCV silhouette extraction</span>
          </div>
          <div className="landing-feature">
            <Info size={14} />
            <span>Live 2D LBM on CPU + cached educational fields</span>
          </div>
        </div>
      </div>
    </div>
  );
}