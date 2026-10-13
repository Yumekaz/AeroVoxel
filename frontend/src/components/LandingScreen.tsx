import { Upload, Wind, ShieldCheck, ArrowRight } from "lucide-react";
import { WindTunnelViewer } from "./WindTunnelViewer";
interface Props {
  onTryDemo: () => void;
  onUpload: () => void;
}
export function LandingScreen({ onTryDemo, onUpload }: Props) {
  return (
    <div className="landing-screen">
      <nav className="landing-nav">
        <div className="brand">
          <span className="logo-icon">AV</span>
          <strong>AeroVoxel</strong>
        </div>
        <span>Your browser. Your wind tunnel.</span>
      </nav>
      <main className="landing-layout">
        <div className="landing-content">
          <span className="landing-badge">THE VIRTUAL WIND TUNNEL</span>
          <h1 className="landing-title">
            Make the
            <br />
            invisible <em>flow.</em>
          </h1>
          <p className="landing-pitch">
            Explore how air moves around everyday shapes. Start with a demo, or
            turn a photograph into an educational airflow experiment.
          </p>
          <div className="landing-actions">
            <button className="primary-button" onClick={onTryDemo}>
              Open wind tunnel <ArrowRight size={17} />
            </button>
            <button className="secondary-button" onClick={onUpload}>
              <Upload size={16} />
              Upload capture
            </button>
          </div>
          <p className="landing-honesty">
            Educational prototype · CPU simulation · Optional ML drag proxy.
            Single-image geometry is approximate; results are not certified CFD.
          </p>
        </div>
        <div className="landing-visual">
          <WindTunnelViewer
            caseId="airfoil"
            windSpeed={15}
            windAngle={0}
            showStreamlines
            showPressure
            showWake={false}
            flowData={null}
            showVoxels={false}
            showSlicePlane={false}
            slicePosition={0}
            paused={
              window.matchMedia("(prefers-reduced-motion: reduce)").matches
            }
            cameraView="side"
          />
          <span className="landing-visual-caption">
            AIRFOIL · ILLUSTRATIVE FLOW PREVIEW
          </span>
        </div>
      </main>
      <footer className="landing-footnote">
        <span>
          <Wind size={15} />
          Interactive flow fields
        </span>
        <span>
          <Upload size={15} />
          Photo-to-silhouette pipeline
        </span>
        <span>
          <ShieldCheck size={15} />
          Transparent educational limits
        </span>
      </footer>
    </div>
  );
}
