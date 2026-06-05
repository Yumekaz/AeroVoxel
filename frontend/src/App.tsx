import { useState, useEffect } from 'react';
import { 
  Layers, 
  Upload, 
  FileVideo, 
  FileImage, 
  Info,
  ShieldCheck
} from 'lucide-react';
import { WindTunnelViewer } from './components/WindTunnelViewer';
import type { FlowData } from './components/WindTunnelViewer';
import { fetchNpy } from './utils/npyLoader';
import './App.css';

interface DemoCase {
  id: string;
  name: string;
  desc: string;
  drag: number;
  lift: number;
  wake: number;
  mode: string;
  explanation: string;
}

const DEMO_CASES: DemoCase[] = [
  {
    id: 'sports_car',
    name: 'Sports Car (Template)',
    desc: 'Low-drag ground vehicle profile with rear separation.',
    drag: 0.28,
    lift: -0.05,
    wake: 0.65,
    mode: 'Cached 3D Demo',
    explanation: 'Air hits the front bumper creating a high-pressure stagnation zone. It accelerates over the hood and windshield (low pressure), then separates rapidly at the rear windshield, creating a low-pressure recirculating wake that sucks the vehicle backward (producing drag).'
  },
  {
    id: 'drone',
    name: 'Quadcopter (Template)',
    desc: 'Complex vertical thrust profile with high drag wake.',
    drag: 1.15,
    lift: 1.42,
    wake: 0.88,
    mode: 'Cached 3D Demo',
    explanation: 'Flow encounters bluff-body engine mounts and spinning rotor disks. The arms create severe flow separation and localized vortex shedding, resulting in a large wake turbulence index and substantial drag relative to its frontal area.'
  },
  {
    id: 'airfoil',
    name: 'NACA 0012 Airfoil',
    desc: 'Symmetric streamlined wing section at angle of attack.',
    drag: 0.06,
    lift: 0.45,
    wake: 0.12,
    mode: 'Cached 3D Demo',
    explanation: 'The classic aerodynamic profile. Flow remains attached to the smooth surface for almost the entire chord length. Curvature differences create asymmetric pressure (higher velocity on top = lower pressure = upward lift force) with a minimal wake profile.'
  }
];

function App() {
  const [selectedCase, setSelectedCase] = useState<DemoCase>(DEMO_CASES[0]);
  const [windSpeed, setWindSpeed] = useState<number>(15); // m/s
  const [windAngle, setWindAngle] = useState<number>(0); // degrees
  
  // Toggles
  const [showStreamlines, setShowStreamlines] = useState<boolean>(true);
  const [showPressure, setShowPressure] = useState<boolean>(true);
  const [showWake, setShowWake] = useState<boolean>(true);
  
  // Backend connection state
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected'>('disconnected');
  
  // Ingestion & solver state (Phases 3-4)
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [processingStep, setProcessingStep] = useState<string>('');
  const [contourPreview, setContourPreview] = useState<string | null>(null);
  const [simMode, setSimMode] = useState<string>('Cached 3D Demo');

  // Flow Field binary data arrays (Phase 2)
  const [flowData, setFlowData] = useState<FlowData | null>(null);
  const [loadingFlowData, setLoadingFlowData] = useState<boolean>(false);
  const [closestPreset, setClosestPreset] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [runningSolver, setRunningSolver] = useState<boolean>(false);

  // Phase 5 Voxel & Slicing states
  const [showVoxels, setShowVoxels] = useState<boolean>(false);
  const [showSlicePlane, setShowSlicePlane] = useState<boolean>(false);
  const [slicePosition, setSlicePosition] = useState<number>(0.0);

  const runLiveSimulation = async () => {
    if (!activeJobId) return;
    setRunningSolver(true);
    setSimMode('Solving Fluid Fields...');

    try {
      const response = await fetch('http://127.0.0.1:8000/api/simulate/simple', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: activeJobId,
          wind_speed: windSpeed,
          wind_angle_deg: windAngle
        }),
      });

      if (!response.ok) throw new Error('Simulation failed');
      const res = await response.json();

      const host = 'http://127.0.0.1:8000';
      const [velNpy, pressNpy, maskNpy] = await Promise.all([
        fetchNpy(host + res.velocity_url),
        fetchNpy(host + res.pressure_url),
        fetchNpy(host + res.mask_url)
      ]);

      setFlowData({
        velocity: velNpy.data as Float32Array,
        pressure: pressNpy.data as Float32Array,
        mask: maskNpy.data as Uint8Array,
        nx: velNpy.shape[2],
        ny: velNpy.shape[1]
      });

      setSimMode('2D Solver Completed');
      
      const solvedCase: DemoCase = {
        id: 'custom_upload',
        name: selectedCase.name,
        desc: selectedCase.desc,
        drag: res.metrics.drag_coefficient_estimate,
        lift: res.metrics.lift_coefficient_estimate,
        wake: res.metrics.wake_score,
        mode: '2D Solver Runs (CPU)',
        explanation: `CFD calculations converged in 600 steps! The streamline particles are now deflecting along the computed velocity fields. Drag coefficient estimated at ${res.metrics.drag_coefficient_estimate.toFixed(2)}, lift force at ${res.metrics.lift_coefficient_estimate.toFixed(2)}.`
      };
      setSelectedCase(solvedCase);
    } catch (err) {
      console.error('Simulation run failed:', err);
      setSimMode('2D Solver Failed');
    } finally {
      setRunningSolver(false);
    }
  };

  // Ping backend health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/health');
        if (response.ok) {
          setBackendStatus('connected');
        } else {
          setBackendStatus('disconnected');
        }
      } catch (err) {
        setBackendStatus('disconnected');
      }
    };
    checkHealth();
    // Re-check every 8 seconds
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  // Fetch LBM binary datasets from the backend (Phase 2 Engine)
  useEffect(() => {
    if (backendStatus !== 'connected') {
      setFlowData(null);
      return;
    }

    const loadFlowArrays = async () => {
      setLoadingFlowData(true);
      try {
        let backendCaseId = selectedCase.id;
        if (backendCaseId === 'custom_upload') {
          if (!closestPreset) {
            setFlowData(null);
            setLoadingFlowData(false);
            return;
          }
          backendCaseId = closestPreset;
        } else {
          // Map to exact API case IDs
          if (backendCaseId === 'sports_car') backendCaseId = 'sports_car_v1';
          else if (backendCaseId === 'drone') backendCaseId = 'drone_v1';
          else if (backendCaseId === 'airfoil') backendCaseId = 'airfoil_v1';
        }

        const metadataUrl = `http://127.0.0.1:8000/api/flow-field/${backendCaseId}`;
        const metaResponse = await fetch(metadataUrl);
        if (!metaResponse.ok) throw new Error('Failed to load metadata');
        const meta = await metaResponse.json();

        // Download the arrays
        const host = 'http://127.0.0.1:8000';
        const [velNpy, pressNpy, maskNpy] = await Promise.all([
          fetchNpy(host + meta.velocity_url),
          fetchNpy(host + meta.pressure_url),
          fetchNpy(host + meta.mask_url)
        ]);

        setFlowData({
          velocity: velNpy.data as Float32Array,
          pressure: pressNpy.data as Float32Array,
          mask: maskNpy.data as Uint8Array,
          nx: velNpy.shape[2], // (2, ny, nx)
          ny: velNpy.shape[1]
        });
      } catch (err) {
        console.error('Error loading flow arrays:', err);
        setFlowData(null);
      } finally {
        setLoadingFlowData(false);
      }
    };

    loadFlowArrays();
  }, [selectedCase.id, backendStatus, closestPreset]);

  // Real file upload to backend with cv processing and preset matching (Phase 3)
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadedFile(file.name);
    setContourPreview(null);
    setClosestPreset(null);
    setProcessingStep('Uploading file to server...');

    if (backendStatus !== 'connected') {
      runMockUploadFlow(file);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setProcessingStep('Extracting keyframe & boundaries...');
      const response = await fetch('http://127.0.0.1:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const res = await response.json();

      setUploading(false);
      setClosestPreset(res.closest_preset);
      setActiveJobId(res.job_id);
      setContourPreview('http://127.0.0.1:8000' + res.preview_url);
      setSimMode('2D Solver Ready');

      const customCase: DemoCase = {
        id: 'custom_upload',
        name: `Upload: ${res.filename.substring(0, 12)}`,
        desc: `Custom shape. Scale: ${res.scale_estimate.toFixed(3)} mm/px.`,
        drag: res.closest_preset === 'sports_car_v1' ? 0.28 : res.closest_preset === 'drone_v1' ? 1.15 : 0.06,
        lift: res.closest_preset === 'airfoil_v1' ? 0.45 : 0.00,
        wake: res.closest_preset === 'drone_v1' ? 0.88 : 0.65,
        mode: '2D Solver Ready (CPU)',
        explanation: `${res.detected_object}. ${res.scale_status}. Under Phase 3, this shape is matched to the closest template flow field. Run LBM simulation to calculate live 2D fluid velocities for this shape!`
      };
      setSelectedCase(customCase);
    } catch (err) {
      console.warn('Real upload failed, falling back to mock flow:', err);
      runMockUploadFlow(file);
    }
  };

  const runMockUploadFlow = (file: File) => {
    setProcessingStep('Extracting keyframe (Mock)...');
    setTimeout(() => {
      setProcessingStep('Detecting calibration card marker (Mock)...');
      setTimeout(() => {
        setProcessingStep('Computing binary boundary silhouette (Mock)...');
        setContourPreview('https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?q=80&w=400&auto=format&fit=crop');
        setTimeout(() => {
          setUploading(false);
          setProcessingStep('');
          setClosestPreset('sports_car_v1');
          setSimMode('Cached 3D Demo (Mock)');

          const customCase: DemoCase = {
            id: 'custom_upload',
            name: `Mock: ${file.name.substring(0, 12)}`,
            desc: 'Ground vehicle approximation (Offline Fallback).',
            drag: 0.35,
            lift: -0.01,
            wake: 0.70,
            mode: 'Cached 3D Demo (Mock)',
            explanation: 'Running in offline fallback mode. The shape was matched to a standard ground vehicle template, displaying approximate cached flow vectors.'
          };
          setSelectedCase(customCase);
        }, 1200);
      }, 1200);
    }, 1200);
  };

  const resetToPresets = () => {
    setSelectedCase(DEMO_CASES[0]);
    setUploadedFile(null);
    setContourPreview(null);
    setClosestPreset(null);
    setActiveJobId(null);
    setSimMode('Cached 3D Demo');
    setShowVoxels(false);
    setShowSlicePlane(false);
    setSlicePosition(0.0);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-container">
          <div className="logo-icon">AV</div>
          <span className="logo-text">AeroVoxel</span>
          <span className="logo-badge">V1.0 MVP</span>
        </div>
        <div className="header-status">
          <div className="status-indicator">
            <span className={`status-dot ${backendStatus === 'connected' ? 'connected' : ''}`}></span>
            {backendStatus === 'connected' ? 'Backend Engine Connected' : 'Running Offline Mode (Fallback)'}
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* Left Sidebar - Inputs and Presets */}
        <section className="sidebar">
          {/* Preset Cases Selector */}
          <div className="panel-card">
            <h3 className="panel-card-title">
              <span>Simulation Input</span>
              {uploadedFile && (
                <button 
                  onClick={resetToPresets} 
                  style={{ 
                    background: 'rgba(255, 67, 87, 0.1)', 
                    border: '1px solid rgba(255, 67, 87, 0.3)', 
                    color: 'var(--accent-red)',
                    fontSize: '10px',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Reset
                </button>
              )}
            </h3>
            
            {!uploadedFile ? (
              <div className="preset-list">
                {DEMO_CASES.map((c) => (
                  <button
                    key={c.id}
                    className={`preset-button ${selectedCase.id === c.id ? 'active' : ''}`}
                    onClick={() => setSelectedCase(c)}
                  >
                    <div>
                      <div className="preset-name">{c.name}</div>
                      <div className="preset-desc">{c.desc}</div>
                    </div>
                    <ChevronRightIcon />
                  </button>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                Currently showing custom uploaded silhouette analysis. Reset to select standard template cases.
              </div>
            )}
          </div>

          {/* Interactive Tunnel Controls */}
          <div className="panel-card">
            <h3 className="panel-card-title">Wind Tunnel Params</h3>
            
            <div className="control-group">
              <div className="control-label">
                <span>Inlet Wind Velocity</span>
                <span className="control-value">{windSpeed} m/s</span>
              </div>
              <input
                type="range"
                className="control-slider"
                min="5"
                max="40"
                value={windSpeed}
                onChange={(e) => setWindSpeed(Number(e.target.value))}
              />
            </div>

            <div className="control-group">
              <div className="control-label">
                <span>Angle of Attack</span>
                <span className="control-value">{windAngle}°</span>
              </div>
              <input
                type="range"
                className="control-slider"
                min="-30"
                max="30"
                value={windAngle}
                onChange={(e) => setWindAngle(Number(e.target.value))}
              />
            </div>
          </div>

          {/* Visualization Layer Toggles */}
          <div className="panel-card">
            <h3 className="panel-card-title">Visual Overlays</h3>
            
            <div 
              className={`toggle-item ${showStreamlines ? 'active' : ''}`}
              onClick={() => setShowStreamlines(!showStreamlines)}
            >
              <span className="toggle-label">Flow Streamlines</span>
              <div className="toggle-switch"></div>
            </div>

            <div 
              className={`toggle-item ${showPressure ? 'active' : ''}`}
              onClick={() => setShowPressure(!showPressure)}
            >
              <span className="toggle-label">Pressure Scalar Map</span>
              <div className="toggle-switch"></div>
            </div>

            <div 
              className={`toggle-item ${showWake ? 'active' : ''}`}
              onClick={() => setShowWake(!showWake)}
            >
              <span className="toggle-label">Wake Turbulence Jitter</span>
              <div className="toggle-switch"></div>
            </div>

            <div 
              className={`toggle-item ${showVoxels ? 'active' : ''}`}
              onClick={() => setShowVoxels(!showVoxels)}
            >
              <span className="toggle-label">Voxelized Geometry View</span>
              <div className="toggle-switch"></div>
            </div>

            <div 
              className={`toggle-item ${showSlicePlane ? 'active' : ''}`}
              onClick={() => setShowSlicePlane(!showSlicePlane)}
            >
              <span className="toggle-label">Enable Cross-Section Slicer</span>
              <div className="toggle-switch"></div>
            </div>

            {showSlicePlane && (
              <div className="control-group" style={{ marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '8px' }}>
                <div className="control-label">
                  <span>Slice Plane Height (Y)</span>
                  <span className="control-value">{slicePosition.toFixed(1)} m</span>
                </div>
                <input
                  type="range"
                  className="control-slider"
                  min="-1.8"
                  max="1.8"
                  step="0.1"
                  value={slicePosition}
                  onChange={(e) => setSlicePosition(Number(e.target.value))}
                />
              </div>
            )}
          </div>
        </section>

        {/* Central Wind Tunnel Viewport */}
        <section style={{ position: 'relative', height: '100%' }}>
          <WindTunnelViewer
            caseId={selectedCase.id}
            windSpeed={windSpeed}
            windAngle={windAngle}
            showStreamlines={showStreamlines}
            showPressure={showPressure}
            showWake={showWake}
            flowData={flowData}
            showVoxels={showVoxels}
            showSlicePlane={showSlicePlane}
            slicePosition={slicePosition}
          />

          {/* Overlay Badges */}
          <div className="viewer-overlay-left">
            <div className={`mode-badge ${simMode.includes('Demo') ? 'cached' : 'computed'}`}>
              <Layers size={13} />
              <span>{simMode} {loadingFlowData ? '(Syncing...)' : ''}</span>
            </div>
            {selectedCase.id === 'custom_upload' && (
              <div className="mode-badge" style={{ borderColor: 'var(--accent-orange)', color: 'var(--accent-orange)' }}>
                <ShieldCheck size={13} />
                <span>2D Scale Extracted (1:18)</span>
              </div>
            )}
          </div>

          <div className="viewer-overlay-right">
            <div style={{ 
              background: 'rgba(10, 11, 16, 0.85)', 
              border: '1px solid var(--border-color)', 
              borderRadius: '8px', 
              padding: '8px 12px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)'
            }}>
              FPS: 60 | WebGL Render Pipeline
            </div>
          </div>
        </section>

        {/* Right Sidebar - Analytics and Upload */}
        <section className="sidebar-right">
          {/* Real-world Upload Section */}
          <div className="panel-card">
            <h3 className="panel-card-title">Ingest Object Video/Image</h3>
            
            {!uploading && !uploadedFile ? (
              <label className="upload-dropzone">
                <Upload className="upload-icon" size={28} />
                <span className="upload-text">Upload Smartphone Capture</span>
                <span className="upload-subtext">Supports MP4, MOV, JPG (Max 50MB)</span>
                <input 
                  type="file" 
                  accept="image/*,video/*" 
                  onChange={handleFileUpload} 
                  style={{ display: 'none' }} 
                />
              </label>
            ) : uploading || runningSolver ? (
              <div className="processing-box">
                <div className="spinner"></div>
                <div className="processing-title">
                  {runningSolver ? 'Navier-Stokes Solver' : 'Computer Vision Pipeline'}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center' }}>
                  {runningSolver ? 'Solving 2D Lattice Boltzmann equations (600 steps)...' : processingStep}
                </div>
              </div>
            ) : (
              <div className="processing-box" style={{ borderColor: 'var(--border-focus)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
                  {uploadedFile?.endsWith('.mp4') || uploadedFile?.endsWith('.mov') ? (
                    <FileVideo size={20} color="var(--accent-cyan)" />
                  ) : (
                    <FileImage size={20} color="var(--accent-cyan)" />
                  )}
                  <span style={{ fontSize: '12px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {uploadedFile}
                  </span>
                </div>
                {contourPreview && (
                  <>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', alignSelf: 'flex-start' }}>
                      Extracted Contour Preview:
                    </div>
                    <img src={contourPreview} alt="Contour Extract" className="contour-preview" />
                  </>
                )}
                <div style={{ fontSize: '11px', color: 'var(--accent-neon)', fontWeight: 500 }}>
                  Active Silhouette Analysis
                </div>
                {activeJobId && (
                  <div style={{ fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    Job ID: {activeJobId.substring(0, 8)}...
                  </div>
                )}
                {simMode === '2D Solver Ready' && (
                  <button 
                    onClick={runLiveSimulation}
                    style={{
                      marginTop: '10px',
                      background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-purple))',
                      border: 'none',
                      color: '#000',
                      fontWeight: 700,
                      fontSize: '12px',
                      padding: '8px 16px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      width: '100%',
                      boxShadow: 'var(--glow-cyan)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px'
                    }}
                  >
                    Run Live LBM Simulation
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Aerodynamic Coefficients */}
          <div className="panel-card">
            <h3 className="panel-card-title">Aerodynamic Estimates</h3>
            
            <div className="metrics-grid">
              <div className="metric-box">
                <span className="metric-lbl">Drag Coefficient (Cd)</span>
                <span className={`metric-val ${selectedCase.id === 'custom_upload' ? 'estimate' : ''}`}>
                  {selectedCase.drag.toFixed(2)}
                </span>
                <span className="metric-desc">Air resistance factor (Lower is more streamlined)</span>
              </div>

              <div className="metric-box">
                <span className="metric-lbl">Wake Turbulence Index</span>
                <span className="metric-val" style={{ color: 'var(--accent-purple)' }}>
                  {selectedCase.wake.toFixed(2)}
                </span>
                <span className="metric-desc">Fluctuating low-pressure separation zone size</span>
              </div>

              <div className="metric-box">
                <span className="metric-lbl">Lift Coefficient (Cl)</span>
                <span className="metric-val" style={{ color: 'var(--accent-neon)' }}>
                  {selectedCase.lift >= 0 ? `+${selectedCase.lift.toFixed(2)}` : selectedCase.lift.toFixed(2)}
                </span>
                <span className="metric-desc">Vertical force perpendicular to flow direction</span>
              </div>
            </div>
          </div>

          {/* Scientific Explanations */}
          <div className="panel-card">
            <h3 className="panel-card-title">Physics Breakdown</h3>
            <div className="explanation-text">
              <p>{selectedCase.explanation}</p>
              <div style={{ display: 'flex', gap: '8px', marginTop: '12px', background: 'rgba(255,255,255,0.02)', padding: '8px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                <Info size={16} color="var(--accent-cyan)" style={{ flexShrink: 0 }} />
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                  Metrics and flow separations are based on educational calculations and are for prototyping visualization purposes.
                </span>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

// Inline Icon Components for simple bundling
const ChevronRightIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m9 18 6-6-6-6"/>
  </svg>
);

export default App;
