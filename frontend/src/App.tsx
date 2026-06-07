import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Layers,
  Upload,
  FileVideo,
  FileImage,
  Info,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { WindTunnelViewer } from './components/WindTunnelViewer';
import type { FlowData } from './components/WindTunnelViewer';
import { LandingScreen } from './components/LandingScreen';
import { fetchNpy } from './utils/npyLoader';
import { API_BASE_URL } from './config';
import type { ActiveCase, ApiDemoCase, DataSource } from './types';
import { PROCESSING_STEPS, OFFLINE_PREVIEW } from './types';
import './App.css';

function toActiveCase(api: ApiDemoCase): ActiveCase {
  return {
    id: api.case_id.replace('_v1', ''),
    backendId: api.case_id,
    name: api.name,
    desc: api.description,
    drag: api.drag_coefficient_estimate,
    lift: api.lift_coefficient_estimate,
    wake: api.wake_score,
    mode: api.mode,
    modeLabel: api.mode_label,
    explanation: api.explanation,
    isCustom: false,
  };
}

const FALLBACK_CASES: ActiveCase[] = [
  {
    id: 'sports_car',
    backendId: 'sports_car_v1',
    name: 'Sports Car (Template)',
    desc: 'Low-drag ground vehicle profile with rear separation.',
    drag: 0.28,
    lift: -0.05,
    wake: 0.65,
    mode: 'cached_2d',
    modeLabel: 'Cached 2D demonstration field',
    explanation:
      'Air hits the front bumper creating a high-pressure stagnation zone. It accelerates over the hood and windshield (low pressure), then separates at the rear, creating a recirculating wake.',
    isCustom: false,
  },
  {
    id: 'drone',
    backendId: 'drone_v1',
    name: 'Quadcopter (Template)',
    desc: 'Complex vertical thrust profile with high drag wake.',
    drag: 1.15,
    lift: 1.42,
    wake: 0.88,
    mode: 'cached_2d',
    modeLabel: 'Cached 2D demonstration field',
    explanation:
      'Flow encounters bluff-body engine mounts and rotor disks. The arms create flow separation and localized vortex shedding.',
    isCustom: false,
  },
  {
    id: 'airfoil',
    backendId: 'airfoil_v1',
    name: 'NACA 0012 Airfoil',
    desc: 'Symmetric streamlined wing section at angle of attack.',
    drag: 0.06,
    lift: 0.45,
    wake: 0.12,
    mode: 'cached_2d',
    modeLabel: 'Cached 2D demonstration field',
    explanation:
      'Flow remains attached to the smooth surface for most of the chord. Curvature differences create asymmetric pressure with a minimal wake profile.',
    isCustom: false,
  },
];

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [demoCases, setDemoCases] = useState<ActiveCase[]>(FALLBACK_CASES);
  const [selectedCase, setSelectedCase] = useState<ActiveCase>(FALLBACK_CASES[0]);
  const [windSpeed, setWindSpeed] = useState(15);
  const [windAngle, setWindAngle] = useState(0);

  const [showStreamlines, setShowStreamlines] = useState(true);
  const [showPressure, setShowPressure] = useState(true);
  const [showWake, setShowWake] = useState(true);
  const [showVoxels, setShowVoxels] = useState(false);
  const [showSlicePlane, setShowSlicePlane] = useState(false);
  const [slicePosition, setSlicePosition] = useState(0.0);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected'>('disconnected');
  const [dataSource, setDataSource] = useState<DataSource>('cached');
  const [simModeLabel, setSimModeLabel] = useState('Cached 2D demonstration field');

  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [processingStepIndex, setProcessingStepIndex] = useState(0);
  const [contourPreview, setContourPreview] = useState<string | null>(null);
  const [scaleEstimate, setScaleEstimate] = useState<number | null>(null);
  const [scaleStatus, setScaleStatus] = useState<string | null>(null);

  const [flowData, setFlowData] = useState<FlowData | null>(null);
  const [loadingFlowData, setLoadingFlowData] = useState(false);
  const [closestPreset, setClosestPreset] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [runningSolver, setRunningSolver] = useState(false);
  const [solverReady, setSolverReady] = useState(false);
  const [lastSolverAngle, setLastSolverAngle] = useState<number | null>(null);

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [fps, setFps] = useState(60);

  const loadFlowForCase = useCallback(
    async (activeCase: ActiveCase, presetOverride?: string | null) => {
      if (backendStatus !== 'connected') {
        setFlowData(null);
        return;
      }

      setLoadingFlowData(true);
      try {
        let backendCaseId = activeCase.backendId;
        if (activeCase.isCustom) {
          const preset = presetOverride ?? closestPreset;
          if (!preset) {
            setFlowData(null);
            return;
          }
          backendCaseId = preset;
        }

        const metaResponse = await fetch(`${API_BASE_URL}/api/flow-field/${backendCaseId}`);
        if (!metaResponse.ok) throw new Error('Failed to load metadata');
        const meta = await metaResponse.json();

        const [velNpy, pressNpy, maskNpy] = await Promise.all([
          fetchNpy(`${API_BASE_URL}${meta.velocity_url}`),
          fetchNpy(`${API_BASE_URL}${meta.pressure_url}`),
          fetchNpy(`${API_BASE_URL}${meta.mask_url}`),
        ]);

        setFlowData({
          velocity: velNpy.data as Float32Array,
          pressure: pressNpy.data as Float32Array,
          mask: maskNpy.data as Uint8Array,
          nx: velNpy.shape[2],
          ny: velNpy.shape[1],
        });

        if (!activeCase.isCustom) {
          setSimModeLabel(meta.mode_label ?? 'Cached 2D demonstration field');
          setDataSource('cached');
        }
      } catch (err) {
        console.error('Error loading flow arrays:', err);
        setFlowData(null);
      } finally {
        setLoadingFlowData(false);
      }
    },
    [backendStatus, closestPreset]
  );

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        setBackendStatus(response.ok ? 'connected' : 'disconnected');
      } catch {
        setBackendStatus('disconnected');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (backendStatus !== 'connected') return;

    const fetchCases = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/demo-cases`);
        if (!response.ok) return;
        const cases: ApiDemoCase[] = await response.json();
        const mapped = cases.map(toActiveCase);
        setDemoCases(mapped);
        if (!selectedCase.isCustom) {
          setSelectedCase((prev) => mapped.find((c) => c.backendId === prev.backendId) ?? mapped[0]);
        }
      } catch (err) {
        console.warn('Failed to load demo cases from API, using fallback list.', err);
      }
    };

    fetchCases();
  }, [backendStatus]);

  useEffect(() => {
    if (dataSource === 'computed') return;
    if (selectedCase.isCustom && !closestPreset) return;
    loadFlowForCase(selectedCase);
  }, [selectedCase, backendStatus, closestPreset, dataSource, loadFlowForCase]);

  const runLiveSimulation = async () => {
    if (!activeJobId) return;
    setRunningSolver(true);
    setSimModeLabel('Running educational 2D LBM solver…');

    try {
      const response = await fetch(`${API_BASE_URL}/api/simulate/simple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: activeJobId,
          wind_speed: windSpeed,
          wind_angle_deg: windAngle,
        }),
      });

      if (!response.ok) throw new Error('Simulation failed');
      const res = await response.json();

      const [velNpy, pressNpy, maskNpy] = await Promise.all([
        fetchNpy(`${API_BASE_URL}${res.velocity_url}`),
        fetchNpy(`${API_BASE_URL}${res.pressure_url}`),
        fetchNpy(`${API_BASE_URL}${res.mask_url}`),
      ]);

      setFlowData({
        velocity: velNpy.data as Float32Array,
        pressure: pressNpy.data as Float32Array,
        mask: maskNpy.data as Uint8Array,
        nx: velNpy.shape[2],
        ny: velNpy.shape[1],
      });

      setDataSource('computed');
      setSimModeLabel('Educational 2D LBM solver output');
      setLastSolverAngle(windAngle);
      setSolverReady(false);

      setSelectedCase((prev) => ({
        ...prev,
        drag: res.metrics.drag_coefficient_estimate,
        lift: res.metrics.lift_coefficient_estimate,
        wake: res.metrics.wake_score,
        mode: 'simple_solver',
        modeLabel: 'Educational 2D LBM solver output',
        explanation: `The 2D LBM solver converged in 600 steps on your uploaded silhouette. Approximate drag coefficient: ${res.metrics.drag_coefficient_estimate.toFixed(2)}. Lift estimate: ${res.metrics.lift_coefficient_estimate.toFixed(2)}. These are educational-style metrics, not certified aerodynamic coefficients.`,
      }));
    } catch (err) {
      console.error('Simulation run failed:', err);
      setSimModeLabel('2D LBM solver failed — showing closest cached field');
      setDataSource('cached');
      if (closestPreset) loadFlowForCase(selectedCase, closestPreset);
    } finally {
      setRunningSolver(false);
    }
  };

  const runMockUploadFlow = (file: File) => {
    setProcessingStepIndex(0);
    const advance = (step: number, delay: number, done: () => void) => {
      setTimeout(() => {
        setProcessingStepIndex(step);
        done();
      }, delay);
    };

    advance(1, 800, () =>
      advance(2, 800, () =>
        advance(3, 800, () => {
          setUploading(false);
          setContourPreview(OFFLINE_PREVIEW);
          setClosestPreset('sports_car_v1');
          setActiveJobId(null);
          setSolverReady(false);
          setDataSource('offline_fallback');
          setSimModeLabel('Offline fallback (cached template)');
          setScaleEstimate(null);
          setScaleStatus('Offline mode — scale not detected');

          setSelectedCase({
            id: 'custom_upload',
            backendId: 'custom_upload',
            name: `Offline: ${file.name.substring(0, 16)}`,
            desc: 'Matched to sports car template (offline fallback).',
            drag: 0.28,
            lift: -0.05,
            wake: 0.65,
            mode: 'cached_2d',
            modeLabel: 'Offline fallback (cached template)',
            explanation:
              'Backend unavailable. AeroVoxel matched your upload to the cached sports car demonstration field so the demo keeps running.',
            isCustom: true,
          });
        })
      )
    );
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setShowLanding(false);
    setUploading(true);
    setUploadedFile(file.name);
    setContourPreview(null);
    setClosestPreset(null);
    setActiveJobId(null);
    setSolverReady(false);
    setDataSource('cached');
    setProcessingStepIndex(0);
    setLastSolverAngle(null);

    if (backendStatus !== 'connected') {
      runMockUploadFlow(file);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setProcessingStepIndex(1);
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const res = await response.json();

      setProcessingStepIndex(2);
      setClosestPreset(res.closest_preset);
      setActiveJobId(res.job_id);
      setContourPreview(`${API_BASE_URL}${res.preview_url}`);
      setScaleEstimate(res.scale_estimate);
      setScaleStatus(res.scale_status);
      setProcessingStepIndex(3);
      setSolverReady(true);
      setSimModeLabel('Upload processed — ready for 2D LBM solver');

      const presetCase = demoCases.find((c) => c.backendId === res.closest_preset);

      setSelectedCase({
        id: 'custom_upload',
        backendId: 'custom_upload',
        name: `Upload: ${res.filename.substring(0, 16)}`,
        desc: `Matched to ${presetCase?.name ?? res.closest_preset}. Scale: ${res.scale_estimate.toFixed(3)} mm/px.`,
        drag: presetCase?.drag ?? 0.28,
        lift: presetCase?.lift ?? 0.0,
        wake: presetCase?.wake ?? 0.65,
        mode: 'upload_ready',
        modeLabel: 'Upload processed — ready for 2D LBM solver',
        explanation: `${res.detected_object}. ${res.scale_status}. Run the educational 2D LBM solver to compute flow for your silhouette. Until then, the closest cached template field is shown.`,
        isCustom: true,
      });

      await loadFlowForCase(
        { ...FALLBACK_CASES[0], isCustom: true, backendId: 'custom_upload' },
        res.closest_preset
      );
    } catch (err) {
      console.warn('Real upload failed, falling back to offline mode:', err);
      runMockUploadFlow(file);
    } finally {
      setUploading(false);
    }
  };

  const resetToPresets = () => {
    setSelectedCase(demoCases[0] ?? FALLBACK_CASES[0]);
    setUploadedFile(null);
    setContourPreview(null);
    setClosestPreset(null);
    setActiveJobId(null);
    setSolverReady(false);
    setDataSource('cached');
    setSimModeLabel('Cached 2D demonstration field');
    setScaleEstimate(null);
    setScaleStatus(null);
    setLastSolverAngle(null);
    setShowVoxels(false);
    setShowSlicePlane(false);
    setSlicePosition(0.0);
  };

  const enterDashboard = (mode: 'demo' | 'upload') => {
    setShowLanding(false);
    if (mode === 'upload') {
      uploadInputRef.current?.click();
    }
  };

  const angleNeedsRerun =
    dataSource === 'computed' &&
    lastSolverAngle !== null &&
    Math.abs(lastSolverAngle - windAngle) > 0.5;

  if (showLanding) {
    return (
      <>
        <input
          ref={uploadInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
        <LandingScreen
          onTryDemo={() => {
            setShowLanding(false);
            resetToPresets();
          }}
          onUpload={() => enterDashboard('upload')}
        />
      </>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo-container">
          <div className="logo-icon">AV</div>
          <span className="logo-text">AeroVoxel</span>
          <span className="logo-badge">MVP</span>
        </div>
        <div className="header-status">
          <div className="status-indicator">
            <span className={`status-dot ${backendStatus === 'connected' ? 'connected' : ''}`} />
            {backendStatus === 'connected'
              ? 'Backend Connected'
              : 'Offline Mode — Cached Fallback Active'}
          </div>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="sidebar">
          <div className="panel-card">
            <h3 className="panel-card-title">
              <span>Simulation Input</span>
              {uploadedFile && (
                <button type="button" className="reset-btn" onClick={resetToPresets}>
                  Reset
                </button>
              )}
            </h3>

            {!uploadedFile ? (
              <div className="preset-list">
                {demoCases.map((c) => (
                  <button
                    key={c.backendId}
                    type="button"
                    className={`preset-button ${selectedCase.backendId === c.backendId ? 'active' : ''}`}
                    onClick={() => {
                      setSelectedCase(c);
                      setDataSource('cached');
                      setSimModeLabel(c.modeLabel);
                      setLastSolverAngle(null);
                    }}
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
              <p className="sidebar-note">
                Custom upload active. Reset to return to template demo cases.
              </p>
            )}
          </div>

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
              {dataSource === 'computed' && angleNeedsRerun && (
                <p className="control-hint">Re-run LBM solver to apply new angle to flow field.</p>
              )}
              {dataSource === 'cached' && (
                <p className="control-hint">Cached fields: angle rotates the 3D view. Run LBM after upload for angled inlet flow.</p>
              )}
            </div>
          </div>

          <div className="panel-card">
            <h3 className="panel-card-title">Visual Overlays</h3>
            <ToggleRow label="Flow Streamlines" active={showStreamlines} onToggle={() => setShowStreamlines(!showStreamlines)} />
            <ToggleRow label="Pressure Scalar Map" active={showPressure} onToggle={() => setShowPressure(!showPressure)} />
            <ToggleRow label="Wake Turbulence Jitter" active={showWake} onToggle={() => setShowWake(!showWake)} />
            <ToggleRow label="Voxelized Geometry View" active={showVoxels} onToggle={() => setShowVoxels(!showVoxels)} />
            <ToggleRow label="Cross-Section Slicer" active={showSlicePlane} onToggle={() => setShowSlicePlane(!showSlicePlane)} />
            {showSlicePlane && (
              <div className="control-group slice-control">
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

        <section className="viewer-section">
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
            onFpsUpdate={setFps}
          />

          <div className="viewer-overlay-left">
            <div className={`mode-badge ${dataSource === 'computed' ? 'computed' : 'cached'}`}>
              <Layers size={13} />
              <span>
                {simModeLabel}
                {loadingFlowData ? ' (loading…)' : ''}
              </span>
            </div>
            {selectedCase.isCustom && scaleStatus && (
              <div className="mode-badge scale-badge">
                <ShieldCheck size={13} />
                <span>
                  {scaleEstimate !== null
                    ? `Scale: ${scaleEstimate.toFixed(3)} mm/px`
                    : scaleStatus}
                </span>
              </div>
            )}
          </div>

          <div className="viewer-overlay-right">
            <div className="fps-badge">FPS: {fps} · WebGL</div>
          </div>
        </section>

        <section className="sidebar-right">
          <div className="panel-card">
            <h3 className="panel-card-title">Ingest Object Video/Image</h3>

            {!uploading && !uploadedFile ? (
              <label className="upload-dropzone">
                <Upload className="upload-icon" size={28} />
                <span className="upload-text">Upload Smartphone Capture</span>
                <span className="upload-subtext">MP4, MOV, JPG, PNG · Max 50MB</span>
                <span className="upload-tip">
                  Tip: place a credit card next to the object for scale calibration.
                </span>
                <input
                  ref={uploadInputRef}
                  type="file"
                  accept="image/*,video/*"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
              </label>
            ) : uploading || runningSolver ? (
              <div className="processing-box">
                <div className="spinner" />
                <div className="processing-title">
                  {runningSolver ? '2D LBM Solver' : 'Computer Vision Pipeline'}
                </div>
                {runningSolver ? (
                  <p className="processing-subtext">
                    Running educational 2D lattice Boltzmann solver (600 steps)…
                  </p>
                ) : (
                  <>
                    <div className="processing-steps">
                      {PROCESSING_STEPS.map((step, i) => (
                        <div
                          key={step}
                          className={`processing-step ${i <= processingStepIndex ? 'done' : ''} ${i === processingStepIndex ? 'active' : ''}`}
                        >
                          <span className="step-dot" />
                          {step}
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="processing-box upload-complete">
                <div className="upload-file-row">
                  {uploadedFile?.match(/\.(mp4|mov)$/i) ? (
                    <FileVideo size={20} color="var(--accent-cyan)" />
                  ) : (
                    <FileImage size={20} color="var(--accent-cyan)" />
                  )}
                  <span className="upload-filename">{uploadedFile}</span>
                </div>
                {contourPreview && (
                  <>
                    <span className="preview-label">Extracted contour preview</span>
                    <img src={contourPreview} alt="Contour preview" className="contour-preview" />
                  </>
                )}
                {activeJobId && (
                  <span className="job-id">Job: {activeJobId.substring(0, 8)}…</span>
                )}
                {solverReady && backendStatus === 'connected' && (
                  <button type="button" className="run-solver-btn" onClick={runLiveSimulation}>
                    Run 2D LBM Simulation
                  </button>
                )}
                {angleNeedsRerun && (
                  <button type="button" className="run-solver-btn secondary" onClick={runLiveSimulation}>
                    Re-run with new angle ({windAngle}°)
                  </button>
                )}
              </div>
            )}
          </div>

          <div className="panel-card">
            <h3 className="panel-card-title">Aerodynamic Estimates</h3>
            <div className="metrics-grid">
              <MetricBox
                label="Drag Coefficient (Cd)"
                value={selectedCase.drag.toFixed(2)}
                desc="Approximate air resistance estimate (educational)"
                highlight={selectedCase.isCustom}
              />
              <MetricBox
                label="Wake Turbulence Index"
                value={selectedCase.wake.toFixed(2)}
                desc="Low-pressure separation zone indicator"
                color="var(--accent-purple)"
              />
              <MetricBox
                label="Lift Coefficient (Cl)"
                value={
                  selectedCase.lift >= 0
                    ? `+${selectedCase.lift.toFixed(2)}`
                    : selectedCase.lift.toFixed(2)
                }
                desc="Approximate vertical force estimate (educational)"
                color="var(--accent-neon)"
              />
            </div>
          </div>

          <div className="panel-card">
            <h3 className="panel-card-title">Physics Breakdown</h3>
            <div className="explanation-text">
              <p>{selectedCase.explanation}</p>
              <div className="honesty-callout">
                <Info size={16} color="var(--accent-cyan)" />
                <span>
                  Metrics are approximate educational estimates for prototyping visualization —
                  not certified aerodynamic coefficients.
                </span>
              </div>
            </div>
          </div>

          <div className="panel-card">
            <button
              type="button"
              className="technical-toggle"
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            >
              <span>Technical Details</span>
              {showTechnicalDetails ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {showTechnicalDetails && (
              <div className="technical-details">
                <DetailRow label="Simulation mode" value={simModeLabel} />
                <DetailRow label="Grid resolution" value="128 × 64 (2D)" />
                <DetailRow
                  label="Data source"
                  value={
                    dataSource === 'computed'
                      ? 'Live 2D LBM solver'
                      : dataSource === 'offline_fallback'
                        ? 'Offline cached fallback'
                        : 'Precomputed demonstration field'
                  }
                />
                <DetailRow label="Wind speed" value={`${windSpeed} m/s`} />
                <DetailRow
                  label="Wind angle"
                  value={
                    dataSource === 'computed' && lastSolverAngle !== null
                      ? `${lastSolverAngle}° (applied to solver inlet)`
                      : `${windAngle}° (visual / pending solver run)`
                  }
                />
                <div className="limitations-block">
                  <strong>Known limitations</strong>
                  <ul>
                    <li>2D educational simulation — not full 3D CFD</li>
                    <li>3D object geometry is illustrative, not reconstructed from video</li>
                    <li>Uploads map to closest template until LBM solver runs</li>
                    <li>Drag/lift values are heuristic estimates, not wind-tunnel validated</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function ToggleRow({
  label,
  active,
  onToggle,
}: {
  label: string;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <div className={`toggle-item ${active ? 'active' : ''}`} onClick={onToggle}>
      <span className="toggle-label">{label}</span>
      <div className="toggle-switch" />
    </div>
  );
}

function MetricBox({
  label,
  value,
  desc,
  highlight,
  color,
}: {
  label: string;
  value: string;
  desc: string;
  highlight?: boolean;
  color?: string;
}) {
  return (
    <div className="metric-box">
      <span className="metric-lbl">{label}</span>
      <span className={`metric-val ${highlight ? 'estimate' : ''}`} style={color ? { color } : undefined}>
        {value}
      </span>
      <span className="metric-desc">{desc}</span>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span className="detail-label">{label}</span>
      <span className="detail-value">{value}</span>
    </div>
  );
}

const ChevronRightIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m9 18 6-6-6-6" />
  </svg>
);

export default App;