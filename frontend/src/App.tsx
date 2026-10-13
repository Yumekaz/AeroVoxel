import { useState, useEffect, useRef, useCallback } from "react";
import {
  Upload,
  FileImage,
  Info,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  Car,
  Plane,
  Circle,
  Wind,
  Box,
  RotateCcw,
  Pause,
  Play,
  PanelRightClose,
  PanelRightOpen,
  ArrowUpRight,
  Home,
} from "lucide-react";
import { WindTunnelViewer } from "./components/WindTunnelViewer";
import type { FlowData } from "./components/WindTunnelViewer";
import { LandingScreen } from "./components/LandingScreen";
import { domainHeight, isDemoModel } from "./components/demoModels";
import { fetchNpy } from "./utils/npyLoader";
import { API_BASE_URL } from "./config";
import type { ActiveCase, ApiDemoCase, DataSource } from "./types";
import { PROCESSING_STEPS, OFFLINE_PREVIEW } from "./types";
import "./App.css";

function toActiveCase(api: ApiDemoCase): ActiveCase {
  return {
    id: api.case_id.replace("_v1", ""),
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
    metricsSource: api.metrics_source,
  };
}

const FALLBACK_CASES: ActiveCase[] = [
  {
    id: "sports_car",
    backendId: "sports_car_v1",
    name: "Sports Car (Template)",
    desc: "Low-drag ground vehicle profile with rear separation.",
    drag: 0.28,
    lift: -0.05,
    wake: 0.65,
    mode: "cached_2d",
    modeLabel: "Cached 2D demonstration field",
    explanation:
      "Air hits the front bumper creating a high-pressure stagnation zone. It accelerates over the hood and windshield (low pressure), then separates at the rear, creating a recirculating wake.",
    isCustom: false,
  },
  {
    id: "drone",
    backendId: "drone_v1",
    name: "Quadcopter (Template)",
    desc: "Complex vertical thrust profile with high drag wake.",
    drag: 1.15,
    lift: 1.42,
    wake: 0.88,
    mode: "cached_2d",
    modeLabel: "Cached 2D demonstration field",
    explanation:
      "Flow encounters bluff-body engine mounts and rotor disks. The arms create flow separation and localized vortex shedding.",
    isCustom: false,
  },
  {
    id: "airfoil",
    backendId: "airfoil_v1",
    name: "NACA 0012 Airfoil",
    desc: "Symmetric streamlined wing section at angle of attack.",
    drag: 0.06,
    lift: 0.45,
    wake: 0.12,
    mode: "cached_2d",
    modeLabel: "Cached 2D demonstration field",
    explanation:
      "Flow remains attached to the smooth surface for most of the chord. Curvature differences create asymmetric pressure with a minimal wake profile.",
    isCustom: false,
  },
  {
    id: "cylinder",
    backendId: "cylinder_v1",
    name: "Circular Cylinder (Validation)",
    desc: "Classic 2D bluff body for educational wake and stagnation visualization.",
    drag: 1.1,
    lift: 0.0,
    wake: 0.82,
    mode: "cached_2d",
    modeLabel: "Educational 2D LBM field",
    explanation:
      "Oncoming flow stagnates on the windward face (high pressure), accelerates around the shoulders, then separates and forms a recirculating wake. Educational bluff-body case — not certified Cd.",
    isCustom: false,
  },
  // sphere_3d_v1 is intentionally omitted from offline fallback: it only appears
  // when backend reports assets (velocity/pressure/mask .npy) are present.
];

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [demoCases, setDemoCases] = useState<ActiveCase[]>(FALLBACK_CASES);
  const [selectedCase, setSelectedCase] = useState<ActiveCase>(
    FALLBACK_CASES[0],
  );
  const [windSpeed, setWindSpeed] = useState(15);
  const [windAngle, setWindAngle] = useState(0);

  const [showStreamlines, setShowStreamlines] = useState(true);
  const [showPressure, setShowPressure] = useState(true);
  const [showWake, setShowWake] = useState(true);
  const [showVoxels, setShowVoxels] = useState(false);
  const [showSlicePlane, setShowSlicePlane] = useState(false);
  const [slicePosition, setSlicePosition] = useState(0.0);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const [backendStatus, setBackendStatus] = useState<
    "connected" | "disconnected"
  >("disconnected");
  const [dataSource, setDataSource] = useState<DataSource>("cached");
  const [simModeLabel, setSimModeLabel] = useState(
    "Cached 2D demonstration field",
  );

  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [processingStepIndex, setProcessingStepIndex] = useState(0);
  const [contourPreview, setContourPreview] = useState<string | null>(null);
  const [, setScaleEstimate] = useState<number | null>(null);
  const [scaleStatus, setScaleStatus] = useState<string | null>(null);

  const [flowData, setFlowData] = useState<FlowData | null>(null);
  const [loadingFlowData, setLoadingFlowData] = useState(false);
  const [closestPreset, setClosestPreset] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [runningSolver, setRunningSolver] = useState(false);
  const [solverReady, setSolverReady] = useState(false);
  const [lastSolverAngle, setLastSolverAngle] = useState<number | null>(null);

  // Educational ML surrogate (parallel to LBM — does not replace live solver)
  const [surrogateAvailable, setSurrogateAvailable] = useState<boolean | null>(
    null,
  );
  const [surrogateHint, setSurrogateHint] = useState<string | null>(null);
  const [surrogateStatusDisclaimer, setSurrogateStatusDisclaimer] = useState<
    string | null
  >(null);
  const [maskForSurrogate, setMaskForSurrogate] = useState<{
    flat: number[];
    nx: number;
    ny: number;
  } | null>(null);
  const [surrogateLoading, setSurrogateLoading] = useState(false);
  const [surrogateError, setSurrogateError] = useState<string | null>(null);
  const [surrogatePred, setSurrogatePred] = useState<{
    cd: number;
    model: string;
    disclaimer: string;
  } | null>(null);

  const [previewMask, setPreviewMask] = useState<Pick<
    FlowData,
    "mask" | "nx" | "ny"
  > | null>(null);
  const maskImportRef = useRef<HTMLInputElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [fps, setFps] = useState(60);
  const [paused, setPaused] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [cameraView, setCameraView] = useState<"side" | "perspective">("side");
  const [resetCamera, setResetCamera] = useState(0);
  const [resultsOpen, setResultsOpen] = useState(true);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastSolverSpeed, setLastSolverSpeed] = useState<number | null>(null);

  const effectiveDataSource: DataSource =
    backendStatus === "disconnected" && dataSource !== "computed"
      ? "offline_fallback"
      : dataSource;
  const effectiveSimModeLabel =
    effectiveDataSource === "offline_fallback"
      ? "Offline fallback — procedural viewer (backend disconnected)"
      : simModeLabel;
  const effectiveSurrogateAvailable =
    backendStatus === "disconnected" ? null : surrogateAvailable;
  const effectiveSurrogateHint =
    backendStatus === "disconnected" ? null : surrogateHint;

  const storeMaskForSurrogate = useCallback(
    (data: ArrayLike<number>, nx: number, ny: number) => {
      const n = nx * ny;
      const flat = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        flat[i] = Number(data[i]) > 0.5 ? 1 : 0;
      }
      setMaskForSurrogate({ flat, nx, ny });
      setSurrogatePred(null);
      setSurrogateError(null);
    },
    [],
  );

  const flowSequence = useRef(0);
  const loadFlowForCase = useCallback(
    async (activeCase: ActiveCase, presetOverride?: string | null) => {
      const sequence = ++flowSequence.current;
      if (backendStatus !== "connected") {
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

        const metaResponse = await fetch(
          `${API_BASE_URL}/api/flow-field/${backendCaseId}`,
        );
        if (!metaResponse.ok) throw new Error("Failed to load metadata");
        const meta = await metaResponse.json();

        const [velNpy, pressNpy, maskNpy] = await Promise.all([
          fetchNpy(`${API_BASE_URL}${meta.velocity_url}`),
          fetchNpy(`${API_BASE_URL}${meta.pressure_url}`),
          fetchNpy(`${API_BASE_URL}${meta.mask_url}`),
        ]);

        if (sequence !== flowSequence.current) return;
        const nx = velNpy.shape[2];
        const ny = velNpy.shape[1];
        setFlowData({
          velocity: velNpy.data as Float32Array,
          pressure: pressNpy.data as Float32Array,
          mask: maskNpy.data as Uint8Array,
          nx,
          ny,
        });
        // Prefer explicit mask shape when present (ny, nx)
        const maskNy = maskNpy.shape.length >= 2 ? maskNpy.shape[0] : ny;
        const maskNx = maskNpy.shape.length >= 2 ? maskNpy.shape[1] : nx;
        if (!activeCase.isCustom)
          storeMaskForSurrogate(
            maskNpy.data as ArrayLike<number>,
            maskNx,
            maskNy,
          );

        if (!activeCase.isCustom) {
          setSimModeLabel(meta.mode_label ?? "Cached 2D demonstration field");
          setDataSource(
            activeCase.mode === "real_3d_lbm" ? "real_3d_lbm" : "cached",
          );
        }
      } catch (err) {
        console.error("Error loading flow arrays:", err);
        if (sequence === flowSequence.current) setFlowData(null);
      } finally {
        if (sequence === flowSequence.current) setLoadingFlowData(false);
      }
    },
    [backendStatus, closestPreset, storeMaskForSurrogate],
  );

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
          setBackendStatus("connected");
        } else {
          setBackendStatus("disconnected");
        }
      } catch {
        setBackendStatus("disconnected");
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  // Surrogate model availability (does not replace LBM)
  useEffect(() => {
    if (backendStatus !== "connected") {
      return;
    }
    let cancelled = false;
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/surrogate/status`);
        if (!response.ok) throw new Error("status failed");
        const body = await response.json();
        if (cancelled) return;
        setSurrogateAvailable(Boolean(body.model_available));
        setSurrogateHint(body.hint ?? null);
        setSurrogateStatusDisclaimer(body.educational_disclaimer ?? null);
      } catch {
        if (!cancelled) {
          setSurrogateAvailable(false);
          setSurrogateHint("Could not reach surrogate status endpoint");
        }
      }
    };
    fetchStatus();
    return () => {
      cancelled = true;
    };
  }, [backendStatus]);

  useEffect(() => {
    if (backendStatus !== "connected") return;

    const fetchCases = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/demo-cases`);
        if (!response.ok) return;
        const cases: ApiDemoCase[] = await response.json();
        const mapped = cases.map(toActiveCase);
        // Only list cases the API says are asset-backed (hides sphere until cache exists).
        setDemoCases(mapped.length > 0 ? mapped : FALLBACK_CASES);
        if (!selectedCase.isCustom) {
          setSelectedCase((prev) => {
            const list = mapped.length > 0 ? mapped : FALLBACK_CASES;
            return list.find((c) => c.backendId === prev.backendId) ?? list[0];
          });
        }
      } catch (err) {
        console.warn(
          "Failed to load demo cases from API, using fallback list.",
          err,
        );
      }
    };

    fetchCases();
  }, [backendStatus, selectedCase.isCustom]);

  useEffect(() => {
    if (effectiveDataSource === "computed") return;
    if (selectedCase.isCustom && !closestPreset) return;
    const load = async () => {
      await loadFlowForCase(selectedCase);
    };
    void load();
  }, [
    selectedCase,
    backendStatus,
    closestPreset,
    effectiveDataSource,
    loadFlowForCase,
  ]);

  const runLiveSimulation = async () => {
    if (!activeJobId) return;
    flowSequence.current++;
    setLoadingFlowData(false);
    setRunningSolver(true);
    setSimModeLabel("Running educational 2D LBM solver…");

    try {
      const response = await fetch(`${API_BASE_URL}/api/simulate/simple`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: activeJobId,
          wind_speed: windSpeed,
          wind_angle_deg: windAngle,
        }),
      });

      if (!response.ok) throw new Error("Simulation failed");
      const res = await response.json();

      const [velNpy, pressNpy, maskNpy] = await Promise.all([
        fetchNpy(`${API_BASE_URL}${res.velocity_url}`),
        fetchNpy(`${API_BASE_URL}${res.pressure_url}`),
        fetchNpy(`${API_BASE_URL}${res.mask_url}`),
      ]);

      const nx = velNpy.shape[2];
      const ny = velNpy.shape[1];
      setFlowData({
        velocity: velNpy.data as Float32Array,
        pressure: pressNpy.data as Float32Array,
        mask: maskNpy.data as Uint8Array,
        nx,
        ny,
      });
      const maskNy = maskNpy.shape.length >= 2 ? maskNpy.shape[0] : ny;
      const maskNx = maskNpy.shape.length >= 2 ? maskNpy.shape[1] : nx;
      storeMaskForSurrogate(maskNpy.data as ArrayLike<number>, maskNx, maskNy);

      setPreviewMask(null);
      setDataSource("computed");
      setSimModeLabel("Educational 2D LBM solver output");
      setLastSolverAngle(windAngle);
      setLastSolverSpeed(windSpeed);
      setSolverReady(false);

      setSelectedCase((prev) => ({
        ...prev,
        drag: res.metrics.drag_coefficient_estimate,
        lift: res.metrics.lift_coefficient_estimate,
        wake: res.metrics.wake_score,
        mode: "simple_solver",
        modeLabel: "Educational 2D LBM solver output",
        explanation: `The 2D LBM solver converged in 600 steps on your uploaded silhouette. Approximate drag coefficient: ${res.metrics.drag_coefficient_estimate.toFixed(2)}. Lift estimate: ${res.metrics.lift_coefficient_estimate.toFixed(2)}. These are educational-style metrics, not certified aerodynamic coefficients.`,
      }));
    } catch (err) {
      console.error("Simulation run failed:", err);
      setSimModeLabel("2D LBM solver failed — showing closest cached field");
      setDataSource("cached");
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
          setClosestPreset("sports_car_v1");
          setActiveJobId(null);
          setSolverReady(false);
          setDataSource("offline_fallback");
          setSimModeLabel("Offline fallback (cached template)");
          setScaleEstimate(null);
          setScaleStatus("Offline mode — scale not detected");

          setSelectedCase({
            id: "custom_upload",
            backendId: "custom_upload",
            name: `Offline: ${file.name.substring(0, 16)}`,
            desc: "Matched to sports car template (offline fallback).",
            drag: 0.28,
            lift: -0.05,
            wake: 0.65,
            mode: "cached_2d",
            modeLabel: "Offline fallback (cached template)",
            explanation:
              "Backend unavailable. AeroVoxel matched your upload to the cached sports car demonstration field so the demo keeps running.",
            isCustom: true,
          });
        }),
      ),
    );
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setShowLanding(false);
    setPreviewMask(null);
    setUploadError(null);
    setUploading(true);
    setUploadedFile(file.name);
    setContourPreview(null);
    setClosestPreset(null);
    setActiveJobId(null);
    setSolverReady(false);
    setDataSource("cached");
    setProcessingStepIndex(0);
    setLastSolverAngle(null);
    setSurrogatePred(null);
    setSurrogateError(null);
    setMaskForSurrogate(null);

    if (backendStatus !== "connected") {
      runMockUploadFlow(file);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setProcessingStepIndex(1);
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");
      const res = await response.json();

      setProcessingStepIndex(2);
      setClosestPreset(res.closest_preset);
      setActiveJobId(res.job_id);
      setContourPreview(`${API_BASE_URL}${res.preview_url}`);
      setScaleEstimate(res.scale_estimate);
      setScaleStatus(res.scale_status);
      setProcessingStepIndex(3);
      setSolverReady(true);
      setSimModeLabel("Upload processed — ready for 2D LBM solver");

      const presetCase = demoCases.find(
        (c) => c.backendId === res.closest_preset,
      );

      setSelectedCase({
        id: "custom_upload",
        backendId: "custom_upload",
        name: `Upload: ${res.filename.substring(0, 16)}`,
        desc: `Matched to ${presetCase?.name ?? res.closest_preset}. Scale: ${res.scale_estimate.toFixed(3)} mm/px.`,
        drag: presetCase?.drag ?? 0.28,
        lift: presetCase?.lift ?? 0.0,
        wake: presetCase?.wake ?? 0.65,
        mode: "upload_ready",
        modeLabel: "Upload processed — ready for 2D LBM solver",
        explanation: `${res.detected_object}. ${res.scale_status}. Run the educational 2D LBM solver to compute flow for your silhouette. Until then, the closest cached template field is shown.`,
        isCustom: true,
      });

      await loadFlowForCase(
        { ...FALLBACK_CASES[0], isCustom: true, backendId: "custom_upload" },
        res.closest_preset,
      );

      // Prefer the uploaded job mask for the surrogate (not only the closest-template field)
      try {
        const jobMask = await fetchNpy(
          `${API_BASE_URL}/api/simulate/result/${res.job_id}/mask`,
        );
        if (jobMask.shape.length >= 2) {
          storeMaskForSurrogate(
            jobMask.data as ArrayLike<number>,
            jobMask.shape[1],
            jobMask.shape[0],
          );
        }
      } catch (maskErr) {
        console.warn(
          "Could not load upload mask for surrogate; using loaded flow mask.",
          maskErr,
        );
      }
    } catch (err) {
      console.warn("Upload failed:", err);
      setUploadedFile(null);
      setUploadError(
        "Could not process this capture. Please try a clear image or a supported video.",
      );
    } finally {
      setUploading(false);
    }
  };

  const resetToPresets = () => {
    setPreviewMask(null);
    setUploadError(null);
    setSelectedCase(demoCases[0] ?? FALLBACK_CASES[0]);
    setUploadedFile(null);
    setContourPreview(null);
    setClosestPreset(null);
    setActiveJobId(null);
    setSolverReady(false);
    setDataSource("cached");
    setSimModeLabel("Cached 2D demonstration field");
    setScaleEstimate(null);
    setScaleStatus(null);
    setLastSolverAngle(null);
    setShowVoxels(false);
    setShowSlicePlane(false);
    setSlicePosition(0.0);
    setSurrogatePred(null);
    setSurrogateError(null);
  };

  const runSurrogatePredict = async () => {
    if (!maskForSurrogate || backendStatus !== "connected") return;
    if (effectiveSurrogateAvailable === false) return;

    setSurrogateLoading(true);
    setSurrogateError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/surrogate/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          flat_mask: maskForSurrogate.flat,
          nx: maskForSurrogate.nx,
          ny: maskForSurrogate.ny,
        }),
      });
      if (response.status === 503) {
        const detail = await response.json().catch(() => null);
        setSurrogateAvailable(false);
        setSurrogateError(
          typeof detail?.detail === "string"
            ? detail.detail
            : "Surrogate model not available. Train locally first.",
        );
        return;
      }
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(
          typeof detail?.detail === "string"
            ? detail.detail
            : `Predict failed (${response.status})`,
        );
      }
      const body = await response.json();
      setSurrogatePred({
        cd: Number(body.cd_force_proxy_pred),
        model: String(body.model_name ?? "surrogate"),
        disclaimer: String(
          body.educational_disclaimer ??
            surrogateStatusDisclaimer ??
            "Educational surrogate only — not certified CFD.",
        ),
      });
    } catch (err) {
      console.error("Surrogate predict failed:", err);
      setSurrogateError(
        err instanceof Error ? err.message : "Surrogate prediction failed",
      );
      setSurrogatePred(null);
    } finally {
      setSurrogateLoading(false);
    }
  };

  const enterDashboard = (mode: "demo" | "upload") => {
    setShowLanding(false);
    if (mode === "upload") {
      uploadInputRef.current?.click();
    }
  };

  const angleNeedsRerun =
    effectiveDataSource === "computed" &&
    lastSolverAngle !== null &&
    (Math.abs(lastSolverAngle - windAngle) > 0.5 ||
      lastSolverSpeed !== windSpeed);

  if (showLanding) {
    return (
      <>
        <input
          ref={uploadInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleFileUpload}
          style={{ display: "none" }}
        />
        <LandingScreen
          onTryDemo={() => {
            setShowLanding(false);
            resetToPresets();
          }}
          onUpload={() => enterDashboard("upload")}
        />
      </>
    );
  }

  const importMask = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    flowSequence.current++;
    setLoadingFlowData(false);
    setUploadError(null);
    setUploading(true);
    setResultsOpen(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_BASE_URL}/api/recon/import`, {
        method: "POST",
        body: form,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Mask import failed");
      const mask = await fetchNpy(`${API_BASE_URL}${body.mask_url}`);
      setFlowData(null);
      setPreviewMask({
        mask: mask.data as Uint8Array,
        nx: mask.shape[1],
        ny: mask.shape[0],
      });
      storeMaskForSurrogate(mask.data, mask.shape[1], mask.shape[0]);
      setActiveJobId(body.job_id);
      setUploadedFile(file.name);
      setClosestPreset(null);
      setContourPreview(`${API_BASE_URL}${body.preview_url}`);
      setSolverReady(true);
      setLastSolverAngle(null);
      setLastSolverSpeed(null);
      setScaleStatus("Normalized domain mask; physical scale is unknown");
      setDataSource("cached");
      setSimModeLabel("Imported geometry; simulation has not run");
      setSelectedCase({
        id: "custom_reconstruction",
        backendId: body.job_id,
        name: "Imported reconstruction",
        desc: "Solver mask imported from offline reconstruction",
        drag: 0,
        lift: 0,
        wake: 0,
        mode: "upload_ready",
        modeLabel: "Imported mask",
        explanation:
          "Inspect the imported obstacle mask, then run CPU LBM to compute its flow. No simulation results exist yet.",
        isCustom: true,
      });
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Could not import the mask",
      );
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const smoothDemo =
    isDemoModel(selectedCase.id) && !showVoxels && !previewMask;
  const geometryLabel = smoothDemo
    ? "Smooth demo model"
    : "Solver mask geometry";
  const displayGrid = flowData ?? previewMask;
  const sliceLimit = domainHeight(displayGrid) / 2 - 0.1;
  const sourceLabel = previewMask
    ? "Imported mask · ready to simulate"
    : !flowData
      ? "Illustrative preview · no loaded field"
      : effectiveDataSource === "computed"
        ? "Live 2D LBM field"
        : effectiveDataSource === "real_3d_lbm"
          ? "Offline 3D center slice"
          : effectiveDataSource === "offline_fallback"
            ? "Illustrative preview · offline"
            : "Cached 2D demonstration";
  const caseIcon = (id: string) =>
    id.includes("car") ? (
      <Car />
    ) : id.includes("drone") ? (
      <Box />
    ) : id.includes("airfoil") ? (
      <Plane />
    ) : (
      <Circle />
    );

  return (
    <div className={`app-container ${resultsOpen ? "" : "results-collapsed"}`}>
      <header className="app-header">
        <button
          className="brand"
          onClick={() => setShowLanding(true)}
          aria-label="AeroVoxel home"
        >
          <span className="logo-icon">AV</span>
          <strong>AeroVoxel</strong>
          <span className="logo-badge">LAB</span>
        </button>
        <div className="header-actions">
          <button
            className="secondary-button import-button"
            disabled={
              backendStatus !== "connected" || uploading || runningSolver
            }
            onClick={() => maskImportRef.current?.click()}
            aria-label="Import reconstruction mask"
          >
            <Box size={16} />
            <span>Import mask</span>
          </button>
          <span className="connection-status">
            <i className={backendStatus} />
            {backendStatus === "connected"
              ? "Backend connected"
              : "Offline preview"}
          </span>
          <button
            className="primary-button"
            aria-label="Upload capture"
            disabled={uploading || runningSolver}
            onClick={() => uploadInputRef.current?.click()}
          >
            <Upload size={17} />
            <span>Upload capture</span>
          </button>
        </div>
        <input
          ref={maskImportRef}
          type="file"
          accept=".npy"
          onChange={importMask}
          hidden
        />
        <input
          ref={uploadInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleFileUpload}
          hidden
        />
      </header>
      <nav className="object-picker" aria-label="Demo objects">
        {demoCases.map((c) => (
          <button
            key={c.backendId}
            disabled={uploading || runningSolver}
            className={`object-choice ${selectedCase.backendId === c.backendId ? "selected" : ""}`}
            aria-pressed={selectedCase.backendId === c.backendId}
            onClick={() => {
              resetToPresets();
              setSelectedCase(c);
              setDataSource(
                c.mode === "real_3d_lbm" ? "real_3d_lbm" : "cached",
              );
              setSimModeLabel(c.modeLabel);
              setMaskForSurrogate(null);
            }}
          >
            {caseIcon(c.id)}
            <span>
              <strong>
                {c.name.replace(" (Template)", "").replace(" (Validation)", "")}
              </strong>
              <small>{c.desc}</small>
            </span>
          </button>
        ))}
        {uploadedFile && (
          <button
            className="object-choice selected custom-choice"
            disabled={uploading || runningSolver}
            onClick={resetToPresets}
          >
            <FileImage />
            <span>
              <strong>{uploadedFile}</strong>
              <small>Custom capture · click to reset</small>
            </span>
          </button>
        )}
      </nav>
      <main className="experiment-workspace">
        <section className="viewer-section" aria-label="Wind tunnel">
          <WindTunnelViewer
            caseId={selectedCase.id}
            windSpeed={windSpeed}
            windAngle={windAngle}
            showStreamlines={showStreamlines}
            showPressure={showPressure}
            showWake={showWake}
            flowData={flowData}
            previewMask={previewMask}
            showVoxels={showVoxels}
            showSlicePlane={showSlicePlane}
            slicePosition={slicePosition}
            paused={paused}
            cameraView={cameraView}
            resetCamera={resetCamera}
            onFpsUpdate={setFps}
          />
          <div
            className="viewer-source"
            title="Cached fields keep their original inlet conditions. Re-run a custom simulation to apply wind parameters."
          >
            <span className="source-dot" />
            {sourceLabel}
            {loadingFlowData && " · Loading…"}
          </div>
          {flowData && effectiveDataSource !== "computed" && (
            <div className="field-hint">
              Cached inlet conditions · wind changes apply on a custom run
            </div>
          )}
          <div className="camera-toolbar" aria-label="Camera controls">
            <button
              aria-pressed={cameraView === "side"}
              className={cameraView === "side" ? "active" : ""}
              onClick={() => setCameraView("side")}
            >
              Side
            </button>
            <button
              aria-pressed={cameraView === "perspective"}
              className={cameraView === "perspective" ? "active" : ""}
              onClick={() => setCameraView("perspective")}
            >
              Perspective
            </button>
            <button
              onClick={() => setResetCamera((n) => n + 1)}
              aria-label="Reset camera"
            >
              <RotateCcw size={16} />
            </button>
            <button
              onClick={() => setResultsOpen((v) => !v)}
              aria-label={resultsOpen ? "Collapse results" : "Show results"}
              aria-expanded={resultsOpen}
            >
              {resultsOpen ? (
                <PanelRightClose size={17} />
              ) : (
                <PanelRightOpen size={17} />
              )}
            </button>
          </div>
          <div className="control-dock">
            <label className="dock-slider">
              <span>
                <Wind size={16} />
                Wind speed <output>{windSpeed} m/s</output>
              </span>
              <input
                aria-label="Wind speed"
                type="range"
                min="5"
                max="40"
                value={windSpeed}
                onChange={(e) => setWindSpeed(Number(e.target.value))}
              />
            </label>
            <label className="dock-slider">
              <span>
                <ArrowUpRight size={16} />
                Inlet angle <output>{windAngle}°</output>
              </span>
              <input
                aria-label="Inlet angle"
                type="range"
                min="-30"
                max="30"
                value={windAngle}
                onChange={(e) => setWindAngle(Number(e.target.value))}
              />
            </label>
            <ToggleRow
              label="Flow paths"
              active={showStreamlines}
              onToggle={() => setShowStreamlines((v) => !v)}
            />
            <ToggleRow
              label="Pressure"
              active={showPressure}
              onToggle={() => setShowPressure((v) => !v)}
            />
            <button
              className="pause-button"
              aria-pressed={paused}
              onClick={() => setPaused((v) => !v)}
            >
              {paused ? <Play size={17} /> : <Pause size={17} />}
              {paused ? "Play" : "Pause"}
            </button>
          </div>
          {(uploading || runningSolver) && (
            <div className="working-overlay" role="status">
              <span className="spinner" />
              <strong>
                {runningSolver
                  ? "Computing your flow field"
                  : PROCESSING_STEPS[processingStepIndex]}
              </strong>
              <span>
                {runningSolver
                  ? "600 steps · CPU LBM"
                  : "Preparing your capture"}
              </span>
            </div>
          )}
        </section>
        <aside
          className="results-panel"
          aria-label="Results and object details"
          hidden={!resultsOpen}
        >
          <div className="results-heading">
            <h2>Results</h2>
            <button
              aria-label="Collapse results"
              onClick={() => setResultsOpen(false)}
            >
              <PanelRightClose size={18} />
            </button>
          </div>
          <div className="results-origin">
            <Info size={15} />
            <span>
              {previewMask
                ? "Geometry preview"
                : effectiveDataSource === "computed"
                  ? "Educational solver estimates"
                  : selectedCase.metricsSource
                    ? "Cached CPU estimates"
                    : "Demo estimates"}
              <small>
                {previewMask
                  ? "No simulation has run yet"
                  : effectiveDataSource === "computed"
                    ? "Computed for your silhouette"
                    : selectedCase.metricsSource
                      ? "From this precomputed flow field"
                      : "Reference values, not a new solve"}
              </small>
            </span>
          </div>
          {!previewMask && (
            <>
              <MetricBox
                label="Drag coefficient (Cd)"
                value={selectedCase.drag.toFixed(2)}
                desc="Educational resistance estimate"
              />
              <MetricBox
                label="Lift coefficient (Cl)"
                value={selectedCase.lift.toFixed(2)}
                desc="Educational lift estimate"
              />
              <MetricBox
                label="Wake index"
                value={selectedCase.wake.toFixed(2)}
                desc="Qualitative wake indicator"
              />
            </>
          )}
          {previewMask && (
            <p className="notice">
              Geometry imported. Inspect the mask below, then run a simulation
              to get results.
            </p>
          )}
          {uploadError && (
            <p className="error-message" role="alert">
              {uploadError}
            </p>
          )}
          {uploadedFile && (
            <section className="capture-details">
              <h3>Your capture</h3>
              <p>{uploadedFile}</p>
              {contourPreview && (
                <img src={contourPreview} alt="Detected object contour" />
              )}
              {scaleStatus && <small>{scaleStatus}</small>}
              {activeJobId && (
                <button
                  className="primary-button"
                  disabled={runningSolver}
                  onClick={runLiveSimulation}
                >
                  {angleNeedsRerun
                    ? "Re-run with new parameters"
                    : solverReady
                      ? "Run 2D simulation"
                      : "Run again"}
                </button>
              )}
            </section>
          )}
          {angleNeedsRerun && (
            <p className="notice">
              Parameters changed. Re-run to update the computed field.
            </p>
          )}
          {backendStatus === "connected" && maskForSurrogate && (
            <section className="surrogate-section">
              <h3>ML surrogate</h3>
              <small>
                Fast drag proxy from the current mask. Fixed training
                conditions; independent of wind controls.
              </small>
              <button
                className="secondary-button"
                onClick={runSurrogatePredict}
                disabled={
                  effectiveSurrogateAvailable !== true ||
                  surrogateLoading ||
                  runningSolver ||
                  uploading
                }
              >
                {surrogateLoading ? "Predicting…" : "Predict drag proxy"}
              </button>
              {effectiveSurrogateAvailable === false && (
                <small>{effectiveSurrogateHint ?? "Model unavailable"}</small>
              )}
              {surrogateError && (
                <p className="error-message" role="alert">
                  {surrogateError}
                </p>
              )}
              {surrogatePred && (
                <div className="surrogate-value">
                  <strong>{surrogatePred.cd.toFixed(3)}</strong>
                  <small>{surrogatePred.model} · educational ML proxy</small>
                  <p>{surrogatePred.disclaimer}</p>
                </div>
              )}
            </section>
          )}
          <details className="physics-details">
            <summary>What am I seeing?</summary>
            <p>{selectedCase.explanation}</p>
            <p>
              {flowData
                ? smoothDemo
                  ? selectedCase.id === "sports_car" ||
                    selectedCase.id === "drone"
                    ? "Airflow is computed from this model's side silhouette on a coarse 2D grid. Enable voxel geometry to inspect the cells. The 3D view adds depth for inspection; the airflow remains 2D."
                    : "The smooth 3D demo model illustrates the object. Airflow follows the coarse 2D solver mask; enable voxel geometry to inspect that mask."
                  : "Airflow and voxel geometry use the loaded 2D solver mask."
                : "The offline scene is an illustrative flow preview, not a computed simulation."}
            </p>
          </details>
          <details className="physics-details">
            <summary>More view controls</summary>
            <ToggleRow
              label="Wake detail"
              active={showWake}
              onToggle={() => setShowWake((v) => !v)}
            />
            <ToggleRow
              label="Voxel geometry"
              active={showVoxels}
              onToggle={() => setShowVoxels((v) => !v)}
            />
            <ToggleRow
              label="Slice plane"
              active={showSlicePlane}
              onToggle={() => setShowSlicePlane((v) => !v)}
            />
            {showSlicePlane && (
              <label className="slice-label">
                Plane height{" "}
                <input
                  aria-label="Slice plane height"
                  type="range"
                  min={-sliceLimit}
                  max={sliceLimit}
                  step="0.1"
                  value={slicePosition}
                  onChange={(e) => setSlicePosition(Number(e.target.value))}
                />
              </label>
            )}
          </details>
          <button
            className="technical-toggle"
            onClick={() => setShowTechnicalDetails((v) => !v)}
            aria-expanded={showTechnicalDetails}
          >
            Technical details{" "}
            {showTechnicalDetails ? (
              <ChevronUp size={16} />
            ) : (
              <ChevronDown size={16} />
            )}
          </button>
          {showTechnicalDetails && (
            <div className="technical-details">
              <DetailRow label="Source" value={sourceLabel} />
              <DetailRow label="Geometry" value={geometryLabel} />
              {selectedCase.id.includes("car") && (
                <a
                  className="model-credit"
                  href="/models/ATTRIBUTION.md"
                  target="_blank"
                  rel="noreferrer"
                >
                  Car model credit
                </a>
              )}
              <DetailRow
                label="Grid"
                value={
                  displayGrid
                    ? `${displayGrid.nx} × ${displayGrid.ny}`
                    : "Procedural preview"
                }
              />
              <DetailRow label="Rendering" value={`${fps} FPS · WebGL`} />
              <p>{effectiveSimModeLabel}</p>
              <p>
                Prototype geometry and educational flow. Not certified
                aerodynamic analysis.
              </p>
            </div>
          )}
        </aside>
      </main>
      <footer className="workspace-footer">
        <span>
          <ShieldCheck size={14} />
          Educational prototype
        </span>
        <span>
          {selectedCase.name} ·{" "}
          {previewMask ? "Imported solver mask" : geometryLabel}
        </span>
        <button onClick={() => setShowLanding(true)}>
          <Home size={14} />
          Overview
        </button>
      </footer>
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
    <button
      type="button"
      role="switch"
      aria-checked={active}
      className={`toggle-item ${active ? "active" : ""}`}
      onClick={onToggle}
    >
      <span>{label}</span>
      <span className="toggle-switch" aria-hidden="true" />
    </button>
  );
}
function MetricBox({
  label,
  value,
  desc,
}: {
  label: string;
  value: string;
  desc: string;
}) {
  return (
    <div className="metric-box">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{desc}</small>
    </div>
  );
}
function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <span>{value}</span>
    </div>
  );
}
export default App;
