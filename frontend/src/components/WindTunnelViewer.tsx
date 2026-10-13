import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  createDemoModel,
  isDemoModel,
  cellSize,
  domainHeight,
  loadDetailedCar,
  hasDetailedCar,
  maskBounds,
} from "./demoModels";
import type { MaskGrid } from "./demoModels";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface FlowData {
  velocity: Float32Array;
  pressure: Float32Array;
  mask: Uint8Array;
  nx: number;
  ny: number;
}
interface Props {
  caseId: string;
  windSpeed: number;
  windAngle: number;
  showStreamlines: boolean;
  showPressure: boolean;
  showWake: boolean;
  flowData: FlowData | null;
  previewMask?: Pick<FlowData, "mask" | "nx" | "ny"> | null;
  showVoxels: boolean;
  showSlicePlane: boolean;
  slicePosition: number;
  paused?: boolean;
  cameraView?: "side" | "perspective";
  resetCamera?: number;
  onFpsUpdate?: (fps: number) => void;
}

// Flow paths share the exact XY coordinate mapping with the solver mask.
const fieldPoint = (x: number, y: number, flow: FlowData) => {
  const fx = THREE.MathUtils.clamp(
    ((x + 8) / 16) * (flow.nx - 1),
    0,
    flow.nx - 1,
  );
  const fy = THREE.MathUtils.clamp(
    (flow.ny - 1) / 2 - y / cellSize(flow),
    0,
    flow.ny - 1,
  );
  const nearest = Math.round(fy) * flow.nx + Math.round(fx);
  if (flow.mask[nearest]) return { solid: true, vx: 0, vy: 0, p: 0 };
  const x0 = Math.floor(fx),
    y0 = Math.floor(fy),
    tx = fx - x0,
    ty = fy - y0;
  let vx = 0,
    vy = 0,
    pressure = 0,
    weight = 0;
  // Interpolate fluid neighbours only; solid velocities must not smear into air.
  for (let dy = 0; dy <= 1; dy++)
    for (let dx = 0; dx <= 1; dx++) {
      const i =
        Math.min(y0 + dy, flow.ny - 1) * flow.nx +
        Math.min(x0 + dx, flow.nx - 1);
      const w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty);
      if (flow.mask[i]) continue;
      vx += flow.velocity[i] * w;
      vy -= flow.velocity[flow.nx * flow.ny + i] * w;
      pressure += flow.pressure[i] * w;
      weight += w;
    }
  return {
    solid: weight < 1e-6,
    vx: vx / Math.max(weight, 1e-6),
    vy: vy / Math.max(weight, 1e-6),
    p: pressure / Math.max(weight, 1e-6),
  };
};
const pressureColor = (p: number, span: number) => {
  const t = THREE.MathUtils.clamp(p / span, -1, 1);
  return new THREE.Color(0x32d8ee).lerp(
    new THREE.Color(t > 0 ? 0xff765c : 0x4486ff),
    Math.abs(t),
  );
};

export function WindTunnelViewer(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  const [renderError, setRenderError] = useState(false);
  const [carReady, setCarReady] = useState(hasDetailedCar);
  const [assetError, setAssetError] = useState(false);
  useEffect(() => {
    if (!props.caseId.includes("car")) return;
    let active = true;
    loadDetailedCar()
      .then(() => {
        if (active) setCarReady(true);
      })
      .catch(() => {
        if (active) setAssetError(true);
      });
    return () => {
      active = false;
    };
  }, [props.caseId]);
  const cameraControl = useRef<(() => void) | null>(null);
  useEffect(() => {
    latest.current = props;
  });
  useEffect(() => {
    cameraControl.current?.();
  }, [props.cameraView, props.resetCamera]);

  useEffect(() => {
    const host = container.current;
    if (!host) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      queueMicrotask(() => setRenderError(true));
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    renderer.setClearColor(0x091522);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    // Geometry and lighting are static between object/overlay changes.
    renderer.shadowMap.autoUpdate = false;
    host.replaceChildren(renderer.domElement);
    const scene = new THREE.Scene();
    const environment = new RoomEnvironment();
    const pmrem = new THREE.PMREMGenerator(renderer);
    const environmentTarget = pmrem.fromScene(environment, 0.04);
    scene.environment = environmentTarget.texture;
    environment.dispose();
    pmrem.dispose();
    const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 5;
    controls.maxDistance = 28;
    const focus = new THREE.Vector3();
    const fitSize = new THREE.Vector3(4.8, 2, 2);
    let framedCase = "";
    let framedGrid: MaskGrid | null | undefined;
    const placeCamera = () => {
      const side = latest.current.cameraView !== "perspective";
      const tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const distance = Math.max(
        5,
        (fitSize.y * 0.9) / tangent,
        (fitSize.x * 1.6 + fitSize.z * 0.35) / (2 * tangent * camera.aspect),
      );
      camera.position.set(
        focus.x + (side ? 0 : -distance * 0.7),
        focus.y + (side ? 0 : distance * 0.32),
        side ? distance : distance * 0.85,
      );
      controls.target.copy(focus);
      controls.update();
    };
    cameraControl.current = placeCamera;
    placeCamera();
    scene.add(new THREE.HemisphereLight(0xdceaff, 0x1a2830, 0.7));
    const key = new THREE.DirectionalLight(0xfff3df, 1.7);
    key.position.set(-3, 6, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -12;
    key.shadow.camera.right = 12;
    key.shadow.camera.top = 12;
    key.shadow.camera.bottom = -12;
    key.shadow.camera.far = 40;
    key.shadow.bias = -0.0002;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aacd, 0.8);
    rim.position.set(4, 3, -5);
    scene.add(rim);
    const grid = new THREE.GridHelper(24, 24, 0x37576d, 0x1a3041);
    grid.position.y = -2.08;
    (Array.isArray(grid.material) ? grid.material : [grid.material]).forEach(
      (m) => {
        m.transparent = true;
        m.opacity = 0.22;
        m.depthWrite = false;
      },
    );
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.MeshBasicMaterial({ color: 0x091522, toneMapped: false }),
    );
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);
    const shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.ShadowMaterial({
        opacity: 0.24,
        color: 0x000000,
        depthWrite: false,
      }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.receiveShadow = true;
    scene.add(shadow);
    const tunnel = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(16, 4, 4)),
      new THREE.LineBasicMaterial({
        color: 0x29475b,
        transparent: true,
        opacity: 0.14,
      }),
    );
    scene.add(tunnel);
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(16, 4),
      new THREE.MeshBasicMaterial({
        color: 0x33d7ed,
        opacity: 0.08,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    plane.rotation.x = -Math.PI / 2;
    scene.add(plane);
    const object = new THREE.Group();
    scene.add(object);
    const trails = new THREE.Group();
    scene.add(trails);
    let previousKey = "";
    let detailedLoaded = hasDetailedCar();
    const tracerCount = 600,
      historyLength = 30;
    type Tracer = {
      x: number;
      y: number;
      z: number;
      age: number;
      life: number;
      history: Float32Array;
    };
    let tracers: Tracer[] = [];
    let smoke: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> | null =
      null;
    let pressureSpan = 0.02,
      previewA = 2.8,
      previewB = 0.8,
      domainHalf = 4;
    let lastTraceCase = "";
    let lastTraceFlow: FlowData | null = null;
    let lastTraceMask: Props["previewMask"] = null;
    const viewDirection = new THREE.Vector3();
    let seedMin = -2,
      seedMax = 2;
    const sample = (x: number, y: number) => {
      const p = latest.current;
      if (p.flowData) {
        const v = fieldPoint(x, y, p.flowData);
        return { ...v, vx: v.vx / 0.08, vy: v.vy / 0.08 };
      }
      // Elliptic potential-flow preview. Computed fields get no artificial noise.
      const xx = x / previewA,
        yy = y / previewB,
        r2 = xx * xx + yy * yy;
      if (r2 < 1.05) return { solid: true, vx: 0, vy: 0, p: 0 };
      const inverse = 1 / Math.max(r2 * r2, 1e-6);
      const vx = 1 - (xx * xx - yy * yy) * inverse;
      let vy = (-2 * xx * yy * inverse * previewB) / previewA;
      if (p.showWake && x > previewA) {
        const envelope =
          Math.exp(-(x - previewA) / 3) * Math.exp((-y * y) / 0.7);
        vy += Math.sin(clock * 2.2 - (x - previewA) * 2.5) * 0.14 * envelope;
      }
      return { solid: false, vx, vy, p: 0.018 * (1 - vx * vx - vy * vy) };
    };
    const respawn = (tracer: Tracer, fillDomain: boolean) => {
      for (let attempt = 0; attempt < 24; attempt++) {
        tracer.x = fillDomain
          ? -7.85 + Math.random() * 15.6
          : -7.85 + Math.random() * 0.15;
        // A smoke rake puts more visualization markers in the interaction band.
        // Seeding changes visibility only; velocity integration remains field-driven.
        tracer.y =
          Math.random() < 0.72
            ? seedMin + Math.random() * (seedMax - seedMin)
            : -domainHalf * 0.95 + Math.random() * domainHalf * 1.9;
        if (!sample(tracer.x, tracer.y).solid) break;
      }
      tracer.z = (Math.random() - 0.5) * 2.6;
      tracer.age = fillDomain ? 0.5 : 0;
      tracer.life = 7 + Math.random() * 6;
      let tailX = tracer.x,
        tailY = tracer.y;
      for (let h = 0; h < historyLength; h++) {
        tracer.history[h * 3] = tailX;
        tracer.history[h * 3 + 1] = tailY;
        tracer.history[h * 3 + 2] = tracer.z;
        if (fillDomain) {
          const v = sample(tailX, tailY);
          const x = tailX - v.vx * 0.035,
            y = tailY - v.vy * 0.035;
          if (
            x > -8 &&
            Math.abs(y) < domainHalf * 0.99 &&
            !sample(x, y).solid
          ) {
            tailX = x;
            tailY = y;
          }
        }
      }
    };
    const release = (group: THREE.Group) => {
      group.traverse((child) => {
        if (
          child instanceof THREE.Mesh ||
          child instanceof THREE.Line ||
          child instanceof THREE.Points
        ) {
          if (!child.userData.sharedGeometry) child.geometry.dispose();
          const materials = Array.isArray(child.material)
            ? child.material
            : [child.material];
          materials.forEach((m) => m.dispose());
        }
      });
      group.clear();
    };
    const rebuild = () => {
      release(object);
      release(trails);
      const p = latest.current;
      const silver = new THREE.MeshStandardMaterial({
        color: 0xd5e5ec,
        metalness: 0.25,
        roughness: 0.38,
        wireframe: p.showVoxels,
      });
      const field = p.flowData ?? p.previewMask;
      domainHalf = domainHeight(field) / 2;
      grid.position.y = -domainHalf - 0.02;
      floor.position.y = -domainHalf - 0.06;
      shadow.position.y = -domainHalf - 0.04;
      tunnel.geometry.dispose();
      tunnel.geometry = new THREE.EdgesGeometry(
        new THREE.BoxGeometry(16, domainHalf * 2, 4),
      );
      if (isDemoModel(p.caseId) && !p.showVoxels && !p.previewMask) {
        silver.dispose();
        object.add(createDemoModel(p.caseId, p.flowData));
      } else if (field) {
        const step = cellSize(field);
        const count = Array.from(field.mask).reduce(
          (n, value) => n + (value > 0 ? 1 : 0),
          0,
        );
        const geometry = new THREE.BoxGeometry(step, step, 0.65);
        const cells = new THREE.InstancedMesh(geometry, silver, count);
        const matrix = new THREE.Matrix4();
        let index = 0;
        for (let y = 0; y < field.ny; y++)
          for (let x = 0; x < field.nx; x++) {
            if (!field.mask[y * field.nx + x]) continue;
            matrix.makeTranslation(x * step - 8, domainHalf - y * step, 0);
            cells.setMatrixAt(index++, matrix);
          }
        object.add(cells);
      } else {
        // Unknown/custom geometry has no inferred vehicle classification.
        object.add(
          new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.8, 0.65), silver),
        );
      }
      object.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      const flow = p.flowData;
      pressureSpan = 0.02;
      const bounds = new THREE.Box3().setFromObject(object);
      const interaction = field ? maskBounds(field) : bounds;
      seedMin = Math.max(-domainHalf * 0.95, interaction.min.y - 0.7);
      seedMax = Math.min(domainHalf * 0.95, interaction.max.y + 0.7);
      if (framedCase !== p.caseId || framedGrid !== field) {
        bounds.getCenter(focus);
        bounds.getSize(fitSize);
        focus.x += fitSize.x * 0.15;
        framedCase = p.caseId;
        framedGrid = field;
        placeCamera();
      }
      const depth = Math.max(4, bounds.max.z - bounds.min.z + 2);
      tunnel.geometry.dispose();
      tunnel.geometry = new THREE.EdgesGeometry(
        new THREE.BoxGeometry(16, domainHalf * 2, depth),
      );
      plane.geometry.dispose();
      plane.geometry = new THREE.PlaneGeometry(16, depth);
      previewA =
        Math.max(Math.abs(bounds.min.x), Math.abs(bounds.max.x)) + 0.08;
      previewB =
        Math.max(Math.abs(bounds.min.y), Math.abs(bounds.max.y)) + 0.08;
      if (flow) {
        const magnitudes = Array.from(flow.pressure)
          .filter((_, i) => !flow.mask[i])
          .map(Math.abs)
          .sort((a, b) => a - b);
        pressureSpan = Math.max(
          magnitudes[Math.floor(magnitudes.length * 0.95)] ?? 0.02,
          1e-5,
        );
      }
      const positions: number[] = [];
      const colors: number[] = [];
      for (let seed = 0; seed < (p.previewMask ? 0 : 26); seed++) {
        const y0 = -domainHalf * 0.94 + (seed / 25) * domainHalf * 1.88;
        for (const z of [-0.65, 0.65]) {
          const path: THREE.Vector3[] = [];
          let x = -7.9,
            y = y0;
          for (let step = 0; step < 340; step++) {
            if (x > 7.95 || x < -8 || Math.abs(y) > domainHalf * 0.995) break;
            let pressure: number;
            if (flow) {
              const v = fieldPoint(x, y, flow);
              if (v.solid || !Number.isFinite(v.vx + v.vy + v.p)) break;
              path.push(new THREE.Vector3(x, y, z));
              pressure = v.p;
              const magnitude = Math.hypot(v.vx, v.vy);
              if (magnitude < 1e-5) break;
              x += (v.vx / magnitude) * 0.055;
              y += (v.vy / magnitude) * 0.055;
            } else {
              // Illustrative potential-flow-like preview, never a simulation result.
              const sign = y0 < 0 ? -1 : 1;
              const spread = p.caseId.includes("airfoil") ? 0.8 : 1.35;
              const bend =
                sign *
                spread *
                Math.exp((-x * x) / 5.4) *
                Math.exp(-Math.abs(y0) / 1.4);
              y = y0 + bend;
              if (p.showWake && x > 1.8)
                y +=
                  Math.sin(x * 3 + seed * 0.12) *
                  0.06 *
                  Math.exp(-(x - 1.8) / 3);
              path.push(new THREE.Vector3(x, y, z));
              pressure =
                0.018 *
                (Math.exp(-((x + 1.6) ** 2) / 0.7) - Math.exp((-x * x) / 3)) *
                Math.exp(-Math.abs(y0));
              x += 0.055;
            }
            if (path.length > 1) {
              const a = path[path.length - 2],
                b = path[path.length - 1];
              positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
              const c = p.showPressure
                ? pressureColor(pressure, pressureSpan)
                : new THREE.Color(0x3ddbea);
              colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
            }
          }
        }
      }
      if (flow && p.showWake) {
        // Optional arrows expose slow/reversed downstream cells from the field,
        // without adding synthetic turbulence to computed data.
        for (let gy = 4; gy < flow.ny - 4; gy += 8) {
          for (let gx = Math.floor(flow.nx / 2); gx < flow.nx - 4; gx += 8) {
            const i = gy * flow.nx + gx;
            const vx = flow.velocity[i],
              vy = -flow.velocity[flow.nx * flow.ny + i];
            if (flow.mask[i] || vx >= 0.05 || Math.hypot(vx, vy) < 1e-5)
              continue;
            const direction = new THREE.Vector3(vx, vy, 0).normalize();
            trails.add(
              new THREE.ArrowHelper(
                direction,
                new THREE.Vector3(
                  (gx / (flow.nx - 1)) * 16 - 8,
                  domainHalf - gy * cellSize(flow),
                  0.9,
                ),
                0.22,
                0x78b9fa,
                0.06,
                0.04,
              ),
            );
          }
        }
      }
      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3),
      );
      lineGeometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(colors, 3),
      );
      trails.add(
        new THREE.LineSegments(
          lineGeometry,
          new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.045,
            toneMapped: false,
          }),
        ),
      );
      if (
        lastTraceCase !== p.caseId ||
        lastTraceFlow !== p.flowData ||
        lastTraceMask !== p.previewMask
      ) {
        tracers = p.previewMask
          ? []
          : Array.from({ length: tracerCount }, () => {
              const tracer: Tracer = {
                x: 0,
                y: 0,
                z: 0,
                age: 0,
                life: 0,
                history: new Float32Array(historyLength * 3),
              };
              respawn(tracer, true);
              return tracer;
            });
        lastTraceCase = p.caseId;
        lastTraceFlow = p.flowData;
        lastTraceMask = p.previewMask;
      }
      const smokeGeometry = new THREE.BufferGeometry();
      const vertexCount = tracers.length * historyLength * 2;
      smokeGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(
          new Float32Array(vertexCount * 3),
          3,
        ).setUsage(THREE.DynamicDrawUsage),
      );
      smokeGeometry.setAttribute(
        "color",
        new THREE.BufferAttribute(
          new Float32Array(vertexCount * 3),
          3,
        ).setUsage(THREE.DynamicDrawUsage),
      );
      smokeGeometry.setAttribute(
        "trailOpacity",
        new THREE.BufferAttribute(new Float32Array(vertexCount), 1).setUsage(
          THREE.DynamicDrawUsage,
        ),
      );
      const uv = new Float32Array(vertexCount * 2),
        indices: number[] = [];
      for (let i = 0; i < tracers.length; i++)
        for (let h = 0; h < historyLength; h++) {
          const v = (i * historyLength + h) * 2;
          uv[v * 2] = h / (historyLength - 1);
          uv[v * 2 + 1] = -1;
          uv[(v + 1) * 2] = h / (historyLength - 1);
          uv[(v + 1) * 2 + 1] = 1;
          if (h < historyLength - 1)
            indices.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
        }
      smokeGeometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
      smokeGeometry.setIndex(indices);
      smoke = new THREE.Mesh(
        smokeGeometry,
        new THREE.ShaderMaterial({
          vertexColors: true,
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          blending: THREE.NormalBlending,
          vertexShader: `attribute float trailOpacity; varying vec3 vColor; varying float vOpacity; varying vec2 vUv;
          void main() { vColor = color; vOpacity = trailOpacity; vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
          fragmentShader: `varying vec3 vColor; varying float vOpacity; varying vec2 vUv;
          void main() { float edge = exp(-vUv.y * vUv.y * 5.0);
            float head = smoothstep(0.0, 0.12, vUv.x);
            gl_FragColor = vec4(vColor, edge * head * vOpacity * 0.13);
            #include <tonemapping_fragment>
            #include <colorspace_fragment>
          }`,
        }),
      );
      smoke.frustumCulled = false;
      trails.add(smoke);
      renderer.shadowMap.needsUpdate = true;
    };
    const resize = () => {
      const width = Math.max(host.clientWidth, 1),
        height = Math.max(host.clientHeight, 1);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      if (width <= 550) placeCamera();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let frame = 0,
      lastTime = performance.now(),
      clock = 0,
      fpsTime = lastTime,
      frames = 0;
    const animate = (now: number) => {
      const p = latest.current;
      const assetReady = hasDetailedCar();
      if (assetReady !== detailedLoaded) {
        detailedLoaded = assetReady;
        framedCase = "";
      }
      const key = `${p.caseId}|${p.showVoxels}|${p.showPressure}|${p.showWake}|${p.flowData?.nx ?? ""}|${assetReady}`;
      if (
        key !== previousKey ||
        activeFlow !== p.flowData ||
        activeMask !== p.previewMask
      ) {
        previousKey = key;
        activeFlow = p.flowData;
        activeMask = p.previewMask;
        rebuild();
      }
      const dt = Math.max(0, Math.min((now - lastTime) / 1000, 0.05));
      lastTime = now;
      if (!p.paused) clock += (dt * Math.max(p.windSpeed, 1)) / 15;
      if (smoke) {
        const geometry = smoke.geometry;
        const positions = geometry.attributes.position.array as Float32Array;
        const colors = geometry.attributes.color.array as Float32Array;
        const opacity = geometry.attributes.trailOpacity.array as Float32Array;
        camera.getWorldDirection(viewDirection);
        const speed = (2.3 * Math.max(p.windSpeed, 1)) / 15;
        const substeps = Math.max(1, Math.ceil(dt / 0.012));
        tracers.forEach((tracer, i) => {
          if (!p.paused) {
            tracer.age += dt;
            let reset = tracer.age > tracer.life;
            for (let step = 0; step < substeps && !reset; step++) {
              const v = sample(tracer.x, tracer.y),
                timeStep = (dt / substeps) * speed;
              const midpoint = sample(
                tracer.x + (v.vx * timeStep) / 2,
                tracer.y + (v.vy * timeStep) / 2,
              );
              const x = tracer.x + midpoint.vx * timeStep,
                y = tracer.y + midpoint.vy * timeStep;
              const end = sample(x, y);
              if (
                v.solid ||
                midpoint.solid ||
                end.solid ||
                !Number.isFinite(x + y) ||
                x < -8 ||
                x > 7.95 ||
                Math.abs(y) > domainHalf * 0.99
              )
                reset = true;
              else {
                tracer.x = x;
                tracer.y = y;
              }
            }
            if (reset) respawn(tracer, false);
            tracer.history.copyWithin(3, 0, (historyLength - 1) * 3);
            tracer.history[0] = tracer.x;
            tracer.history[1] = tracer.y;
            tracer.history[2] = tracer.z;
          }
          const history = tracer.history;
          const local = sample(tracer.x, tracer.y);
          const c = new THREE.Color(0xbdcbd3);
          if (p.showPressure)
            c.lerp(pressureColor(local.p, pressureSpan), 0.32);
          const birth = Math.min(tracer.age / 0.3, 1),
            death = Math.min((tracer.life - tracer.age) / 0.6, 1);
          for (let h = 0; h < historyLength; h++) {
            const j = h * 3,
              neighbour = (h === 0 ? 1 : h - 1) * 3;
            const tangentSign = h === 0 ? -1 : 1;
            const tx = (history[neighbour] - history[j]) * tangentSign,
              ty = (history[neighbour + 1] - history[j + 1]) * tangentSign;
            let sx = -viewDirection.z * ty,
              sy = viewDirection.z * tx,
              sz = viewDirection.x * ty - viewDirection.y * tx;
            const norm = Math.hypot(sx, sy, sz) || 1,
              fade = 1 - h / (historyLength - 1);
            const width = (0.045 + 0.1 * Math.sqrt(1 - fade)) * Math.sqrt(fade);
            sx *= width / norm;
            sy *= width / norm;
            sz *= width / norm;
            for (let side = 0; side < 2; side++) {
              const vertex = (i * historyLength + h) * 2 + side,
                offset = vertex * 3,
                sign = side === 0 ? -1 : 1;
              positions[offset] = history[j] + sx * sign;
              positions[offset + 1] = history[j + 1] + sy * sign;
              positions[offset + 2] = history[j + 2] + sz * sign;
              colors[offset] = c.r;
              colors[offset + 1] = c.g;
              colors[offset + 2] = c.b;
              opacity[vertex] = Math.max(
                0,
                Math.pow(fade, 1.4) * birth * death,
              );
            }
          }
        });
        geometry.attributes.position.needsUpdate = true;
        geometry.attributes.color.needsUpdate = true;
        geometry.attributes.trailOpacity.needsUpdate = true;
      }
      // The field and its mask stay aligned; inlet angle is already in computed data.
      trails.visible = p.showStreamlines;
      plane.visible = p.showSlicePlane;
      plane.position.y = p.slicePosition;
      controls.update();
      renderer.render(scene, camera);
      frames++;
      if (now - fpsTime > 1000) {
        p.onFpsUpdate?.(Math.round((frames * 1000) / (now - fpsTime)));
        frames = 0;
        fpsTime = now;
      }
      frame = requestAnimationFrame(animate);
    };
    let activeFlow: FlowData | null = null;
    let activeMask: Props["previewMask"] = null;
    frame = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      cameraControl.current = null;
      release(object);
      release(trails);
      grid.geometry.dispose();
      (Array.isArray(grid.material) ? grid.material : [grid.material]).forEach(
        (m) => m.dispose(),
      );
      tunnel.geometry.dispose();
      (tunnel.material as THREE.Material).dispose();
      plane.geometry.dispose();
      (plane.material as THREE.Material).dispose();
      shadow.geometry.dispose();
      (shadow.material as THREE.Material).dispose();
      floor.geometry.dispose();
      (floor.material as THREE.Material).dispose();
      environmentTarget.dispose();
      key.shadow.dispose();
      renderer.dispose();
      host.replaceChildren();
    };
  }, []);
  return (
    <div className="tunnel-view-container">
      <div
        ref={container}
        className="canvas-container"
        role="img"
        aria-label={`${props.caseId} airflow visualization. ${props.previewMask ? "Imported obstacle mask; simulation has not run." : props.flowData ? "Loaded simulation field." : "Illustrative procedural preview."}`}
      />
      {props.caseId.includes("car") && !carReady && !assetError && (
        <div className="working-overlay" role="status">
          <span className="spinner" />
          <strong>Preparing the car model</strong>
        </div>
      )}
      {props.caseId.includes("car") && assetError && (
        <div className="viewer-guidance">
          Detailed model unavailable · basic preview
        </div>
      )}
      {renderError && (
        <div className="viewer-error">
          WebGL is unavailable. Try a browser with hardware acceleration
          enabled.
        </div>
      )}
      {props.showPressure && !props.previewMask && (
        <div className="pressure-legend">
          <span>
            Relative pressure {props.flowData ? "(field)" : "(illustrative)"}
          </span>
          <div className="legend-bar" />
          <div className="legend-labels">
            <span>Low (−)</span>
            <span>Neutral</span>
            <span>High (+)</span>
          </div>
        </div>
      )}
      <div className="viewer-guidance">Drag to orbit · Scroll to zoom</div>
    </div>
  );
}
