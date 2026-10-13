import * as THREE from "three";
import sharedGeometry from "../../../shared/demoGeometry.json" with { type: "json" };

export type MaskGrid = { mask: Uint8Array; nx: number; ny: number };
export const cellSize = (grid: Pick<MaskGrid, "nx">) =>
  16 / Math.max(grid.nx - 1, 1);
export const domainHeight = (grid?: Pick<MaskGrid, "nx" | "ny"> | null) =>
  grid ? cellSize(grid) * (grid.ny - 1) : 8;
export function maskBounds(grid: MaskGrid) {
  const step = cellSize(grid);
  const bounds = new THREE.Box3();
  for (let y = 0; y < grid.ny; y++)
    for (let x = 0; x < grid.nx; x++) {
      if (!grid.mask[y * grid.nx + x]) continue;
      bounds.expandByPoint(
        new THREE.Vector3(x * step - 8, ((grid.ny - 1) / 2 - y) * step, 0),
      );
    }
  bounds.expandByScalar(step / 2);
  return bounds;
}
export const isDemoModel = (id: string) =>
  /^(sports_car|drone|airfoil|cylinder|sphere_3d)(_v1)?$/.test(id);

// Original finite-trailing-edge NACA four-digit thickness distribution, as in
// the cache generator. Reference: https://tmbwg.github.io/turbmodels/naca0012_val.html
export const naca0012HalfThickness = (x: number) =>
  0.6 *
  (0.2969 * Math.sqrt(x) -
    0.126 * x -
    0.3516 * x * x +
    0.2843 * x * x * x -
    0.1015 * x * x * x * x);

function mesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  position?: [number, number, number],
) {
  const result = new THREE.Mesh(geometry, material);
  if (position) result.position.set(...position);
  return result;
}
function roundedBox(
  w: number,
  h: number,
  d: number,
  radius: number,
  material: THREE.Material,
) {
  const r = Math.min(radius, w / 2, h / 2, d / 2);
  const shape = new THREE.Shape();
  shape.moveTo(-w / 2 + r, -h / 2);
  shape.lineTo(w / 2 - r, -h / 2);
  shape.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  shape.lineTo(w / 2, h / 2 - r);
  shape.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  shape.lineTo(-w / 2 + r, h / 2);
  shape.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  shape.lineTo(-w / 2, -h / 2 + r);
  shape.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: d - 2 * r,
    bevelEnabled: true,
    bevelSize: r,
    bevelThickness: r,
    bevelSegments: 3,
    curveSegments: 12,
    steps: 1,
  });
  geometry.translate(0, 0, -(d - 2 * r) / 2);
  return mesh(geometry, material);
}
let carTemplate: THREE.Group | null = null;
let carPromise: Promise<void> | null = null;
export const hasDetailedCar = () => carTemplate !== null;

export function installDetailedCar(source: THREE.Group) {
  // Bounds must follow drawn triangle indices, not unused accessor vertices.
  // Some sample-asset primitives share larger position buffers.
  const checked = new Set<THREE.BufferGeometry>();
  source.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || checked.has(child.geometry)) return;
    checked.add(child.geometry);
    const positions = child.geometry.attributes.position,
      index = child.geometry.index;
    const bounds = new THREE.Box3(),
      point = new THREE.Vector3();
    for (let i = 0; i < (index?.count ?? positions.count); i++)
      bounds.expandByPoint(
        point.fromBufferAttribute(positions, index ? index.getX(i) : i),
      );
    child.geometry.boundingBox = bounds;
  });
  source.rotation.y -= Math.PI / 2;
  source.updateMatrixWorld(true);
  let headlightsX = 0,
    headlights = 0;
  source.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const materials = Array.isArray(child.material)
      ? child.material
      : [child.material];
    if (materials.some((m) => m.name === "Headlight")) {
      headlightsX += new THREE.Box3()
        .setFromObject(child)
        .getCenter(new THREE.Vector3()).x;
      headlights++;
    }
  });
  if (headlights && headlightsX / headlights > 0) {
    source.rotation.y += Math.PI;
    source.updateMatrixWorld(true);
  }
  const initial = new THREE.Box3().setFromObject(source);
  source.scale.multiplyScalar(4.9 / initial.getSize(new THREE.Vector3()).x);
  source.updateMatrixWorld(true);
  const center = new THREE.Box3()
    .setFromObject(source)
    .getCenter(new THREE.Vector3());
  source.position.sub(center);
  source.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const materials = Array.isArray(child.material)
      ? child.material
      : [child.material];
    for (const material of materials) {
      if (!(material instanceof THREE.MeshStandardMaterial)) continue;
      material.envMapIntensity = 0.65;
      if (material.name.startsWith("Paint 1")) {
        material.color.setHex(0x718895);
        material.metalness = 0.65;
        material.roughness = 0.36;
        material.normalScale.setScalar(0.035);
        if (material instanceof THREE.MeshPhysicalMaterial) {
          material.clearcoat = 1;
          material.clearcoatRoughness = 0.12;
        }
      } else if (material.name.startsWith("Paint 2")) {
        material.color.setHex(0x243843);
        material.metalness = 0.6;
        material.roughness = 0.32;
        material.normalScale.setScalar(0.035);
      } else if (material.name === "Glass") {
        material.color.setHex(0x1b3040);
        material.metalness = 0.08;
        material.roughness = 0.1;
        material.transparent = true;
        material.opacity = 0.72;
        material.depthWrite = false;
        if (material instanceof THREE.MeshPhysicalMaterial)
          material.transmission = 0;
      } else if (
        material.name.startsWith("Interior") ||
        ["Floormat", "Dashboard", "Panel Sides"].includes(material.name)
      ) {
        material.color.setHex(0x19212a);
        material.metalness = 0.05;
        material.roughness = 0.8;
      } else if (material.name.startsWith("Rim")) {
        material.color.setHex(0x3d4e5a);
        material.metalness = 0.8;
        material.roughness = 0.38;
      } else if (material.name.startsWith("Tire")) {
        material.color.setHex(0x131a20);
        material.metalness = 0;
        material.roughness = 0.87;
      }
    }
  });
  carTemplate = source;
}
export function loadDetailedCar() {
  if (!carPromise)
    carPromise = import("three/examples/jsm/loaders/GLTFLoader.js")
      .then(({ GLTFLoader }) =>
        new GLTFLoader().loadAsync("/models/concept-car.glb"),
      )
      .then((gltf) => {
        installDetailedCar(gltf.scene);
      });
  return carPromise;
}

function detailedCarModel() {
  if (!carTemplate) return carModel();
  const model = carTemplate.clone(true);
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.userData.sharedGeometry = true;
    child.material = Array.isArray(child.material)
      ? child.material.map((m) => m.clone())
      : child.material.clone();
  });
  const group = new THREE.Group();
  group.add(model);
  return group;
}

function carModel() {
  const group = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({
    color: 0x97cbdc,
    metalness: 0.48,
    roughness: 0.3,
  });
  const glass = new THREE.MeshStandardMaterial({
    color: 0x153847,
    metalness: 0.35,
    roughness: 0.18,
  });
  const rubber = new THREE.MeshStandardMaterial({
    color: 0x111c26,
    roughness: 0.87,
  });
  const chrome = new THREE.MeshStandardMaterial({
    color: 0xdfe9ef,
    metalness: 0.7,
    roughness: 0.25,
  });
  // A lofted body has a continuous curved hood, fenders and rear deck.
  const sections = [
    [-2.45, 0.3, 0.64, 0.66],
    [-2.2, 0.2, 0.85, 0.93],
    [-1.65, 0.18, 0.98, 1],
    [-0.95, 0.18, 1.0, 1.02],
    [0.55, 0.18, 0.92, 1.02],
    [1.55, 0.18, 0.86, 0.98],
    [2.2, 0.23, 0.73, 0.86],
    [2.4, 0.32, 0.64, 0.68],
  ];
  const positions: number[] = [],
    indices: number[] = [],
    ring = 40;
  for (const [x, bottom, top, halfWidth] of sections)
    for (let j = 0; j < ring; j++) {
      const angle = (j / ring) * Math.PI * 2;
      positions.push(
        x,
        (top + bottom) / 2 + (Math.cos(angle) * (top - bottom)) / 2,
        Math.sin(angle) * halfWidth,
      );
    }
  for (let i = 0; i < sections.length - 1; i++)
    for (let j = 0; j < ring; j++) {
      const a = i * ring + j,
        b = i * ring + ((j + 1) % ring),
        c = (i + 1) * ring + j,
        d = (i + 1) * ring + ((j + 1) % ring);
      indices.push(a, b, c, b, d, c);
    }
  // Close the front and rear without open, unlit cavities.
  for (const end of [0, sections.length - 1]) {
    const [x, bottom, top] = sections[end],
      center = positions.length / 3;
    positions.push(x, (bottom + top) / 2, 0);
    for (let j = 0; j < ring; j++) {
      const a = end * ring + j,
        b = end * ring + ((j + 1) % ring);
      if (end === 0) indices.push(center, b, a);
      else indices.push(center, a, b);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  group.add(mesh(geometry, paint));
  // Sloping windshield, curved roof and rear window are distinct from the body.
  const cabin = new THREE.Shape();
  cabin.moveTo(-1.25, 0.8);
  cabin.bezierCurveTo(-0.96, 0.94, -0.8, 1.65, -0.4, 1.72);
  cabin.lineTo(0.52, 1.7);
  cabin.bezierCurveTo(0.92, 1.68, 1.1, 1.12, 1.45, 0.83);
  cabin.closePath();
  const cabinGeometry = new THREE.ExtrudeGeometry(cabin, {
    depth: 1.58,
    bevelEnabled: true,
    bevelSize: 0.035,
    bevelThickness: 0.035,
    bevelSegments: 3,
    curveSegments: 20,
  });
  cabinGeometry.translate(0, 0, -0.79);
  group.add(mesh(cabinGeometry, glass));
  const roof = roundedBox(1.02, 0.07, 1.57, 0.03, paint);
  roof.position.set(0.04, 1.73, 0);
  group.add(roof);
  for (const z of [-0.83, 0.83]) {
    const pillar = mesh(new THREE.BoxGeometry(0.08, 0.8, 0.055), paint, [
      0.05,
      1.26,
      z,
    ]);
    group.add(pillar);
    const sill = mesh(new THREE.BoxGeometry(2.4, 0.06, 0.08), chrome, [
      0.1,
      0.89,
      z,
    ]);
    group.add(sill);
    const mirror = roundedBox(0.25, 0.13, 0.18, 0.04, paint);
    mirror.position.set(-0.85, 1.02, z * 1.3);
    group.add(mirror);
  }
  for (const x of [-1.55, 1.55])
    for (const z of [-1, 1]) {
      const wheel = mesh(
        new THREE.CylinderGeometry(0.48, 0.48, 0.3, 40),
        rubber,
        [x, 0.32, z],
      );
      wheel.rotation.x = Math.PI / 2;
      group.add(wheel);
      const hub = mesh(
        new THREE.CylinderGeometry(0.27, 0.27, 0.315, 32),
        chrome,
        [x, 0.32, z],
      );
      hub.rotation.x = Math.PI / 2;
      group.add(hub);
      const cap = mesh(
        new THREE.CylinderGeometry(0.08, 0.08, 0.33, 16),
        rubber,
        [x, 0.32, z],
      );
      cap.rotation.x = Math.PI / 2;
      group.add(cap);
    }
  const headlights = new THREE.MeshStandardMaterial({
    color: 0xc9f6ff,
    emissive: 0x599da8,
    emissiveIntensity: 0.6,
    roughness: 0.2,
  });
  const tailLights = new THREE.MeshStandardMaterial({
    color: 0xef6253,
    emissive: 0x901a12,
    emissiveIntensity: 0.3,
  });
  for (const z of [-0.55, 0.55]) {
    group.add(
      mesh(new THREE.BoxGeometry(0.065, 0.12, 0.32), headlights, [
        -2.39,
        0.58,
        z,
      ]),
    );
    group.add(
      mesh(new THREE.BoxGeometry(0.065, 0.1, 0.36), tailLights, [
        2.33,
        0.57,
        z,
      ]),
    );
  }
  group.add(
    mesh(new THREE.BoxGeometry(0.07, 0.16, 0.72), rubber, [-2.46, 0.4, 0]),
  );
  return group;
}
function droneModel() {
  const group = new THREE.Group();
  const frame = new THREE.MeshStandardMaterial({
    color: 0x91a8b5,
    metalness: 0.42,
    roughness: 0.35,
    envMapIntensity: 0.6,
  });
  const dark = new THREE.MeshStandardMaterial({
    color: 0x213846,
    metalness: 0.3,
    roughness: 0.4,
  });
  const rotorMaterial = new THREE.MeshStandardMaterial({
    color: 0x55c3d5,
    metalness: 0.25,
    roughness: 0.35,
  });
  const core = roundedBox(1.15, 0.4, 0.88, 0.14, frame);
  group.add(core);
  const battery = roundedBox(0.66, 0.12, 0.58, 0.04, dark);
  battery.position.y = 0.26;
  group.add(battery);
  for (const x of [-1.85, 1.85])
    for (const z of [-1.85, 1.85]) {
      const length = Math.hypot(x, z),
        arm = mesh(new THREE.CylinderGeometry(0.065, 0.09, length, 12), dark);
      arm.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(x, 0, z).normalize(),
      );
      arm.position.set(x / 2, 0, z / 2);
      group.add(arm);
      const motor = mesh(
        new THREE.CylinderGeometry(0.18, 0.2, 0.3, 24),
        frame,
        [x, 0.08, z],
      );
      group.add(motor);
      for (let blade = 0; blade < 2; blade++) {
        const prop = roundedBox(1.65, 0.025, 0.15, 0.012, rotorMaterial);
        prop.position.set(x, 0.26, z);
        prop.rotation.y = (blade * Math.PI) / 2 + 0.3;
        group.add(prop);
      }
      const ring = mesh(new THREE.TorusGeometry(0.91, 0.035, 8, 64), dark, [
        x,
        0.2,
        z,
      ]);
      ring.rotation.x = Math.PI / 2;
      group.add(ring);
    }
  for (const z of [-0.55, 0.55]) {
    const skid = mesh(new THREE.BoxGeometry(0.1, 0.45, 0.08), dark, [
      0,
      -0.35,
      z,
    ]);
    group.add(skid);
    group.add(
      mesh(new THREE.BoxGeometry(0.95, 0.06, 0.08), frame, [0, -0.57, z]),
    );
  }
  return group;
}
export function createDemoModel(id: string, grid?: MaskGrid | null) {
  const group = new THREE.Group();
  const silver = new THREE.MeshStandardMaterial({
    color: 0x9daeb9,
    metalness: 0.55,
    roughness: 0.37,
    envMapIntensity: 0.6,
  });
  const bounds = grid ? maskBounds(grid) : null;
  const center = bounds?.getCenter(new THREE.Vector3()) ?? new THREE.Vector3();
  const size =
    bounds?.getSize(new THREE.Vector3()) ?? new THREE.Vector3(4.8, 2, 0);
  if (id.includes("airfoil")) {
    const step = grid ? cellSize(grid) : 4.6 / 36,
      chord = 36 * step,
      span = 2.6;
    const shape = new THREE.Shape();
    for (let i = 0; i <= 160; i++) {
      const x = (1 - Math.cos((i / 160) * Math.PI)) / 2,
        y = naca0012HalfThickness(x) * chord;
      if (i === 0) shape.moveTo(x * chord, y);
      else shape.lineTo(x * chord, y);
    }
    for (let i = 160; i >= 0; i--) {
      const x = (1 - Math.cos((i / 160) * Math.PI)) / 2;
      shape.lineTo(x * chord, -naca0012HalfThickness(x) * chord);
    }
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: span,
      bevelEnabled: false,
      curveSegments: 160,
      steps: 1,
    });
    geometry.translate(0, 0, -span / 2);
    geometry.computeVertexNormals();
    const section = new THREE.MeshStandardMaterial({
      color: 0x566e7a,
      metalness: 0.5,
      roughness: 0.5,
      envMapIntensity: 0.5,
    });
    const wing = new THREE.Mesh(geometry, [section, silver]);
    wing.rotation.z = (-8 * Math.PI) / 180;
    wing.position.set(
      grid ? 42 * step - 8 : -chord / 2,
      grid ? ((grid.ny - 1) / 2 - 32) * step : 0,
      0,
    );
    group.add(wing);
  } else if (id.includes("sphere")) {
    const radius = bounds ? Math.max(size.x, size.y) / 2 : 1.2;
    group.add(
      mesh(new THREE.SphereGeometry(radius, 64, 40), silver, [
        center.x,
        center.y,
        0,
      ]),
    );
  } else if (id.includes("cylinder")) {
    const radius = bounds ? Math.max(size.x, size.y) / 2 : 1.1;
    const cylinder = mesh(
      new THREE.CylinderGeometry(radius, radius, 2.6, 64),
      silver,
      [center.x, center.y, 0],
    );
    cylinder.rotation.x = Math.PI / 2;
    group.add(cylinder);
  } else {
    const model = id.includes("car") ? detailedCarModel() : droneModel();
    const natural = new THREE.Box3().setFromObject(model),
      naturalSize = natural.getSize(new THREE.Vector3());
    const preset = id.includes("car")
      ? sharedGeometry.cases.sports_car_v1
      : sharedGeometry.cases.drone_v1;
    if (
      grid &&
      grid.nx === sharedGeometry.nx &&
      grid.ny === sharedGeometry.ny
    ) {
      model.position.x += preset.transform.offset_x;
      model.position.y += preset.transform.offset_y;
      group.add(model);
      return group;
    }
    const scale = grid ? size.x / naturalSize.x : 1;
    model.scale.setScalar(scale);
    const fitted = new THREE.Box3().setFromObject(model),
      fittedCenter = fitted.getCenter(new THREE.Vector3());
    model.position.x = center.x - fittedCenter.x;
    model.position.y =
      id.includes("car") && bounds
        ? bounds.min.y - fitted.min.y
        : center.y - fittedCenter.y;
    group.add(model);
  }
  return group;
}
