// Run from frontend with: node --experimental-strip-types scripts/export-demo-masks.mjs
// Project the SAME display geometry into a 2D solver mask; this does not run CFD.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  createDemoModel,
  installDetailedCar,
  cellSize,
} from "../src/components/demoModels.ts";
const root = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const asset = await fs.readFile(
  path.join(root, "frontend/public/models/concept-car.glb"),
);
// The CPU-side exporter reads geometry only; texture decoding belongs to the browser.
const sourceJSONLength = asset.readUInt32LE(12),
  binStart = 20 + sourceJSONLength;
const sourceJSON = JSON.parse(
  asset.subarray(20, 20 + sourceJSONLength).toString("utf8"),
);
function withoutTextures(value) {
  if (Array.isArray(value)) return value.map(withoutTextures);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .filter(
          ([key]) =>
            !key.toLowerCase().endsWith("texture") &&
            !["images", "textures", "samplers"].includes(key),
        )
        .map(([key, item]) => [key, withoutTextures(item)]),
    );
  return value;
}
const geometryJSON = Buffer.from(JSON.stringify(withoutTextures(sourceJSON)));
const paddedJSON = Buffer.concat([
  geometryJSON,
  Buffer.alloc((4 - (geometryJSON.length % 4)) % 4, 32),
]);
const binLength = asset.readUInt32LE(binStart),
  binary = asset.subarray(binStart + 8, binStart + 8 + binLength);
const header = Buffer.alloc(20);
header.writeUInt32LE(0x46546c67, 0);
header.writeUInt32LE(2, 4);
header.writeUInt32LE(28 + paddedJSON.length + binary.length, 8);
header.writeUInt32LE(paddedJSON.length, 12);
header.writeUInt32LE(0x4e4f534a, 16);
const binHeader = Buffer.alloc(8);
binHeader.writeUInt32LE(binary.length, 0);
binHeader.writeUInt32LE(0x004e4942, 4);
const geometryAsset = Buffer.concat([header, paddedJSON, binHeader, binary]);
const gltf = await new GLTFLoader().parseAsync(
  geometryAsset.buffer.slice(
    geometryAsset.byteOffset,
    geometryAsset.byteOffset + geometryAsset.byteLength,
  ),
  "",
);

installDetailedCar(gltf.scene);
function surfaceBounds(model) {
  model.updateMatrixWorld(true);
  const bounds = new THREE.Box3(),
    point = new THREE.Vector3();
  model.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      const index = object.geometry.index,
        count = index?.count ?? object.geometry.attributes.position.count;
      for (let i = 0; i < count; i++)
        bounds.expandByPoint(
          object
            .getVertexPosition(index ? index.getX(i) : i, point)
            .applyMatrix4(object.matrixWorld),
        );
    }
  });
  return bounds;
}
const nx = 128,
  ny = 64,
  step = cellSize({ nx }),
  supersample = 4;
const cases = {};
for (const [id, caseId] of [
  ["sports_car", "sports_car_v1"],
  ["drone", "drone_v1"],
]) {
  const model = createDemoModel(id);
  model.updateMatrixWorld(true);
  const initial = surfaceBounds(model);
  const offsetX = id === "sports_car" ? -0.5 : 0;
  const offsetY = id === "sports_car" ? -3.55 - initial.min.y : 0;
  model.position.x += offsetX;
  model.position.y += offsetY;
  model.updateMatrixWorld(true);
  const raster = new Uint8Array(nx * ny * supersample * supersample),
    width = nx * supersample,
    height = ny * supersample;
  let triangles = 0;
  model.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    const positions = object.geometry.attributes.position,
      index = object.geometry.index;
    const count = index?.count ?? positions.count;
    const points = [
      new THREE.Vector3(),
      new THREE.Vector3(),
      new THREE.Vector3(),
    ];
    for (let i = 0; i < count; i += 3) {
      triangles++;
      for (let j = 0; j < 3; j++) {
        points[j]
          .fromBufferAttribute(positions, index ? index.getX(i + j) : i + j)
          .applyMatrix4(object.matrixWorld);
        points[j].set(
          ((points[j].x + 8) / step) * supersample + supersample / 2,
          ((ny - 1) / 2 - points[j].y / step) * supersample + supersample / 2,
          0,
        );
      }
      const [a, b, c] = points;
      const area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (Math.abs(area) < 1e-7) continue;
      const left = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x))),
        right = Math.min(width - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
      const top = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y))),
        bottom = Math.min(height - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
      for (let y = top; y <= bottom; y++)
        for (let x = left; x <= right; x++) {
          const px = x + 0.5,
            py = y + 0.5;
          const u = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) / area;
          const v = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) / area;
          if (u >= 0 && v >= 0 && u + v <= 1) raster[y * width + x] = 1;
        }
    }
  });
  const mask = new Array(nx * ny).fill(0);
  for (let y = 0; y < ny; y++)
    for (let x = 0; x < nx; x++) {
      // Conservative occupancy keeps thin arms/rotors instead of losing them.
      for (let sy = 0; sy < supersample; sy++)
        for (let sx = 0; sx < supersample; sx++)
          if (raster[(y * supersample + sy) * width + x * supersample + sx])
            mask[y * nx + x] = 1;
    }
  const bound = surfaceBounds(model);
  const solid = mask.reduce((a, b) => a + b, 0);
  if (
    !solid ||
    mask.slice(0, nx).some(Boolean) ||
    mask.slice(-nx).some(Boolean)
  )
    throw new Error("Invalid projected mask");
  cases[caseId] = {
    display_id: id,
    transform: { offset_x: offsetX, offset_y: offsetY, scale: 1 },
    bounds: bound.min.toArray().concat(bound.max.toArray()),
    mask,
    solid_cells: solid,
    projected_triangles: triangles,
  };
  console.log(caseId, solid, "solid cells;", triangles, "triangles projected");
}
const output = {
  version: 1,
  nx,
  ny,
  cell_size: step,
  source:
    "Projected side silhouettes of the displayed Three.js models; conservative 4x raster sampling",
  car_asset_sha256: crypto.createHash("sha256").update(asset).digest("hex"),
  cases,
};
if (process.argv.includes("--check")) {
  const existing = JSON.parse(
    await fs.readFile(path.join(root, "shared/demoGeometry.json"), "utf8"),
  );
  if (JSON.stringify(existing) !== JSON.stringify(output))
    throw new Error(
      "Display geometry changed: regenerate masks and CPU caches",
    );
  console.log("Display geometry matches the published solver projections");
} else {
  await fs.writeFile(
    path.join(root, "shared/demoGeometry.json"),
    JSON.stringify(output),
  );
  console.log("Wrote shared/demoGeometry.json");
}
