import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface FlowData {
  velocity: Float32Array;
  pressure: Float32Array;
  mask: Uint8Array;
  nx: number;
  ny: number;
}

interface WindTunnelViewerProps {
  caseId: string;
  windSpeed: number;
  windAngle: number;
  showStreamlines: boolean;
  showPressure: boolean;
  showWake: boolean;
  flowData: FlowData | null;
  // Phase 5 features
  showVoxels: boolean;
  showSlicePlane: boolean;
  slicePosition: number; // -2.0 to 2.0 (representing Y floor-to-ceiling)
}

export const WindTunnelViewer: React.FC<WindTunnelViewerProps> = ({
  caseId,
  windSpeed,
  windAngle,
  showStreamlines,
  showPressure,
  showWake,
  flowData,
  showVoxels,
  showSlicePlane,
  slicePosition,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const requestRef = useRef<number | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  
  // Active obstacle mesh
  const obstacleRef = useRef<THREE.Group | null>(null);
  
  // Slice plane & Arrows
  const slicePlaneMeshRef = useRef<THREE.Mesh | null>(null);
  const arrowsGroupRef = useRef<THREE.Group | null>(null);
  
  // Particle system state
  const particleCount = 1200;
  const particlesRef = useRef<{
    pos: THREE.Vector3;
    age: number;
    life: number;
    baseY: number;
    baseZ: number;
  }[]>([]);
  const pointsGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);

  // Initialize scene, camera, lights, orbit controls
  useEffect(() => {
    if (!containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020205);
    scene.fog = new THREE.FogExp2(0x020205, 0.05);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(11, 5, 11);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.minDistance = 3;
    controls.maxDistance = 30;

    // 5. Lights
    const ambientLight = new THREE.AmbientLight(0x111122);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x00f2fe, 1.5);
    dirLight1.position.set(5, 10, 7);
    dirLight1.castShadow = true;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x7c3aed, 1.0);
    dirLight2.position.set(-5, 5, -7);
    scene.add(dirLight2);

    // 6. Tunnel Cage / Floor grid
    const gridHelper = new THREE.GridHelper(24, 24, 0x00f2fe, 0x1e293b);
    gridHelper.position.y = -2;
    if (Array.isArray(gridHelper.material)) {
      gridHelper.material.forEach((mat) => {
        mat.transparent = true;
        mat.opacity = 0.2;
      });
    } else {
      gridHelper.material.transparent = true;
      gridHelper.material.opacity = 0.2;
    }
    scene.add(gridHelper);

    // Bounding Box representing the wind tunnel boundaries
    const boxGeom = new THREE.BoxGeometry(16, 4, 6);
    const edges = new THREE.EdgesGeometry(boxGeom);
    const boxMat = new THREE.LineBasicMaterial({ 
      color: 0x334155, 
      transparent: true, 
      opacity: 0.3 
    });
    const tunnelWireframe = new THREE.LineSegments(edges, boxMat);
    tunnelWireframe.position.set(0, 0, 0);
    scene.add(tunnelWireframe);

    // Create Slice Plane mesh (Phase 5)
    const planeGeom = new THREE.PlaneGeometry(16, 6);
    planeGeom.rotateX(-Math.PI / 2);
    const planeMat = new THREE.MeshBasicMaterial({
      color: 0x00f2fe,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const slicePlaneMesh = new THREE.Mesh(planeGeom, planeMat);
    scene.add(slicePlaneMesh);
    slicePlaneMeshRef.current = slicePlaneMesh;

    // Create Slice Arrows Group (Phase 5)
    const arrowsGroup = new THREE.Group();
    scene.add(arrowsGroup);
    arrowsGroupRef.current = arrowsGroup;

    // Generate Arrow grid on the plane
    // 13 columns along X (from -6 to +6), 5 rows along Z (from -2 to +2)
    const arrowDir = new THREE.Vector3(1, 0, 0);
    for (let xVal = -6; xVal <= 6; xVal += 1.0) {
      for (let zVal = -2; zVal <= 2; zVal += 1.0) {
        const arrow = new THREE.ArrowHelper(
          arrowDir, 
          new THREE.Vector3(xVal, 0, zVal), 
          0.6, 
          0x00f2fe, 
          0.15, 
          0.1
        );
        arrowsGroup.add(arrow);
      }
    }

    // 7. Initialize particle data
    const particles = [];
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        pos: new THREE.Vector3(
          (Math.random() - 0.5) * 16,
          (Math.random() - 0.5) * 3.8,
          (Math.random() - 0.5) * 5.8
        ),
        age: Math.random() * 100,
        life: 100 + Math.random() * 50,
        baseY: 0,
        baseZ: 0
      });
      particles[i].baseY = particles[i].pos.y;
      particles[i].baseZ = particles[i].pos.z;
    }
    particlesRef.current = particles;

    // Create Particle geometry
    const pointsGeom = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    pointsGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointsGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    pointsGeometryRef.current = pointsGeom;

    const particleMat = new THREE.PointsMaterial({
      size: 0.08,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const points = new THREE.Points(pointsGeom, particleMat);
    scene.add(points);
    pointsRef.current = points;

    // Handle resizing
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Clean up
    return () => {
      window.removeEventListener('resize', handleResize);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      controls.dispose();
      renderer.dispose();
      if (containerRef.current && renderer.domElement.parentNode) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update Obstacle Geometry (Smooth vs Voxelized) when caseId or showVoxels changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (obstacleRef.current) {
      scene.remove(obstacleRef.current);
      obstacleRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
            else child.material.dispose();
          }
        }
      });
    }

    const group = new THREE.Group();
    let geom: THREE.BufferGeometry;
    const mat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.1,
      metalness: 0.8,
      flatShading: true,
    });

    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00f2fe,
      wireframe: true,
      transparent: true,
      opacity: 0.25
    });

    // Helper to generate voxel cubes procedurally
    const addVoxels = (carGroup: THREE.Group, coordinates: [number, number, number][]) => {
      const voxelGeom = new THREE.BoxGeometry(0.25, 0.25, 0.25);
      const voxelMat = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.3, metalness: 0.7 });
      coordinates.forEach(([x, y, z]) => {
        const mesh = new THREE.Mesh(voxelGeom, voxelMat);
        mesh.position.set(x, y, z);
        carGroup.add(mesh);
      });
    };

    if (showVoxels) {
      // Phase 5 Voxelized representations
      const voxelGroup = new THREE.Group();
      const coords: [number, number, number][] = [];

      if (caseId.includes('car')) {
        // Car profile: low chassis, cabin in middle
        // Bounding box: 2.5 length, 0.6 height, 1.2 width
        for (let x = -1.2; x <= 1.2; x += 0.28) {
          for (let z = -0.5; z <= 0.5; z += 0.28) {
            coords.push([x, -0.4, z]); // Floor board
            coords.push([x, -0.15, z]); // Lower body
            // Cabin in center
            if (x >= -0.6 && x <= 0.4) {
              coords.push([x, 0.1, z]);
              coords.push([x, 0.35, z]);
            }
          }
        }
        addVoxels(voxelGroup, coords);
      } else if (caseId.includes('drone')) {
        // Center cylinder + arms
        for (let r = 0; r <= 0.4; r += 0.2) {
          for (let th = 0; th < Math.PI * 2; th += Math.PI / 4) {
            coords.push([r * Math.cos(th), 0, r * Math.sin(th)]);
            coords.push([r * Math.cos(th), 0.2, r * Math.sin(th)]);
          }
        }
        // Arm vectors (x = y)
        for (let d = 0.4; d <= 1.2; d += 0.25) {
          coords.push([d, 0.1, d]);
          coords.push([-d, 0.1, d]);
          coords.push([d, 0.1, -d]);
          coords.push([-d, 0.1, -d]);
        }
        addVoxels(voxelGroup, coords);
      } else if (caseId.includes('airfoil')) {
        // Tapered wing section
        for (let x = -1.0; x <= 1.0; x += 0.25) {
          const thickness = 0.4 * (1.0 - (x + 1.0)/2.0); // Taper leading-to-trailing
          const maxZ = 1.2;
          for (let z = -maxZ; z <= maxZ; z += 0.3) {
            coords.push([x, 0, z]);
            if (thickness > 0.1) {
              coords.push([x, thickness / 2, z]);
              coords.push([x, -thickness / 2, z]);
            }
          }
        }
        addVoxels(voxelGroup, coords);
      } else {
        // Custom uploads - simple voxel grid block
        for (let x = -0.8; x <= 0.8; x += 0.3) {
          for (let y = -0.5; y <= 0.5; y += 0.3) {
            for (let z = -0.5; z <= 0.5; z += 0.3) {
              coords.push([x, y, z]);
            }
          }
        }
        addVoxels(voxelGroup, coords);
      }
      group.add(voxelGroup);
      group.position.set(0, 0, 0);
    } else {
      // Smooth visual styling (Standard templates)
      if (caseId === 'sports_car' || caseId === 'sports_car_v1') {
        const carGroup = new THREE.Group();
        const bodyGeom = new THREE.BoxGeometry(2.5, 0.6, 1.2);
        const bodyMesh = new THREE.Mesh(bodyGeom, mat);
        bodyMesh.castShadow = true;
        bodyMesh.receiveShadow = true;
        carGroup.add(bodyMesh);

        const cabinGeom = new THREE.BoxGeometry(1.2, 0.5, 1.0);
        cabinGeom.translate(-0.2, 0.45, 0);
        const cabinMesh = new THREE.Mesh(cabinGeom, mat);
        cabinMesh.castShadow = true;
        carGroup.add(cabinMesh);

        const wheelGeom = new THREE.CylinderGeometry(0.35, 0.35, 0.2, 16);
        wheelGeom.rotateX(Math.PI / 2);
        const wheelMat = new THREE.MeshStandardMaterial({ color: 0x0c0d12, roughness: 0.8 });
        
        const wheels = [
          [-0.7, -0.25, 0.6],
          [-0.7, -0.25, -0.6],
          [0.7, -0.25, 0.6],
          [0.7, -0.25, -0.6]
        ];
        wheels.forEach(([x, y, z]) => {
          const mesh = new THREE.Mesh(wheelGeom, wheelMat);
          mesh.position.set(x, y, z);
          carGroup.add(mesh);
        });

        const wMesh = new THREE.Mesh(bodyGeom, wireMat);
        carGroup.add(wMesh);
        
        group.add(carGroup);
        group.position.set(0, -0.5, 0);
      } else if (caseId === 'drone' || caseId === 'drone_v1') {
        const droneGroup = new THREE.Group();
        const coreGeom = new THREE.CylinderGeometry(0.4, 0.4, 0.25, 8);
        const coreMesh = new THREE.Mesh(coreGeom, mat);
        droneGroup.add(coreMesh);

        const armGeom = new THREE.BoxGeometry(2.4, 0.06, 0.12);
        const arm1 = new THREE.Mesh(armGeom, mat);
        arm1.rotateY(Math.PI / 4);
        const arm2 = new THREE.Mesh(armGeom, mat);
        arm2.rotateY(-Math.PI / 4);
        droneGroup.add(arm1, arm2);

        const motorGeom = new THREE.CylinderGeometry(0.12, 0.12, 0.2, 8);
        const propGeom = new THREE.RingGeometry(0.45, 0.5, 32);
        propGeom.rotateX(Math.PI / 2);
        const propMat = new THREE.MeshBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: 0.3, side: THREE.DoubleSide });

        const motorPositions = [
          [0.85, 0.1, 0.85],
          [0.85, 0.1, -0.85],
          [-0.85, 0.1, 0.85],
          [-0.85, 0.1, -0.85]
        ];
        motorPositions.forEach(([x, y, z]) => {
          const motor = new THREE.Mesh(motorGeom, mat);
          motor.position.set(x, y, z);
          
          const prop = new THREE.Mesh(propGeom, propMat);
          prop.position.set(x, y + 0.1, z);
          
          droneGroup.add(motor, prop);
        });

        const wMesh = new THREE.Mesh(coreGeom, wireMat);
        droneGroup.add(wMesh);

        group.add(droneGroup);
        group.position.set(0, 0, 0);
      } else if (caseId === 'airfoil' || caseId === 'airfoil_v1') {
        const wingGroup = new THREE.Group();
        const shape = new THREE.Shape();
        shape.moveTo(-1.0, 0.0);
        shape.bezierCurveTo(-0.6, 0.2, 0.4, 0.35, 1.0, 0.0);
        shape.bezierCurveTo(0.4, -0.35, -0.6, -0.2, -1.0, 0.0);

        const extrudeSettings = {
          depth: 3.0,
          bevelEnabled: false
        };
        
        geom = new THREE.ExtrudeGeometry(shape, extrudeSettings);
        geom.center();
        geom.rotateX(Math.PI / 2);
        
        const wingMesh = new THREE.Mesh(geom, mat);
        wingMesh.castShadow = true;
        wingMesh.receiveShadow = true;
        wingGroup.add(wingMesh);

        const wMesh = new THREE.Mesh(geom, wireMat);
        wingGroup.add(wMesh);

        group.add(wingGroup);
        group.position.set(0, 0, 0);
      } else {
        const boxGeom = new THREE.BoxGeometry(1.8, 1.2, 0.4);
        const mesh = new THREE.Mesh(boxGeom, mat);
        const wMesh = new THREE.Mesh(boxGeom, wireMat);
        group.add(mesh, wMesh);
        group.position.set(0, 0, 0);
      }
    }

    scene.add(group);
    obstacleRef.current = group;
  }, [caseId, showVoxels]);

  // Main animation frame loop: calculates streamlines, updates slice plane, and updates slice vector arrows
  useEffect(() => {
    let lastTime = performance.now();

    const animate = () => {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      const camera = cameraRef.current;
      const scene = sceneRef.current;

      // 1. Rotate obstacle matching windAngle
      if (obstacleRef.current) {
        const targetAngleRad = (windAngle * Math.PI) / 180;
        obstacleRef.current.rotation.y = THREE.MathUtils.lerp(
          obstacleRef.current.rotation.y, 
          targetAngleRad, 
          0.1
        );
      }

      // 2. Position Slice Plane (Phase 5)
      const slicePlaneMesh = slicePlaneMeshRef.current;
      const arrowsGroup = arrowsGroupRef.current;
      
      if (slicePlaneMesh && arrowsGroup) {
        if (showSlicePlane) {
          slicePlaneMesh.visible = true;
          slicePlaneMesh.position.y = slicePosition;
          
          arrowsGroup.visible = true;
          arrowsGroup.position.y = slicePosition;
          
          // Update each arrow helper direction and length
          const velocityScale = windSpeed / 0.08;
          const speed = windSpeed * 0.15;
          const objectPos = new THREE.Vector3(0, obstacleRef.current ? obstacleRef.current.position.y : 0, 0);
          
          let obstacleRadius = 1.2;
          if (caseId.includes('car')) obstacleRadius = 1.4;
          else if (caseId.includes('drone')) obstacleRadius = 1.5;
          else if (caseId.includes('airfoil')) obstacleRadius = 1.1;

          arrowsGroup.children.forEach((child) => {
            if (child instanceof THREE.ArrowHelper) {
              const arrowPos = child.position.clone();
              // Compute local 3D position relative to world
              const worldArrowPos = new THREE.Vector3(arrowPos.x, slicePosition, arrowPos.z);
              
              let vx = speed;
              let vy = 0;
              let pVal = 0.0;
              let isObst = false;
              
              if (flowData) {
                const nx = flowData.nx;
                const ny = flowData.ny;
                
                const grid_x = Math.max(0, Math.min(nx - 1, Math.floor(((worldArrowPos.x + 8) / 16) * (nx - 1))));
                const grid_y = Math.max(0, Math.min(ny - 1, (ny - 1) - Math.floor(((slicePosition + 2) / 4) * (ny - 1))));
                
                const offset = grid_y * nx + grid_x;
                isObst = flowData.mask[offset] > 0;
                
                if (isObst) {
                  vx = 0.001;
                  vy = 0;
                } else {
                  vx = flowData.velocity[offset] * velocityScale * 0.15;
                  vy = -flowData.velocity[ny * nx + offset] * velocityScale * 0.15;
                  pVal = flowData.pressure[offset];
                }
              } else {
                // Procedural Potential Flow (Fallback)
                const distVec = worldArrowPos.clone().sub(objectPos);
                const dist2D = Math.sqrt(distVec.x * distVec.x + distVec.y * distVec.y);
                
                if (dist2D < obstacleRadius * 3.5 && worldArrowPos.x < obstacleRadius * 3.0) {
                  const factor = Math.pow(obstacleRadius / Math.max(dist2D, 0.2), 2);
                  if (dist2D > 0.05) {
                    const normal = distVec.clone().normalize();
                    const radialComponent = speed * normal.x * factor;
                    vx = speed - radialComponent * normal.x;
                    vy = -radialComponent * normal.y;
                  }
                }
              }
              
              // Set arrow helper direction
              const dirVec = new THREE.Vector3(vx, vy, 0);
              const mag = dirVec.length();
              if (mag > 0.001) {
                child.setDirection(dirVec.normalize());
              }
              
              // Length scales with flow speed
              const finalLength = Math.max(0.1, Math.min(1.2, mag * 0.8));
              child.setLength(finalLength, 0.15, 0.08);
              
              // Set color based on pressure
              let color = 0x00f2fe; // neutral cyan
              if (showPressure) {
                if (flowData) {
                  const normP = (pVal + 0.02) / 0.04;
                  if (normP > 0.65) color = 0xff4757; // red
                  else if (normP < 0.35) color = 0x4facfe; // blue
                  else color = 0x10b981; // green
                } else {
                  const dist = worldArrowPos.distanceTo(objectPos);
                  if (dist < obstacleRadius * 2.5) {
                    if (worldArrowPos.x < -0.15) color = 0xff4757;
                    else color = 0x4facfe;
                  }
                }
              }
              child.setColor(color);
            }
          });
        } else {
          slicePlaneMesh.visible = false;
          arrowsGroup.visible = false;
        }
      }

      // 3. Animate streamlines (particles)
      const particles = particlesRef.current;
      const pointsGeom = pointsGeometryRef.current;
      const points = pointsRef.current;

      if (particles && pointsGeom && points) {
        if (showStreamlines) {
          points.visible = true;
          const positions = pointsGeom.attributes.position.array as Float32Array;
          const colors = pointsGeom.attributes.color.array as Float32Array;

          let obstacleRadius = 1.2;
          if (caseId.includes('car')) obstacleRadius = 1.4;
          else if (caseId.includes('drone')) obstacleRadius = 1.5;
          else if (caseId.includes('airfoil')) obstacleRadius = 1.1;

          const speed = windSpeed * 0.15;
          const objectPos = new THREE.Vector3(0, obstacleRef.current ? obstacleRef.current.position.y : 0, 0);

          for (let i = 0; i < particleCount; i++) {
            const p = particles[i];
            
            let vx = speed;
            let vy = 0;
            let vz = 0;
            let pressureVal = 0.0;
            let isObstacle = false;

            if (flowData) {
              const nx = flowData.nx;
              const ny = flowData.ny;
              
              const grid_x = Math.max(0, Math.min(nx - 1, Math.floor(((p.pos.x + 8) / 16) * (nx - 1))));
              const grid_y = Math.max(0, Math.min(ny - 1, (ny - 1) - Math.floor(((p.pos.y + 2) / 4) * (ny - 1))));
              
              const offset = grid_y * nx + grid_x;
              
              isObstacle = flowData.mask[offset] > 0;
              
              if (isObstacle) {
                p.pos.x = -8.0;
                p.pos.y = p.baseY;
                p.pos.z = p.baseZ;
                p.age = 0;
                p.life = 100 + Math.random() * 50;
                continue;
              }
              
              const u_x = flowData.velocity[offset];
              const u_y = -flowData.velocity[ny * nx + offset];
              
              const velocityScale = speed / 0.08;
              vx = u_x * velocityScale * 0.15;
              vy = u_y * velocityScale * 0.15;
              vz = 0;
              
              if (showWake && p.pos.x > 0.0 && u_x < 0.05) {
                const wakeFactor = Math.max(0, 1 - (u_x / 0.08));
                vy += (Math.random() - 0.5) * speed * 0.08 * wakeFactor;
                vz += (Math.random() - 0.5) * speed * 0.08 * wakeFactor;
              }
              
              pressureVal = flowData.pressure[offset];
            } else {
              const distVec = p.pos.clone().sub(objectPos);
              const dist2D = Math.sqrt(distVec.x * distVec.x + distVec.y * distVec.y);

              if (dist2D < obstacleRadius * 3.5 && p.pos.x < obstacleRadius * 3.0) {
                const factor = Math.pow(obstacleRadius / Math.max(dist2D, 0.2), 2);
                if (dist2D > 0.05) {
                  const normal = distVec.clone().normalize();
                  const radialComponent = speed * normal.x * factor;
                  vx = speed - radialComponent * normal.x;
                  vy = -radialComponent * normal.y;
                  
                  if (p.pos.x > 0.2) {
                    vx *= (0.2 + 0.3 * (dist2D / (obstacleRadius * 3)));
                    if (showWake) {
                      vy += (Math.random() - 0.5) * speed * 0.4 * factor;
                      vz += (Math.random() - 0.5) * speed * 0.4 * factor;
                    }
                  }
                }
              }
            }

            p.pos.x += vx * dt * 60;
            p.pos.y += vy * dt * 60;
            p.pos.z += vz * dt * 60;

            const angleRad = (windAngle * Math.PI) / 180;
            if (Math.abs(windAngle) > 0.1 && p.pos.x < -obstacleRadius) {
              p.pos.y += Math.sin(angleRad) * speed * dt * 30;
            }

            if (p.pos.x > 8.0 || p.age > p.life) {
              p.pos.x = -8.0;
              p.pos.y = p.baseY;
              p.pos.z = p.baseZ;
              p.age = 0;
              p.life = 100 + Math.random() * 50;
            } else {
              p.age += 1;
            }

            const idx = i * 3;
            positions[idx] = p.pos.x;
            positions[idx+1] = p.pos.y;
            positions[idx+2] = p.pos.z;

            let r = 0.0, g = 0.85, b = 1.0;

            if (showPressure) {
              if (flowData) {
                const normP = (pressureVal + 0.02) / 0.04;
                const clampP = Math.max(0.0, Math.min(1.0, normP));
                
                if (clampP > 0.65) {
                  const intensity = (clampP - 0.65) / 0.35;
                  r = 0.5 + intensity * 0.5;
                  g = 0.7 * (1 - intensity);
                  b = 0;
                } else if (clampP < 0.35) {
                  const intensity = (0.35 - clampP) / 0.35;
                  r = 0;
                  g = 0.8 * (1 - intensity);
                  b = 0.6 + intensity * 0.4;
                } else {
                  r = 0;
                  g = 0.9;
                  b = 0.3;
                }
              } else {
                const dist = p.pos.distanceTo(objectPos);
                if (dist < obstacleRadius * 2.8) {
                  if (p.pos.x < -0.15) {
                    const intensity = Math.max(0, 1 - (dist / (obstacleRadius * 1.8)));
                    r = intensity;
                    g = 1 - intensity * 0.8;
                    b = 1 - intensity;
                  } else if (p.pos.x >= -0.15 && p.pos.x <= 0.5) {
                    const intensity = Math.max(0, 1 - (dist / (obstacleRadius * 2.0)));
                    r = 0;
                    g = 0.7 * (1 - intensity);
                    b = 1.0;
                  } else {
                    const intensity = Math.max(0, 1 - (dist / (obstacleRadius * 3.0)));
                    r = intensity * 0.6;
                    g = 0.8;
                    b = 0.2;
                  }
                }
              }
            }

            colors[idx] = r;
            colors[idx+1] = g;
            colors[idx+2] = b;
          }

          pointsGeom.attributes.position.needsUpdate = true;
          pointsGeom.attributes.color.needsUpdate = true;
        } else {
          points.visible = false;
        }
      }

      if (rendererRef.current && scene && camera) {
        rendererRef.current.render(scene, camera);
      }

      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [caseId, windSpeed, windAngle, showStreamlines, showPressure, showWake, flowData, showSlicePlane, slicePosition]);

  return (
    <div className="tunnel-view-container">
      <div ref={containerRef} className="canvas-container" />
      
      {/* Visual legends overlay */}
      {showPressure && (
        <div className="pressure-legend">
          <div className="legend-title">Pressure Distribution</div>
          <div className="legend-bar"></div>
          <div className="legend-labels">
            <span>Low (-p)</span>
            <span>Neutral</span>
            <span>High (+p)</span>
          </div>
        </div>
      )}
    </div>
  );
};
