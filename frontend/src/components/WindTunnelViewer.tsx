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
  windSpeed: number; // m/s (typically 5 to 30)
  windAngle: number; // degrees (-45 to 45)
  showStreamlines: boolean;
  showPressure: boolean;
  showWake: boolean;
  flowData: FlowData | null;
}

export const WindTunnelViewer: React.FC<WindTunnelViewerProps> = ({
  caseId,
  windSpeed,
  windAngle,
  showStreamlines,
  showPressure,
  showWake,
  flowData,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const requestRef = useRef<number | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  
  // Keep track of the active obstacle mesh so we can swap it
  const obstacleRef = useRef<THREE.Group | null>(null);
  
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
    camera.position.set(12, 6, 12);
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
        mat.opacity = 0.25;
      });
    } else {
      gridHelper.material.transparent = true;
      gridHelper.material.opacity = 0.25;
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

  // Update Obstacle Geometry when caseId changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (obstacleRef.current) {
      scene.remove(obstacleRef.current);
      obstacleRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((mat) => mat.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
    }

    // If we have flowData, we will build a visual indicator of the actual custom silhouette later.
    // For now, render standard templates or a general mesh representation.
    const group = new THREE.Group();
    let geom: THREE.BufferGeometry;
    const mat = new THREE.MeshStandardMaterial({
      color: 0x1a2333,
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
      // Fallback or custom upload silhouette rendering in 3D: render a generic extruded plate
      const boxGeom = new THREE.BoxGeometry(1.8, 1.2, 0.4);
      const mesh = new THREE.Mesh(boxGeom, mat);
      const wMesh = new THREE.Mesh(boxGeom, wireMat);
      group.add(mesh, wMesh);
      group.position.set(0, 0, 0);
    }

    scene.add(group);
    obstacleRef.current = group;
  }, [caseId]);

  // Main animation frame loop: calculates streamlines & deforms them around obstacle
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

      // 2. Animate streamlines (particles)
      const particles = particlesRef.current;
      const pointsGeom = pointsGeometryRef.current;
      const points = pointsRef.current;

      if (particles && pointsGeom && points) {
        if (showStreamlines) {
          points.visible = true;
          const positions = pointsGeom.attributes.position.array as Float32Array;
          const colors = pointsGeom.attributes.color.array as Float32Array;

          // Procedural radius limits
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

            // Check if we use LBM grid lookup or procedural potential-flow deflection
            if (flowData) {
              const nx = flowData.nx;
              const ny = flowData.ny;
              
              // Map particle coordinates:
              // pos.x is in [-8, 8] -> map to index [0, nx-1]
              // pos.y is in [-2, 2] -> map to index [0, ny-1]
              // Bottom of simulation in Python is y = ny-1, so we flip y coordinate
              const grid_x = Math.max(0, Math.min(nx - 1, Math.floor(((p.pos.x + 8) / 16) * (nx - 1))));
              const grid_y = Math.max(0, Math.min(ny - 1, (ny - 1) - Math.floor(((p.pos.y + 2) / 4) * (ny - 1))));
              
              const offset = grid_y * nx + grid_x;
              
              // Check boundary mask
              isObstacle = flowData.mask[offset] > 0;
              
              if (isObstacle) {
                // Instantly recycle particle to avoid clustering inside obstacle
                p.pos.x = -8.0;
                p.pos.y = p.baseY;
                p.pos.z = p.baseZ;
                p.age = 0;
                p.life = 100 + Math.random() * 50;
                continue;
              }
              
              // Retrieve simulated velocity (Lattice velocity: U_inlet = 0.08)
              // Map velocity components:
              // u_x = velocity[0, y, x]
              // u_y = velocity[1, y, x] (we flip sign for WebGL layout)
              const u_x = flowData.velocity[offset];
              const u_y = -flowData.velocity[ny * nx + offset];
              
              // Scale lattice velocities to match speed sliders
              const velocityScale = speed / 0.08;
              vx = u_x * velocityScale * 0.15; // apply visual scaling
              vy = u_y * velocityScale * 0.15;
              vz = 0;
              
              // Add slight turbulence in wake (behind shape x > 0)
              if (showWake && p.pos.x > 0.0 && u_x < 0.05) {
                const wakeFactor = Math.max(0, 1 - (u_x / 0.08));
                vy += (Math.random() - 0.5) * speed * 0.08 * wakeFactor;
                vz += (Math.random() - 0.5) * speed * 0.08 * wakeFactor;
              }
              
              // Retrieve pressure scalar
              pressureVal = flowData.pressure[offset];
            } else {
              // Procedural Potential-Flow approximation (Offline Fallback)
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

            // Move particle
            p.pos.x += vx * dt * 60;
            p.pos.y += vy * dt * 60;
            p.pos.z += vz * dt * 60;

            // Apply wind angle drift
            const angleRad = (windAngle * Math.PI) / 180;
            if (Math.abs(windAngle) > 0.1 && p.pos.x < -obstacleRadius) {
              p.pos.y += Math.sin(angleRad) * speed * dt * 30;
            }

            // Recycle exit particles
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

            // Default: Cyan
            let r = 0.0, g = 0.85, b = 1.0;

            if (showPressure) {
              if (flowData) {
                // Color scaling from LBM pressure (-0.03 to +0.03 range)
                const normP = (pressureVal + 0.02) / 0.04; // scale to [0, 1]
                const clampP = Math.max(0.0, Math.min(1.0, normP));
                
                if (clampP > 0.65) {
                  // High pressure (Stagnation) - Red/Orange
                  const intensity = (clampP - 0.65) / 0.35;
                  r = 0.5 + intensity * 0.5;
                  g = 0.7 * (1 - intensity);
                  b = 0;
                } else if (clampP < 0.35) {
                  // Low pressure (Separation/Lift acceleration) - Blue
                  const intensity = (0.35 - clampP) / 0.35;
                  r = 0;
                  g = 0.8 * (1 - intensity);
                  b = 0.6 + intensity * 0.4;
                } else {
                  // Neutral - Cyan/Green
                  r = 0;
                  g = 0.9;
                  b = 0.3;
                }
              } else {
                // Procedural pressure colors (Offline Fallback)
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
  }, [caseId, windSpeed, windAngle, showStreamlines, showPressure, showWake, flowData]);

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
