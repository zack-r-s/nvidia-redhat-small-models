import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Device, MachineState } from "./types";
import { bpmPv } from "./types";

const colors = { bpm: "#38bdf8", corrector: "#f59e0b", quadrupole: "#a78bfa", dipole: "#64748b" };
const RADIUS = 3.2;
/** Tube radius in scene units. The tube cross-section IS the deviation display window, not the physical pipe. */
const TUBE_R = 0.16;
/** Selectable display windows: a measured deviation of ±window mm maps to the tube wall. */
export const WINDOWS_MM = [0.5, 1, 2, 5];
/** These are display laps per second, not the beam's physical revolution rate. */
export const PLAYBACK_SPEEDS = [
  { label: "¼×", lapsPerSecond: 0.25, tail: 0.08 },
  { label: "½×", lapsPerSecond: 0.5, tail: 0.14 },
  { label: "1×", lapsPerSecond: 1, tail: 0.24 },
  { label: "2×", lapsPerSecond: 2, tail: 0.42 },
  { label: "Max", lapsPerSecond: 8, tail: 1 },
] as const;
/** Equivalent magnification of the window mapping (scene length / true length, both relative to the ring). */
export const magnificationFor = (windowMm: number, circumference_m: number) => TUBE_R / (windowMm * 1e-3 * (RADIUS / (circumference_m / (2 * Math.PI))));

type OrbitSample = NonNullable<MachineState["orbit_profile"]>["samples"][number];
const TAIL_POINTS = 96;

/** A linear drawing interpolation between dense PyAT samples; it is never used as machine data. */
function orbitAt(samples: OrbitSample[], s: number) {
  let lo = 0;
  let hi = samples.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].s_m <= s) lo = mid;
    else hi = mid;
  }
  const a = samples[lo];
  const b = samples[hi];
  const t = b.s_m > a.s_m ? (s - a.s_m) / (b.s_m - a.s_m) : 0;
  return { x: a.x_mm + (b.x_mm - a.x_mm) * t, y: a.y_mm + (b.y_mm - a.y_mm) * t };
}

function scenePos(s_m: number, x: number, y: number, circumference: number, windowMm: number) {
  const a = (s_m / circumference) * 2 * Math.PI;
  const norm = Math.hypot(x, y);
  const k = norm > windowMm ? windowMm / norm : 1;
  const r = RADIUS + ((x * k) / windowMm) * TUBE_R;
  return { pos: new THREE.Vector3(r * Math.cos(a), ((y * k) / windowMm) * TUBE_R, r * Math.sin(a)), offScale: norm > windowMm };
}

function OrbitPlayback({ state, windowMm, speedStep, active }: { state: MachineState; windowMm: number; speedStep: number; active: boolean }) {
  const profile = state.orbit_profile;
  const circumference = state.lattice.circumference_m || 0;
  const pathKey = JSON.stringify(profile?.samples);
  const samples = useMemo(
    () => profile?.kind === "pyat-closed-orbit-4d"
      ? profile.samples.filter((p) => Number.isFinite(p.s_m) && Number.isFinite(p.x_mm) && Number.isFinite(p.y_mm))
      : [],
    // Polls return new objects even when the physical orbit has not changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pathKey],
  );
  const valid = circumference > 0 && samples.length > 2 && samples[0].s_m === 0 && samples[samples.length - 1].s_m >= circumference - 1e-8;
  const fullGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    if (!valid) return geometry;
    // Each vertex is a PyAT result; Three connects the dense samples only for display.
    const positions = samples.slice(0, -1).flatMap((p) => scenePos(p.s_m, p.x_mm, p.y_mm, circumference, windowMm).pos.toArray());
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geometry;
  }, [samples, circumference, windowMm, valid]);
  const tailGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TAIL_POINTS * 3), 3));
    const colors = new Float32Array(TAIL_POINTS * 3);
    for (let i = 0; i < TAIL_POINTS; i++) {
      const intensity = 0.06 + 0.94 * (i / (TAIL_POINTS - 1)) ** 2;
      colors[3 * i] = 0.12 * intensity;
      colors[3 * i + 1] = intensity;
      colors[3 * i + 2] = 0.38 * intensity;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geometry;
  }, []);
  const tailLine = useMemo(() => {
    const line = new THREE.Line(tailGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.92, depthTest: false }));
    line.frustumCulled = false;
    line.renderOrder = 10;
    line.raycast = () => {};
    return line;
  }, [tailGeometry]);
  const marker = useRef<THREE.Mesh>(null);
  const phase = useRef(0);
  const previousPath = useRef("");
  useEffect(() => () => { fullGeometry.dispose(); }, [fullGeometry]);
  useEffect(() => () => { tailGeometry.dispose(); }, [tailGeometry]);
  useEffect(() => () => { (tailLine.material as THREE.Material).dispose(); }, [tailLine]);
  useFrame((_, delta) => {
    if (!valid || !marker.current) return;
    if (previousPath.current !== pathKey) {
      previousPath.current = pathKey;
      phase.current = 0;
    } else if (active) {
      phase.current = (phase.current + delta * PLAYBACK_SPEEDS[speedStep].lapsPerSecond) % 1;
    }
    const head = orbitAt(samples, phase.current * circumference);
    marker.current.position.copy(scenePos(phase.current * circumference, head.x, head.y, circumference, windowMm).pos);
    if (speedStep === PLAYBACK_SPEEDS.length - 1) return;
    const positions = tailGeometry.getAttribute("position") as THREE.BufferAttribute;
    for (let i = 0; i < TAIL_POINTS; i++) {
      const behind = PLAYBACK_SPEEDS[speedStep].tail * (1 - i / (TAIL_POINTS - 1));
      const p = (phase.current - behind + 1) % 1;
      const value = orbitAt(samples, p * circumference);
      const v = scenePos(p * circumference, value.x, value.y, circumference, windowMm).pos;
      positions.setXYZ(i, v.x, v.y, v.z);
    }
    positions.needsUpdate = true;
  });
  if (!valid) return null;
  return (
    <>
      {speedStep === PLAYBACK_SPEEDS.length - 1
        ? <lineLoop geometry={fullGeometry} frustumCulled={false} renderOrder={10} raycast={() => null}><lineBasicMaterial color="#4ade80" transparent opacity={1} depthTest={false} /></lineLoop>
        : <primitive object={tailLine} />}
      <mesh ref={marker} renderOrder={11} raycast={() => null}>
        <sphereGeometry args={[0.09, 16, 16]} />
        <meshStandardMaterial color="#4ade80" emissive="#16a34a" emissiveIntensity={0.8} depthTest={false} />
      </mesh>
    </>
  );
}

function Controls() {
  const { camera, gl, size } = useThree();
  useEffect(() => {
    // Narrow (portrait) canvases have a smaller horizontal field of view: back the camera off so the ring fits.
    const aspect = size.width / Math.max(1, size.height);
    const k = aspect < 1.3 ? 1.3 / aspect : 1;
    camera.position.set(0, 5.6 * k, 6.4 * k);
    camera.lookAt(0, 0, 0);
    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = false;
    return () => controls.dispose();
  }, [camera, gl, size.width, size.height]);
  return null;
}

function Scene({ state, selected, onSelect, dim, windowMm, speedStep, active }: { state: MachineState; selected: string | null; onSelect: (d: Device | null) => void; dim: boolean; windowMm: number; speedStep: number; active: boolean }) {
  const circumference = state.lattice.circumference_m || 1;
  const at = (s: number) => (s / circumference) * 2 * Math.PI;
  /** Maps a measured (x, y) in mm into the tube cross-section; beyond the window it is clamped to the wall and flagged. */
  return (
    <>
      <ambientLight intensity={1.7} />
      <directionalLight position={[5, 8, 5]} intensity={1.8} />
      {/* Vacuum-chamber look. Its cross-section is the deviation display window (±windowMm), not the physical pipe. Ignores clicks so devices stay selectable. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
        <torusGeometry args={[RADIUS, TUBE_R, 20, 200]} />
        <meshPhysicalMaterial color="#7dd3fc" transparent opacity={0.1} roughness={0.15} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
        <torusGeometry args={[RADIUS, TUBE_R, 12, 100]} />
        <meshBasicMaterial color="#7dd3fc" transparent opacity={0.08} wireframe depthWrite={false} />
      </mesh>
      {/* Reference trajectory: the nominal design path, drawn undeviated. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}><torusGeometry args={[RADIUS, 0.03, 10, 160]} /><meshStandardMaterial color="#475569" /></mesh>
      {state.lattice.devices.map((device) => {
        const a = at(device.s_m);
        const isSel = selected === device.name;
        return (
          <mesh key={device.name} position={[RADIUS * Math.cos(a), 0, RADIUS * Math.sin(a)]} onClick={(e) => { e.stopPropagation(); onSelect(device); }}>
            <boxGeometry args={[device.kind === "bpm" ? 0.14 : 0.08, 0.15, 0.08]} />
            <meshStandardMaterial color={isSel ? "#ffffff" : colors[device.kind]} transparent opacity={dim ? 0.5 : 1} />
          </mesh>
        );
      })}
      {state.lattice.devices.filter((d) => d.kind === "bpm").map((device) => {
        const { pos, offScale } = scenePos(device.s_m, state.pvs[bpmPv(device.name, "X")] ?? 0, state.pvs[bpmPv(device.name, "Y")] ?? 0, circumference, windowMm);
        // Measured samples only. No orbit is drawn between BPMs and nothing is tweened between samples.
        return (
          <group key={`${device.name}-m`}>
            <mesh position={pos} onClick={(e) => { e.stopPropagation(); onSelect(device); }}>
              <sphereGeometry args={[0.07, 14, 14]} /><meshStandardMaterial color={offScale ? "#facc15" : "#f43f5e"} emissive={offScale ? "#854d0e" : "#881337"} transparent opacity={dim ? 0.5 : 1} />
            </mesh>
          </group>
        );
      })}
      {state.orbit_profile && <OrbitPlayback state={state} windowMm={windowMm} speedStep={speedStep} active={active} />}
    </>
  );
}

export default function RingScene(props: { state: MachineState; selected: string | null; onSelect: (d: Device | null) => void; dim: boolean; windowMm: number; speedStep: number; active: boolean }) {
  return (
    <Canvas camera={{ position: [0, 5.6, 6.4], fov: 42 }} onPointerMissed={() => props.onSelect(null)}>
      <Controls />
      <Scene {...props} />
    </Canvas>
  );
}
