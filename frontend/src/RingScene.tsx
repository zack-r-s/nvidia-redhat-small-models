import { Canvas, useThree } from "@react-three/fiber";
import { useEffect } from "react";
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
/** Equivalent magnification of the window mapping (scene length / true length, both relative to the ring). */
export const magnificationFor = (windowMm: number, circumference_m: number) => TUBE_R / (windowMm * 1e-3 * (RADIUS / (circumference_m / (2 * Math.PI))));

function Controls() {
  const { camera, gl } = useThree();
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.enableDamping = false;
    return () => controls.dispose();
  }, [camera, gl]);
  return null;
}

function Scene({ state, selected, onSelect, dim, windowMm }: { state: MachineState; selected: string | null; onSelect: (d: Device | null) => void; dim: boolean; windowMm: number }) {
  const circumference = state.lattice.circumference_m || 1;
  const at = (s: number) => (s / circumference) * 2 * Math.PI;
  /** Maps a measured (x, y) in mm into the tube cross-section; beyond the window it is clamped to the wall and flagged. */
  const scenePos = (s_m: number, x: number, y: number) => {
    const a = at(s_m);
    const norm = Math.hypot(x, y);
    const k = norm > windowMm ? windowMm / norm : 1;
    const r = RADIUS + ((x * k) / windowMm) * TUBE_R;
    return { pos: new THREE.Vector3(r * Math.cos(a), ((y * k) / windowMm) * TUBE_R, r * Math.sin(a)), offScale: norm > windowMm };
  };
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
        const { pos, offScale } = scenePos(device.s_m, state.pvs[bpmPv(device.name, "X")] ?? 0, state.pvs[bpmPv(device.name, "Y")] ?? 0);
        // Measured samples only. No orbit is drawn between BPMs and nothing is tweened between samples.
        return (
          <group key={`${device.name}-m`}>
            <mesh position={pos} onClick={(e) => { e.stopPropagation(); onSelect(device); }}>
              <sphereGeometry args={[0.07, 14, 14]} /><meshStandardMaterial color={offScale ? "#facc15" : "#f43f5e"} emissive={offScale ? "#854d0e" : "#881337"} transparent opacity={dim ? 0.5 : 1} />
            </mesh>
          </group>
        );
      })}
    </>
  );
}

export default function RingScene(props: { state: MachineState; selected: string | null; onSelect: (d: Device | null) => void; dim: boolean; windowMm: number }) {
  return (
    <Canvas camera={{ position: [0, 8, 8], fov: 42 }} onPointerMissed={() => props.onSelect(null)}>
      <Controls />
      <Scene {...props} />
    </Canvas>
  );
}
