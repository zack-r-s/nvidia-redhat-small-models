import { Canvas } from "@react-three/fiber";
import type { MachineState } from "./types";

const colors = { bpm: "#38bdf8", corrector: "#f59e0b", quadrupole: "#a78bfa", dipole: "#64748b" };

function Scene({ state }: { state: MachineState }) {
  const circumference = state.lattice.circumference_m || 1;
  const bpmDevices = state.lattice.devices.filter((device) => device.kind === "bpm");
  return (
    <>
      <ambientLight intensity={1.7} />
      <directionalLight position={[5, 8, 5]} intensity={1.8} />
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[3.2, 0.035, 10, 160]} />
        <meshStandardMaterial color="#475569" />
      </mesh>
      {state.lattice.devices.map((device) => {
        const angle = (device.s_m / circumference) * 2 * Math.PI;
        return (
          <mesh key={device.name} position={[3.2 * Math.cos(angle), 0, 3.2 * Math.sin(angle)]}>
            <boxGeometry args={[device.kind === "bpm" ? 0.12 : 0.08, 0.15, 0.08]} />
            <meshStandardMaterial color={colors[device.kind]} />
          </mesh>
        );
      })}
      {bpmDevices.map((device) => {
        const angle = (device.s_m / circumference) * 2 * Math.PI;
        const x = state.pvs[`RING:DIAG:${device.name}:POSITION:X`] || 0;
        const y = state.pvs[`RING:DIAG:${device.name}:POSITION:Y`] || 0;
        // Only measured BPM samples are drawn; this does not invent an orbit between sensors.
        const radius = 3.2 + Math.max(-0.4, Math.min(0.4, x * 0.15));
        return (
          <mesh key={`${device.name}-reading`} position={[radius * Math.cos(angle), 0.25 + Math.max(-0.3, Math.min(0.3, y * 0.15)), radius * Math.sin(angle)]}>
            <sphereGeometry args={[0.09, 14, 14]} />
            <meshStandardMaterial color="#f43f5e" emissive="#881337" />
          </mesh>
        );
      })}
    </>
  );
}

export default function RingScene({ state }: { state: MachineState }) {
  return (
    <Canvas camera={{ position: [0, 8, 8], fov: 42 }}>
      <Scene state={state} />
    </Canvas>
  );
}
