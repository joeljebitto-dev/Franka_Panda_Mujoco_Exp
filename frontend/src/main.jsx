import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Box,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Gauge,
  Home,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Route,
  Send,
  Settings2,
  SlidersHorizontal,
  Target,
  Wifi,
  WifiOff
} from "lucide-react";
import "./styles.css";

const API_BASE = "";
const MODES = ["manual", "pid", "ik", "trajectory", "ai"];
const MODE_LABELS = {
  manual: "Manual",
  pid: "PID",
  ik: "IK",
  trajectory: "Trajectory",
  ai: "AI"
};
const JOINT_COLORS = ["#0b5fff", "#0f9f6e", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#334155"];
const HOME_Q = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853];

function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [connected, setConnected] = useState(false);
  const [draftQ, setDraftQ] = useState(HOME_Q);
  const [pidDraft, setPidDraft] = useState({ kp: 1, ki: 0, kd: 0.05, integral_limit: 0.5 });
  const [cartDraft, setCartDraft] = useState({ x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0, useOrientation: false });
  const [statusText, setStatusText] = useState("Connecting");
  const [targetInitialized, setTargetInitialized] = useState(false);
  const [cartInitialized, setCartInitialized] = useState(false);

  useTelemetry(setTelemetry, setConnected, setStatusText);

  useEffect(() => {
    if (!telemetry || targetInitialized) return;
    setDraftQ(telemetry.target_q ?? telemetry.q ?? HOME_Q);
    setPidDraft({
      kp: numberOrFirst(telemetry.pid?.kp, 1),
      ki: numberOrFirst(telemetry.pid?.ki, 0),
      kd: numberOrFirst(telemetry.pid?.kd, 0.05),
      integral_limit: 0.5
    });
    setTargetInitialized(true);
  }, [telemetry, targetInitialized]);

  useEffect(() => {
    if (!telemetry || cartInitialized) return;
    const eulerDeg = (telemetry.ee_euler_xyz ?? [0, 0, 0]).map(radToDeg);
    setCartDraft({
      x: telemetry.ee_position?.[0] ?? 0,
      y: telemetry.ee_position?.[1] ?? 0,
      z: telemetry.ee_position?.[2] ?? 0,
      roll: eulerDeg[0],
      pitch: eulerDeg[1],
      yaw: eulerDeg[2],
      useOrientation: false
    });
    setCartInitialized(true);
  }, [telemetry, cartInitialized]);

  const sendCommand = async (path, body = null) => {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail?.message ?? data?.detail?.message ?? JSON.stringify(data.detail ?? data));
    }
    setTelemetry(data);
    return data;
  };

  const runCommand = async (label, fn) => {
    setStatusText(label);
    try {
      await fn();
      setStatusText("Command applied");
    } catch (error) {
      setStatusText(error.message);
    }
  };

  const setMode = (mode) => runCommand(`Mode ${mode}`, () => sendCommand("/api/control/mode", { mode }));
  const sendJointTarget = () => runCommand("Joint target", () => sendCommand("/api/target/joint", { q: draftQ, mode: telemetry?.mode === "trajectory" ? "pid" : telemetry?.mode }));
  const resetRobot = () => runCommand("Reset", () => sendCommand("/api/reset"));
  const setRunning = (running) => runCommand(running ? "Running" : "Paused", () => sendCommand("/api/control/running", { running }));
  const applyPid = () => runCommand("PID gains", () => sendCommand("/api/pid", coercePid(pidDraft)));
  const startTrajectory = () => runCommand("Trajectory", () => sendCommand("/api/trajectory/start", { goal_q: draftQ, duration_s: 3.5, method: "quintic" }));
  const sendCartesian = () => runCommand("IK target", () => {
    const rotation = eulerXyzToMatrix(degToRad(cartDraft.roll), degToRad(cartDraft.pitch), degToRad(cartDraft.yaw));
    return sendCommand("/api/target/cartesian", {
      position: [Number(cartDraft.x), Number(cartDraft.y), Number(cartDraft.z)],
      rotation,
      use_orientation: cartDraft.useOrientation,
      mode_after_solve: "pid"
    });
  });

  const q = telemetry?.q ?? HOME_Q;
  const limits = telemetry?.actuator_limits ?? [
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973]
  ];

  return (
    <main className="dashboard">
      <header className="topbar">
        <div className="brand">
          <Box size={24} />
          <div>
            <h1>Franka Panda 7-DOF Dashboard</h1>
            <p>MuJoCo realtime control and telemetry</p>
          </div>
        </div>
        <div className="status-strip">
          <StatusPill connected={connected} text={connected ? "Connected" : "Disconnected"} />
          <Metric label="FPS" value={formatNumber(telemetry?.metrics?.sim_fps, 0)} />
          <Metric label="dt" value={`${formatNumber((telemetry?.timestep_s ?? 0) * 1000, 1)} ms`} />
          <Metric label="Mode" value={MODE_LABELS[telemetry?.mode] ?? "Manual"} />
          <button className="icon-button" type="button" title={telemetry?.running ? "Pause simulation" : "Resume simulation"} onClick={() => setRunning(!telemetry?.running)}>
            {telemetry?.running ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button className="icon-button" type="button" title="Reset robot" onClick={resetRobot}>
            <RotateCcw size={18} />
          </button>
        </div>
      </header>

      <section className="grid">
        <Panel title="3D MuJoCo Viewport" number="1" icon={<Box size={18} />} className="viewport-panel">
          <RobotViewport q={q} ee={telemetry?.ee_position} connected={connected} />
        </Panel>

        <Panel title="Mode Selector" number="2" icon={<Settings2 size={18} />} className="mode-panel">
          <div className="mode-grid">
            {MODES.map((mode) => (
              <button
                key={mode}
                type="button"
                className={`mode-button ${telemetry?.mode === mode ? "active" : ""}`}
                onClick={() => setMode(mode)}
                disabled={mode === "ai"}
                title={mode === "ai" ? "Reserved for future ML control" : `Switch to ${MODE_LABELS[mode]}`}
              >
                {mode === "ai" && <BrainCircuit size={15} />}
                {MODE_LABELS[mode]}
              </button>
            ))}
          </div>
          <div className="command-state">{statusText}</div>
        </Panel>

        <Panel title="Joint Sliders (q)" number="3" icon={<SlidersHorizontal size={18} />} className="joint-panel">
          <JointSliders q={draftQ} liveQ={q} limits={limits} onChange={setDraftQ} />
          <div className="button-row">
            <button type="button" onClick={() => setDraftQ(Array(7).fill(0))}>All Zero</button>
            <button type="button" onClick={() => setDraftQ(HOME_Q)}>Home</button>
            <button type="button" className="primary" onClick={sendJointTarget}><Send size={15} /> Send</button>
          </div>
        </Panel>

        <Panel title="Cartesian Target Inputs" number="4" icon={<Target size={18} />}>
          <CartesianInputs draft={cartDraft} onChange={setCartDraft} onSubmit={sendCartesian} />
        </Panel>

        <Panel title="PID Gain Editor" number="5" icon={<Gauge size={18} />}>
          <PidEditor draft={pidDraft} onChange={setPidDraft} onSubmit={applyPid} />
        </Panel>

        <Panel title="End-Effector Pose" number="6" icon={<Target size={18} />}>
          <PosePanel telemetry={telemetry} />
        </Panel>

        <Panel title="Transformation Matrix" number="7" icon={<ClipboardList size={18} />}>
          <Matrix values={telemetry?.transform ?? identity4()} />
        </Panel>

        <Panel title="Rotation / Euler / Quaternion" number="8" icon={<RotateCcw size={18} />}>
          <RotationPanel telemetry={telemetry} />
        </Panel>

        <Panel title="Jacobian Matrix (J)" number="9" icon={<BarChart3 size={18} />} className="jacobian-panel">
          <JacobianHeatmap matrix={telemetry?.jacobian ?? zeroMatrix(6, 7)} />
          <div className="condition-row">
            <span>Condition Number</span>
            <strong>{formatNumber(telemetry?.jacobian_condition, 2)}</strong>
          </div>
        </Panel>

        <Panel title="Telemetry & System State" number="10" icon={<Activity size={18} />}>
          <TelemetryPanel telemetry={telemetry} connected={connected} />
        </Panel>

        <Panel title="Live Plots" number="11" icon={<Activity size={18} />} className="plots-panel">
          <div className="plot-grid">
            <LineChart title="Joint Position Tracking" history={telemetry?.history ?? []} series={jointSeries("q")} />
            <LineChart title="Joint Error" history={telemetry?.history ?? []} series={jointSeries("q_error")} />
            <LineChart title="Cartesian Error" history={telemetry?.history ?? []} series={[{ label: "err", color: "#dc2626", get: (p) => p.cartesian_error ?? 0 }]} />
          </div>
        </Panel>

        <Panel title="Event Log / Alerts" number="12" icon={<AlertTriangle size={18} />} className="events-panel">
          <EventLog events={telemetry?.events ?? []} />
          <button type="button" className="wide-button" onClick={startTrajectory}><Route size={15} /> Run Quintic Trajectory</button>
        </Panel>
      </section>
    </main>
  );
}

function useTelemetry(setTelemetry, setConnected, setStatusText) {
  useEffect(() => {
    let socket;
    let pollId;
    let closed = false;

    const connect = () => {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${window.location.host}/ws/telemetry`);
      socket.onopen = () => {
        setConnected(true);
        setStatusText("Live telemetry");
      };
      socket.onmessage = (event) => {
        setTelemetry(JSON.parse(event.data));
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) startPolling();
      };
      socket.onerror = () => {
        setConnected(false);
        socket.close();
      };
    };

    const startPolling = () => {
      if (pollId) return;
      setStatusText("Polling API");
      pollId = window.setInterval(async () => {
        try {
          const response = await fetch(`${API_BASE}/api/state`);
          const data = await response.json();
          setTelemetry(data);
          setConnected(false);
        } catch {
          setStatusText("Backend unavailable");
        }
      }, 500);
    };

    connect();
    return () => {
      closed = true;
      if (socket) socket.close();
      if (pollId) window.clearInterval(pollId);
    };
  }, [setConnected, setStatusText, setTelemetry]);
}

function Panel({ title, number, icon, className = "", children }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-title">
        <span className="panel-number">{number}</span>
        {icon}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function StatusPill({ connected, text }) {
  return (
    <div className={`status-pill ${connected ? "ok" : "bad"}`}>
      {connected ? <Wifi size={15} /> : <WifiOff size={15} />}
      {text}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RobotViewport({ q, ee, connected }) {
  const points = useMemo(() => projectArm(q), [q]);
  const path = points.map((p) => `${p.x},${p.y}`).join(" ");
  return (
    <div className="viewport">
      <svg viewBox="0 0 520 360" role="img" aria-label="Robot arm visualization">
        <defs>
          <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
            <path d="M 28 0 L 0 0 0 28" fill="none" stroke="#dbe4f0" strokeWidth="1" />
          </pattern>
          <linearGradient id="linkGradient" x1="0" x2="1">
            <stop offset="0" stopColor="#f8fafc" />
            <stop offset="1" stopColor="#94a3b8" />
          </linearGradient>
        </defs>
        <rect x="30" y="30" width="460" height="280" fill="url(#grid)" rx="6" />
        <line x1="80" y1="300" x2="145" y2="300" stroke="#dc2626" strokeWidth="3" />
        <line x1="80" y1="300" x2="80" y2="235" stroke="#0b5fff" strokeWidth="3" />
        <line x1="80" y1="300" x2="128" y2="268" stroke="#0f9f6e" strokeWidth="3" />
        <text x="150" y="304" className="axis red">X</text>
        <text x="72" y="230" className="axis blue">Z</text>
        <text x="132" y="264" className="axis green">Y</text>
        <ellipse cx="250" cy="304" rx="72" ry="18" fill="#cbd5e1" opacity="0.5" />
        <rect x="204" y="270" width="92" height="34" rx="5" fill="#e2e8f0" stroke="#64748b" />
        <polyline points={path} fill="none" stroke="#475569" strokeWidth="22" strokeLinecap="round" strokeLinejoin="round" />
        <polyline points={path} fill="none" stroke="url(#linkGradient)" strokeWidth="16" strokeLinecap="round" strokeLinejoin="round" />
        {points.map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.y} r={index === 0 ? 15 : 11} fill="#e5e7eb" stroke="#334155" strokeWidth="3" />
            <circle cx={point.x} cy={point.y} r="4" fill={JOINT_COLORS[index % JOINT_COLORS.length]} />
          </g>
        ))}
        <circle cx={points.at(-1).x} cy={points.at(-1).y} r="7" fill={connected ? "#0f9f6e" : "#dc2626"} />
      </svg>
      <div className="viewport-readout">
        <span>ee</span>
        <strong>{(ee ?? [0, 0, 0]).map((v) => formatNumber(v, 3)).join(", ")}</strong>
      </div>
    </div>
  );
}

function JointSliders({ q, liveQ, limits, onChange }) {
  return (
    <div className="joint-list">
      {q.map((value, index) => {
        const [min, max] = limits[index] ?? [-3.14, 3.14];
        return (
          <label className="joint-row" key={index}>
            <span>q{index + 1}</span>
            <input
              type="range"
              min={min}
              max={max}
              step="0.001"
              value={value}
              onChange={(event) => updateIndex(onChange, q, index, Number(event.target.value))}
            />
            <input
              type="number"
              step="0.001"
              value={roundForInput(value)}
              onChange={(event) => updateIndex(onChange, q, index, Number(event.target.value))}
            />
            <em>{formatNumber(liveQ[index], 2)}</em>
          </label>
        );
      })}
    </div>
  );
}

function CartesianInputs({ draft, onChange, onSubmit }) {
  const update = (key, value) => onChange({ ...draft, [key]: value });
  return (
    <div className="form-grid">
      {["x", "y", "z"].map((key) => (
        <label key={key}>
          <span>{key}</span>
          <input type="number" step="0.001" value={roundForInput(draft[key])} onChange={(e) => update(key, Number(e.target.value))} />
          <em>m</em>
        </label>
      ))}
      {["roll", "pitch", "yaw"].map((key) => (
        <label key={key}>
          <span>{key}</span>
          <input type="number" step="0.1" value={roundForInput(draft[key])} onChange={(e) => update(key, Number(e.target.value))} />
          <em>deg</em>
        </label>
      ))}
      <label className="checkbox-row">
        <input type="checkbox" checked={draft.useOrientation} onChange={(e) => update("useOrientation", e.target.checked)} />
        <span>Use orientation</span>
      </label>
      <button type="button" className="primary wide-button" onClick={onSubmit}><Send size={15} /> Send IK Target</button>
    </div>
  );
}

function PidEditor({ draft, onChange, onSubmit }) {
  const update = (key, value) => onChange({ ...draft, [key]: value });
  return (
    <div className="form-grid pid-form">
      {["kp", "ki", "kd", "integral_limit"].map((key) => (
        <label key={key}>
          <span>{key}</span>
          <input type="number" step="0.01" value={draft[key]} onChange={(e) => update(key, Number(e.target.value))} />
        </label>
      ))}
      <button type="button" className="primary wide-button" onClick={onSubmit}><CheckCircle2 size={15} /> Apply Gains</button>
    </div>
  );
}

function PosePanel({ telemetry }) {
  const position = telemetry?.ee_position ?? [0, 0, 0];
  const euler = (telemetry?.ee_euler_xyz ?? [0, 0, 0]).map(radToDeg);
  return (
    <div className="kv-table">
      {["x", "y", "z"].map((label, index) => <Row key={label} label={label} value={`${formatNumber(position[index], 4)} m`} />)}
      {["roll", "pitch", "yaw"].map((label, index) => <Row key={label} label={label} value={`${formatNumber(euler[index], 2)} deg`} />)}
    </div>
  );
}

function RotationPanel({ telemetry }) {
  const rotation = telemetry?.ee_rotation ?? zeroMatrix(3, 3);
  const quaternion = telemetry?.ee_quaternion_wxyz ?? [1, 0, 0, 0];
  return (
    <div className="rotation-stack">
      <Matrix values={rotation} compact />
      <div className="chip-row">
        {["w", "x", "y", "z"].map((label, index) => (
          <span className="value-chip" key={label}>{label}: {formatNumber(quaternion[index], 3)}</span>
        ))}
      </div>
    </div>
  );
}

function Matrix({ values, compact = false }) {
  return (
    <div className={`matrix ${compact ? "compact" : ""}`}>
      {values.flat().map((value, index) => (
        <span key={index}>{formatNumber(value, 3)}</span>
      ))}
    </div>
  );
}

function JacobianHeatmap({ matrix }) {
  const max = Math.max(0.001, ...matrix.flat().map((v) => Math.abs(v)));
  return (
    <div className="heatmap">
      <div className="heatmap-labels rows">{["vx", "vy", "vz", "wx", "wy", "wz"].map((label) => <span key={label}>{label}</span>)}</div>
      <div className="heatmap-grid">
        {matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => (
          <span
            key={`${rowIndex}-${columnIndex}`}
            title={`${formatNumber(value, 4)}`}
            style={{ background: heatColor(value / max) }}
          />
        )))}
      </div>
      <div className="heatmap-labels cols">{Array.from({ length: 7 }, (_, i) => <span key={i}>q{i + 1}</span>)}</div>
    </div>
  );
}

function TelemetryPanel({ telemetry, connected }) {
  return (
    <div className="kv-table">
      <Row label="WebSocket" value={connected ? "Connected" : "Fallback"} />
      <Row label="Sim Time" value={`${formatNumber(telemetry?.time_s, 3)} s`} />
      <Row label="Backend" value={`${formatNumber(telemetry?.metrics?.backend_latency_ms, 2)} ms`} />
      <Row label="MuJoCo" value={telemetry?.metrics?.mujoco_status ?? "ok"} />
      <Row label="History" value={telemetry?.metrics?.history_points ?? 0} />
      <Row label="Running" value={telemetry?.running ? "yes" : "no"} />
    </div>
  );
}

function EventLog({ events }) {
  if (!events.length) {
    return <div className="empty-log">No events yet</div>;
  }
  return (
    <div className="event-list">
      {events.slice(0, 8).map((event, index) => (
        <div className={`event ${event.level}`} key={`${event.time_s}-${index}`}>
          <span>{formatNumber(event.time_s, 2)}</span>
          <strong>{event.level}</strong>
          <p>{event.message}</p>
        </div>
      ))}
    </div>
  );
}

function LineChart({ title, history, series }) {
  const width = 320;
  const height = 150;
  const padding = 22;
  const points = history.slice(-120);
  const values = points.flatMap((point) => series.map((item) => item.get(point)));
  const min = Math.min(-0.001, ...values);
  const max = Math.max(0.001, ...values);
  const t0 = points[0]?.time_s ?? 0;
  const t1 = points.at(-1)?.time_s ?? 1;
  const x = (t) => padding + ((t - t0) / Math.max(0.001, t1 - t0)) * (width - padding * 2);
  const y = (v) => height - padding - ((v - min) / Math.max(0.001, max - min)) * (height - padding * 2);
  return (
    <div className="chart">
      <h3>{title}</h3>
      <svg viewBox={`0 0 ${width} ${height}`}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cbd5e1" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#cbd5e1" />
        {series.map((item) => {
          const path = points.map((point) => `${x(point.time_s)},${y(item.get(point))}`).join(" ");
          return <polyline key={item.label} points={path} fill="none" stroke={item.color} strokeWidth="2" />;
        })}
      </svg>
      <div className="legend">
        {series.slice(0, 4).map((item) => <span key={item.label} style={{ color: item.color }}>{item.label}</span>)}
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="kv-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function projectArm(q) {
  const base = { x: 250, y: 275 };
  const lengths = [48, 58, 50, 45, 38, 30, 24];
  const points = [base];
  let angle = -Math.PI / 2.3;
  let lift = 0;
  for (let i = 0; i < 7; i += 1) {
    angle += (q[i] ?? 0) * (i % 2 === 0 ? 0.42 : -0.34);
    lift += Math.sin(q[i] ?? 0) * 4;
    const previous = points.at(-1);
    points.push({
      x: previous.x + Math.cos(angle) * lengths[i],
      y: previous.y + Math.sin(angle) * lengths[i] - lift
    });
  }
  return points;
}

function jointSeries(key) {
  return Array.from({ length: 7 }, (_, index) => ({
    label: `q${index + 1}`,
    color: JOINT_COLORS[index],
    get: (point) => point[key]?.[index] ?? 0
  }));
}

function updateIndex(setter, values, index, value) {
  setter(values.map((item, itemIndex) => (itemIndex === index ? value : item)));
}

function eulerXyzToMatrix(roll, pitch, yaw) {
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  return [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr]
  ];
}

function coercePid(draft) {
  return {
    kp: Number(draft.kp),
    ki: Number(draft.ki),
    kd: Number(draft.kd),
    integral_limit: Number(draft.integral_limit)
  };
}

function numberOrFirst(value, fallback) {
  if (Array.isArray(value)) return Number(value[0] ?? fallback);
  return Number(value ?? fallback);
}

function zeroMatrix(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function identity4() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ];
}

function heatColor(value) {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped >= 0) {
    const light = 96 - clamped * 48;
    return `hsl(8 78% ${light}%)`;
  }
  const light = 96 + clamped * 48;
  return `hsl(218 78% ${light}%)`;
}

function formatNumber(value, decimals = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "0";
  if (!Number.isFinite(Number(value))) return "inf";
  return Number(value).toFixed(decimals);
}

function roundForInput(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return 0;
  return Number(value.toFixed ? value.toFixed(4) : value);
}

function degToRad(value) {
  return (Number(value) * Math.PI) / 180;
}

function radToDeg(value) {
  return (Number(value) * 180) / Math.PI;
}

createRoot(document.getElementById("root")).render(<App />);
