import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Box,
  CheckCircle2,
  Gauge,
  Home,
  Info,
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
const MODES = ["manual", "pid", "ik", "python"];
const MODE_LABELS = {
  manual: "Manual",
  pid: "PID",
  ik: "IK",
  trajectory: "Trajectory",
  python: "Python"
};
const JOINT_COLORS = ["#0b5fff", "#0f9f6e", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#334155"];
const HOME_Q = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853];
const RENDER_WIDTH = 640;
const RENDER_HEIGHT = 360;
const RENDER_FPS = 15;
const DEFAULT_RENDER_CAMERA = {
  azimuth: 135,
  elevation: -25,
  distance: 1.55,
  lookat: [0.3, 0, 0.45]
};

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

      <section className="tile-layout">
        <Panel
          title="3D MuJoCo Viewport"
          icon={<Box size={18} />}
          info="Live MuJoCo camera stream. Drag to orbit, Shift-drag to pan, scroll to zoom, and double-click to reset."
          className="viewport-panel"
        >
          <MuJoCoViewport telemetry={telemetry} />
        </Panel>

        <Panel
          title="Command Mode"
          icon={<Settings2 size={18} />}
          info="Select how the arm target is interpreted. Python mode marks commands coming from the Python client."
          className="mode-panel"
        >
          <div className="mode-grid">
            {MODES.map((mode) => (
              <button
                key={mode}
                type="button"
                className={`mode-button ${telemetry?.mode === mode ? "active" : ""}`}
                onClick={() => setMode(mode)}
                title={`Switch to ${MODE_LABELS[mode]}`}
              >
                {MODE_LABELS[mode]}
              </button>
            ))}
          </div>
          <div className="command-state">{statusText}</div>
        </Panel>

        <Panel
          title="Events"
          icon={<AlertTriangle size={18} />}
          info="Runtime messages from reset, mode changes, trajectories, IK solves, and render status."
          className="events-panel"
        >
          <EventLog events={telemetry?.events ?? []} />
        </Panel>

        <Panel
          title="Joint Targets"
          icon={<SlidersHorizontal size={18} />}
          info="Edit the seven joint targets, send them directly, or run a quintic trajectory to the same target."
          className="joint-panel"
        >
          <JointSliders q={draftQ} liveQ={q} limits={limits} onChange={setDraftQ} />
          <div className="button-row">
            <button type="button" onClick={() => setDraftQ(Array(7).fill(0))}>Zero</button>
            <button type="button" onClick={() => setDraftQ(HOME_Q)}><Home size={15} /> Home</button>
            <button type="button" className="primary" onClick={sendJointTarget}><Send size={15} /> Send</button>
            <button type="button" onClick={startTrajectory}><Route size={15} /> Traj</button>
          </div>
        </Panel>

        <Panel
          title="Cartesian IK Target"
          icon={<Target size={18} />}
          info="Solve inverse kinematics for an end-effector position, optionally including roll, pitch, and yaw."
          className="ik-panel"
        >
          <CartesianInputs draft={cartDraft} onChange={setCartDraft} onSubmit={sendCartesian} />
        </Panel>

        <Panel
          title="Jacobian Matrix"
          icon={<BarChart3 size={18} />}
          info="Heatmap of the current 6x7 end-effector Jacobian plus its condition number."
          className="jacobian-panel"
        >
          <JacobianHeatmap matrix={telemetry?.jacobian ?? zeroMatrix(6, 7)} />
          <div className="condition-row">
            <span>Condition Number</span>
            <strong>{formatNumber(telemetry?.jacobian_condition, 2)}</strong>
          </div>
        </Panel>

        <Panel
          title="End-Effector Pose"
          icon={<Target size={18} />}
          info="Current end-effector position and XYZ Euler orientation from MuJoCo forward kinematics."
          className="pose-panel"
        >
          <PosePanel telemetry={telemetry} />
        </Panel>

        <Panel
          title="PID Gains"
          icon={<Gauge size={18} />}
          info="Configure the joint-space PID gains used by PID and IK tracking modes."
          className="pid-panel"
        >
          <PidEditor draft={pidDraft} onChange={setPidDraft} onSubmit={applyPid} />
        </Panel>

        <Panel
          title="Live Plots"
          icon={<Activity size={18} />}
          info="Rolling traces for joint positions, joint error, and Cartesian IK error."
          className="plots-panel"
        >
          <div className="plot-grid">
            <LineChart title="Joint Position Tracking" history={telemetry?.history ?? []} series={jointSeries("q")} />
            <LineChart title="Joint Error" history={telemetry?.history ?? []} series={jointSeries("q_error")} />
            <LineChart title="Cartesian Error" history={telemetry?.history ?? []} series={[{ label: "err", color: "#dc2626", get: (p) => p.cartesian_error ?? 0 }]} />
          </div>
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

function Panel({ title, icon, info, className = "", children }) {
  const [showInfo, setShowInfo] = useState(false);
  return (
    <section className={`panel ${className}`}>
      <div className="panel-title">
        {icon}
        <h2>{title}</h2>
        {info && (
          <button
            className="panel-info-button"
            type="button"
            title={`About ${title}`}
            aria-label={`About ${title}`}
            aria-expanded={showInfo}
            onClick={() => setShowInfo((visible) => !visible)}
          >
            <Info size={14} />
          </button>
        )}
      </div>
      {showInfo && <div className="panel-info">{info}</div>}
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

function MuJoCoViewport({ telemetry }) {
  const canvasRef = useRef(null);
  const render = useMuJoCoRender(canvasRef);
  const ee = telemetry?.ee_position ?? [0, 0, 0];
  const statusText = render.error ? "error" : render.connected ? "live" : "connecting";

  return (
    <div className={`viewport render-viewport ${render.interacting ? "interacting" : ""}`}>
      <canvas
        ref={canvasRef}
        width={render.width}
        height={render.height}
        role="img"
        aria-label="Live MuJoCo 3D render"
        title="Drag to orbit. Shift-drag to pan. Scroll to zoom. Double-click to reset."
        tabIndex={0}
        onPointerDown={render.onPointerDown}
        onPointerMove={render.onPointerMove}
        onPointerUp={render.onPointerUp}
        onPointerCancel={render.onPointerUp}
        onWheel={render.onWheel}
        onDoubleClick={render.resetCamera}
        onContextMenu={(event) => event.preventDefault()}
      />
      {(!render.connected || render.error) && (
        <div className="render-placeholder">
          <Radio size={20} />
          <strong>{render.error ? "Render unavailable" : "Starting render"}</strong>
          <span>{render.error ?? "Waiting for MuJoCo frames"}</span>
        </div>
      )}
      <div className="render-badges">
        <span className={`render-chip ${render.error ? "bad" : render.connected ? "ok" : ""}`}>{statusText}</span>
        <span className="render-chip">{formatNumber(render.fps, 1)} fps</span>
      </div>
      <div className="viewport-tools">
        <button className="render-icon-button" type="button" title="Reset camera" onClick={render.resetCamera}>
          <RotateCcw size={14} />
        </button>
      </div>
      <div className="viewport-readout">
        <span>ee</span>
        <strong>{ee.map((v) => formatNumber(v, 3)).join(", ")}</strong>
      </div>
    </div>
  );
}

function useMuJoCoRender(canvasRef) {
  const socketRef = useRef(null);
  const cameraRef = useRef(defaultRenderCamera());
  const interactionRef = useRef({ active: false, mode: "orbit", pointerId: null, lastX: 0, lastY: 0 });
  const [state, setState] = useState({
    camera: defaultRenderCamera(),
    connected: false,
    error: null,
    fps: 0,
    frames: 0,
    height: RENDER_HEIGHT,
    interacting: false,
    width: RENDER_WIDTH
  });

  const sendCamera = (camera) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "camera", ...camera }));
    }
  };

  const updateCamera = (nextCamera) => {
    const camera = normalizeCamera(nextCamera);
    cameraRef.current = camera;
    setState((current) => ({ ...current, camera }));
    sendCamera(camera);
  };

  const resetCamera = () => updateCamera(defaultRenderCamera());

  const onPointerDown = (event) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    interactionRef.current = {
      active: true,
      mode: event.shiftKey || event.button === 1 || event.button === 2 ? "pan" : "orbit",
      pointerId: event.pointerId,
      lastX: event.clientX,
      lastY: event.clientY
    };
    setState((current) => ({ ...current, interacting: true }));
  };

  const onPointerMove = (event) => {
    const interaction = interactionRef.current;
    if (!interaction.active || interaction.pointerId !== event.pointerId) return;
    event.preventDefault();

    const dx = event.clientX - interaction.lastX;
    const dy = event.clientY - interaction.lastY;
    interaction.lastX = event.clientX;
    interaction.lastY = event.clientY;

    const camera = cameraRef.current;
    if (interaction.mode === "pan") {
      const azimuth = degToRad(camera.azimuth);
      const right = [Math.cos(azimuth), Math.sin(azimuth)];
      const scale = camera.distance * 0.0017;
      updateCamera({
        ...camera,
        lookat: [
          camera.lookat[0] - right[0] * dx * scale,
          camera.lookat[1] - right[1] * dx * scale,
          camera.lookat[2] + dy * scale
        ]
      });
      return;
    }

    updateCamera({
      ...camera,
      azimuth: camera.azimuth - dx * 0.35,
      elevation: camera.elevation - dy * 0.25
    });
  };

  const onPointerUp = (event) => {
    const interaction = interactionRef.current;
    if (interaction.pointerId !== event.pointerId) return;
    interactionRef.current = { active: false, mode: "orbit", pointerId: null, lastX: 0, lastY: 0 };
    setState((current) => ({ ...current, interacting: false }));
  };

  const onWheel = (event) => {
    event.preventDefault();
    const camera = cameraRef.current;
    updateCamera({
      ...camera,
      distance: camera.distance * Math.exp(event.deltaY * 0.001)
    });
  };

  useEffect(() => {
    let socket;
    let reconnectTimer;
    let closed = false;
    let context;
    let imageData;
    let framesThisSecond = 0;
    let lastFpsTick = performance.now();

    const drawFrame = (buffer) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      if (!context) context = canvas.getContext("2d", { alpha: false });
      if (!context) return;
      if (!imageData) imageData = context.createImageData(RENDER_WIDTH, RENDER_HEIGHT);

      const rgb = new Uint8Array(buffer);
      const expectedLength = RENDER_WIDTH * RENDER_HEIGHT * 3;
      if (rgb.length !== expectedLength) {
        setState((current) => ({
          ...current,
          connected: false,
          error: `Frame size ${rgb.length} != ${expectedLength}`
        }));
        return;
      }

      const rgba = imageData.data;
      for (let source = 0, target = 0; source < expectedLength; source += 3, target += 4) {
        rgba[target] = rgb[source];
        rgba[target + 1] = rgb[source + 1];
        rgba[target + 2] = rgb[source + 2];
        rgba[target + 3] = 255;
      }
      context.putImageData(imageData, 0, 0);

      framesThisSecond += 1;
      const now = performance.now();
      const elapsed = now - lastFpsTick;
      if (elapsed >= 1000) {
        const measuredFps = (framesThisSecond * 1000) / elapsed;
        setState((current) => ({
          ...current,
          connected: true,
          error: null,
          fps: measuredFps,
          frames: current.frames + framesThisSecond
        }));
        framesThisSecond = 0;
        lastFpsTick = now;
      }
    };

    const connect = () => {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      const params = new URLSearchParams({
        width: String(RENDER_WIDTH),
        height: String(RENDER_HEIGHT),
        fps: String(RENDER_FPS)
      });
      socket = new WebSocket(`${scheme}://${window.location.host}/ws/render?${params}`);
      socketRef.current = socket;
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        setState((current) => ({ ...current, connected: true, error: null }));
        sendCamera(cameraRef.current);
      };

      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          const message = JSON.parse(event.data);
          if (message.type === "render-info") {
            setState((current) => ({
              ...current,
              width: message.width ?? RENDER_WIDTH,
              height: message.height ?? RENDER_HEIGHT
            }));
          }
          if (message.type === "render-error") {
            setState((current) => ({
              ...current,
              connected: false,
              error: message.message ?? "Render error",
              fps: 0
            }));
          }
          return;
        }
        drawFrame(event.data);
      };

      socket.onclose = () => {
        if (closed) return;
        setState((current) => ({ ...current, connected: false, fps: 0 }));
        reconnectTimer = window.setTimeout(connect, 1000);
      };

      socket.onerror = () => {
        setState((current) => ({ ...current, connected: false, error: "Render socket error", fps: 0 }));
        socket.close();
      };
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (socket) socket.close();
      if (socketRef.current === socket) socketRef.current = null;
    };
  }, [canvasRef]);

  return {
    ...state,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onWheel,
    resetCamera
  };
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

function defaultRenderCamera() {
  return {
    ...DEFAULT_RENDER_CAMERA,
    lookat: [...DEFAULT_RENDER_CAMERA.lookat]
  };
}

function normalizeCamera(camera) {
  const lookat = camera.lookat ?? DEFAULT_RENDER_CAMERA.lookat;
  return {
    azimuth: wrapDegrees(numberOrFallback(camera.azimuth, DEFAULT_RENDER_CAMERA.azimuth)),
    elevation: clamp(numberOrFallback(camera.elevation, DEFAULT_RENDER_CAMERA.elevation), -89, 25),
    distance: clamp(numberOrFallback(camera.distance, DEFAULT_RENDER_CAMERA.distance), 0.3, 4),
    lookat: [
      clamp(numberOrFallback(lookat[0], DEFAULT_RENDER_CAMERA.lookat[0]), -0.6, 1.2),
      clamp(numberOrFallback(lookat[1], DEFAULT_RENDER_CAMERA.lookat[1]), -0.9, 0.9),
      clamp(numberOrFallback(lookat[2], DEFAULT_RENDER_CAMERA.lookat[2]), 0.05, 1.4)
    ]
  };
}

function numberOrFallback(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function wrapDegrees(value) {
  return ((value % 360) + 360) % 360;
}

function numberOrFirst(value, fallback) {
  if (Array.isArray(value)) return Number(value[0] ?? fallback);
  return Number(value ?? fallback);
}

function zeroMatrix(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
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
