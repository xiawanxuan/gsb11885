// 主线程：仅负责渲染、UI 与输入。所有动画重计算在 Worker 中完成。
import { Renderer } from './renderer.js';
import { vec3 } from './math.js';
import { openDB, dbGet, dbPut } from './db.js';

const MAX_GPU_BONES = 64;

const $ = (id) => document.getElementById(id);
const canvas = $('gl');

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const state = {
  boneCount: 32,
  params: { clipA: 'wave', clipB: 'twist', blend: 0, speed: 1, ikEnabled: false, ikWeight: 1, ikTarget: [1.2, 2.0, 0.5] },
  degraded: false,
  geometry: null,
  parents: null,
  segmentLength: 0.35,
  latestFrame: null,
  showBones: true,
  camera: { theta: 0.7, phi: 1.1, dist: 7 },
  fps: 0,
  frameMs: 0,
  simMs: 0,
  longTasks: 0,
  dragMode: null, // 'orbit' | 'ik'
};

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------
const worker = new Worker('./js/worker.js', { type: 'module' });

worker.onmessage = (e) => {
  const d = e.data;
  if (d.type === 'geometry') {
    state.geometry = d;
    state.degraded = d.degraded;
    state.parents = d.parents;
    state.segmentLength = d.segmentLength;
    renderer.uploadGeometry(d);
    updateStatus(d);
  } else if (d.type === 'frame') {
    state.latestFrame = d;
    state.simMs = state.simMs * 0.9 + d.simTime * 0.1;
  } else if (d.type === 'ready') {
    setStatus('就绪');
  }
};

function rebuild(boneCount) {
  state.boneCount = boneCount;
  state.latestFrame = null;
  worker.postMessage({ type: 'rebuild', boneCount });
}

function pushParams() {
  worker.postMessage({ type: 'params', params: state.params });
}

// ---------------------------------------------------------------------------
// 渲染器
// ---------------------------------------------------------------------------
const renderer = new Renderer(canvas, MAX_GPU_BONES);

// ---------------------------------------------------------------------------
// 性能观测：FPS + PerformanceObserver（longtask 监控主线程卡顿）
// ---------------------------------------------------------------------------
let lastT = performance.now();
let fpsSmooth = 60;
if ('PerformanceObserver' in window) {
  try {
    new PerformanceObserver((list) => {
      for (const _ of list.getEntries()) {
        state.longTasks++;
        $('longtasks').textContent = String(state.longTasks);
      }
    }).observe({ entryTypes: ['longtask'] });
  } catch (_) { /* longtask 不可用时静默降级 */ }
}

function loop(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  fpsSmooth = fpsSmooth * 0.95 + (1 / Math.max(dt, 1e-4)) * 0.05;
  state.fps = fpsSmooth;
  state.frameMs = state.frameMs * 0.9 + dt * 1000 * 0.1;

  // 驱动 Worker 仿真（主线程只发消息，不做计算）
  worker.postMessage({ type: 'tick', dt });

  if (state.geometry && state.latestFrame) {
    const cam = state.camera;
    const eye = [
      cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta),
      cam.dist * Math.cos(cam.phi) + state.geometry.height * 0.4,
      cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta),
    ];
    renderer.render({
      camera: { eye, center: [0, state.geometry.height * 0.45, 0] },
      skinMatrices: state.latestFrame.skinMatrices,
      cpuPositions: state.latestFrame.cpuPositions,
      cpuNormals: state.latestFrame.cpuNormals,
      jointPositions: state.latestFrame.jointPositions,
      parents: state.parents,
      ikTarget: state.params.ikEnabled ? state.params.ikTarget : null,
      showBones: state.showBones,
      height: state.geometry.height,
    });
  }
  requestAnimationFrame(loop);
}

// HUD 刷新（低频，避免 DOM 开销影响帧率）
setInterval(() => {
  $('fps').textContent = state.fps.toFixed(0);
  $('frametime').textContent = state.frameMs.toFixed(1) + ' ms';
  $('simtime').textContent = state.simMs.toFixed(2) + ' ms';
  $('bonecount').textContent = String(state.boneCount);
}, 250);

function setStatus(text) { $('status').textContent = text; }

function updateStatus(g) {
  const mode = g.degraded
    ? `降级模式：骨骼数 ${g.boneCount} > 上限 ${g.maxGpuBones} → 刚性绑定 + CPU 蒙皮（Worker）`
    : `GPU 蒙皮：${g.boneCount} 骨骼 × 4 影响（上限 ${g.maxGpuBones}）`;
  $('mode').textContent = mode;
  $('mode').className = g.degraded ? 'degraded' : 'normal';
  setStatus(g.clipsRestored ? '已从 IndexedDB 恢复动画剪辑' : '已生成动画剪辑并写入 IndexedDB');
}

// ---------------------------------------------------------------------------
// 输入：相机轨道 + IK 目标拖动
// ---------------------------------------------------------------------------
function rayFromMouse(mx, my) {
  const rect = canvas.getBoundingClientRect();
  const nx = ((mx - rect.left) / rect.width) * 2 - 1;
  const ny = -(((my - rect.top) / rect.height) * 2 - 1);
  const cam = state.camera;
  const h = state.geometry ? state.geometry.height : 4;
  const eye = [
    cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta),
    cam.dist * Math.cos(cam.phi) + h * 0.4,
    cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta),
  ];
  const center = [0, h * 0.45, 0];
  const aspect = rect.width / rect.height;
  const fov = Math.PI / 4;
  const fwd = vec3.normalize(vec3.sub(center, eye));
  const right = vec3.normalize(vec3.cross(fwd, [0, 1, 0]));
  const up = vec3.cross(right, fwd);
  const t = Math.tan(fov / 2);
  const dir = vec3.normalize(vec3.add(fwd, vec3.add(
    vec3.scale(right, nx * t * aspect),
    vec3.scale(up, ny * t),
  )));
  return { eye, dir };
}

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  state.dragMode = (state.params.ikEnabled && (e.shiftKey || e.button === 2)) ? 'ik' : 'orbit';
});
canvas.addEventListener('pointerup', () => { state.dragMode = null; });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointermove', (e) => {
  if (!state.dragMode) return;
  if (state.dragMode === 'orbit') {
    state.camera.theta -= e.movementX * 0.008;
    state.camera.phi = Math.min(2.9, Math.max(0.25, state.camera.phi - e.movementY * 0.008));
  } else {
    // IK：射线与“过当前目标、面向相机”的平面求交，拖动目标点
    const { eye, dir } = rayFromMouse(e.clientX, e.clientY);
    const cam = state.camera;
    const h = state.geometry ? state.geometry.height : 4;
    const center = [0, h * 0.45, 0];
    const normal = vec3.normalize(vec3.sub([
      cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta),
      cam.dist * Math.cos(cam.phi) + h * 0.4,
      cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta),
    ], center));
    const tgt = state.params.ikTarget;
    const denom = vec3.dot(dir, normal);
    if (Math.abs(denom) > 1e-5) {
      const t = vec3.dot(vec3.sub(tgt, eye), normal) / denom;
      if (t > 0) {
        const hit = vec3.add(eye, vec3.scale(dir, t));
        const maxReach = state.boneCount * state.segmentLength * 0.999;
        const len = vec3.length(hit);
        state.params.ikTarget = len > maxReach ? vec3.scale(vec3.normalize(hit), maxReach) : hit;
        pushParams();
      }
    }
  }
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  state.camera.dist = Math.min(20, Math.max(2.5, state.camera.dist + e.deltaY * 0.01));
}, { passive: false });

// ---------------------------------------------------------------------------
// UI 控件
// ---------------------------------------------------------------------------
function bindUI() {
  const boneSlider = $('bones');
  boneSlider.addEventListener('input', () => {
    $('bones-label').textContent = boneSlider.value;
  });
  boneSlider.addEventListener('change', () => {
    rebuild(parseInt(boneSlider.value, 10));
    saveSettings();
  });

  for (const id of ['clipA', 'clipB']) {
    $(id).addEventListener('change', () => {
      state.params[id] = $(id).value;
      pushParams();
      saveSettings();
    });
  }
  $('blend').addEventListener('input', () => {
    state.params.blend = parseFloat($('blend').value);
    $('blend-label').textContent = state.params.blend.toFixed(2);
    pushParams();
  });
  $('speed').addEventListener('input', () => {
    state.params.speed = parseFloat($('speed').value);
    $('speed-label').textContent = state.params.speed.toFixed(2);
    pushParams();
  });
  $('ik').addEventListener('change', () => {
    state.params.ikEnabled = $('ik').checked;
    $('ikweight-row').style.display = state.params.ikEnabled ? '' : 'none';
    pushParams();
    saveSettings();
  });
  $('ikweight').addEventListener('input', () => {
    state.params.ikWeight = parseFloat($('ikweight').value);
    $('ikweight-label').textContent = state.params.ikWeight.toFixed(2);
    pushParams();
  });
  $('showbones').addEventListener('change', () => {
    state.showBones = $('showbones').checked;
  });
}

// ---------------------------------------------------------------------------
// IndexedDB：UI 设置持久化
// ---------------------------------------------------------------------------
let settingsDB = null;
async function saveSettings() {
  try {
    if (!settingsDB) settingsDB = await openDB();
    await dbPut(settingsDB, 'settings', {
      key: 'ui',
      boneCount: state.boneCount,
      clipA: state.params.clipA,
      clipB: state.params.clipB,
      blend: state.params.blend,
      speed: state.params.speed,
      ikEnabled: state.params.ikEnabled,
    });
  } catch (_) { /* 私密模式等场景下静默失败 */ }
}

async function loadSettings() {
  try {
    settingsDB = await openDB();
    const s = await dbGet(settingsDB, 'settings', 'ui');
    if (s) {
      state.boneCount = s.boneCount;
      state.params.clipA = s.clipA;
      state.params.clipB = s.clipB;
      state.params.blend = s.blend;
      state.params.speed = s.speed;
      state.params.ikEnabled = s.ikEnabled;
      $('bones').value = s.boneCount;
      $('bones-label').textContent = String(s.boneCount);
      $('clipA').value = s.clipA;
      $('clipB').value = s.clipB;
      $('blend').value = s.blend;
      $('blend-label').textContent = s.blend.toFixed(2);
      $('speed').value = s.speed;
      $('speed-label').textContent = s.speed.toFixed(2);
      $('ik').checked = s.ikEnabled;
      $('ikweight-row').style.display = s.ikEnabled ? '' : 'none';
    }
  } catch (_) { /* 无 IndexedDB 时使用默认值 */ }
}

// ---------------------------------------------------------------------------
// 启动
// ---------------------------------------------------------------------------
(async function start() {
  bindUI();
  await loadSettings();
  worker.postMessage({ type: 'init', boneCount: state.boneCount });
  pushParams();
  requestAnimationFrame((t) => { lastT = t; requestAnimationFrame(loop); });
})();
