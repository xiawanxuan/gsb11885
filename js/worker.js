// 动画 Worker：承担全部重计算 —— 关键帧采样、动画混合、IK 求解、
// 蒙皮矩阵计算，以及超上限时的 CPU 刚性蒙皮降级。
// 每帧通过 Transferable TypedArray 回传结果，主线程只做渲染。
import { createSkeleton, createClip, restPose, sampleClip, blendPose, solveCCD, computeWorldMatrices, computeSkinMatrices, SEGMENT_LENGTH } from './skeleton.js';
import { createSkinnedMesh, cpuSkin } from './mesh.js';
import { openDB, dbGet, dbPut, dbGetAll } from './db.js';

export const MAX_GPU_BONES = 64; // 骨骼数上限：超出则降级

const state = {
  skeleton: null,
  mesh: null,
  clips: {},          // name -> clip
  params: {
    clipA: 'wave',
    clipB: 'twist',
    blend: 0.0,       // A/B 混合因子
    speed: 1.0,
    ikEnabled: false,
    ikWeight: 1.0,
    ikTarget: [1.2, 2.0, 0.5],
  },
  time: 0,
  degraded: false,    // 是否处于降级模式（刚性绑定 + CPU 蒙皮）
  db: null,
  poseA: null,
  poseB: null,
  poseIK: null,
};

function allocatePoses(boneCount) {
  state.poseA = restPose(boneCount);
  state.poseB = restPose(boneCount);
  state.poseIK = restPose(boneCount);
}

async function init(boneCount) {
  state.skeleton = createSkeleton(boneCount);
  state.degraded = boneCount > MAX_GPU_BONES;
  // 降级：刚性绑定网格（每顶点 1 骨骼）；正常：4 影响平滑蒙皮
  state.mesh = createSkinnedMesh(boneCount, state.degraded ? 10 : 14, state.degraded);
  allocatePoses(boneCount);

  // 剪辑：优先从 IndexedDB 恢复，否则程序化生成并持久化
  let restored = false;
  try {
    if (!state.db) state.db = await openDB();
    const stored = await dbGetAll(state.db, 'clips');
    const usable = stored && stored.length && stored.every((c) => c.boneCount === boneCount);
    if (usable) {
      for (const c of stored) state.clips[c.name] = c;
      restored = true;
    } else {
      for (const name of ['wave', 'twist', 'idle']) {
        const clip = createClip(name, boneCount);
        state.clips[name] = clip;
        await dbPut(state.db, 'clips', clip);
      }
    }
  } catch (err) {
    for (const name of ['wave', 'twist', 'idle']) state.clips[name] = createClip(name, boneCount);
  }

  const m = state.mesh;
  postMessage({
    type: 'geometry',
    positions: m.positions,
    normals: m.normals,
    joints: m.joints,
    weights: m.weights,
    indices: m.indices,
    boneCount,
    parents: state.skeleton.bones.map((b) => b.parent),
    segmentLength: SEGMENT_LENGTH,
    height: m.height,
    degraded: state.degraded,
    maxGpuBones: MAX_GPU_BONES,
    clipsRestored: restored,
  }); // 几何数据仅发送一次，使用结构化克隆（Worker 降级路径仍需保留网格数据）
}

function tick(dt) {
  if (!state.skeleton) return; // init 尚未完成（IndexedDB 异步）
  const t0 = performance.now();
  const p = state.params;
  const n = state.skeleton.boneCount;
  state.time += dt * p.speed;

  // 1) 关键帧采样（slerp/lerp 插值）
  sampleClip(state.clips[p.clipA] || state.clips.wave, state.time, state.poseA);
  sampleClip(state.clips[p.clipB] || state.clips.twist, state.time, state.poseB);
  // 2) 动画混合
  blendPose(state.poseA, state.poseB, p.blend, state.poseIK);
  // 3) IK（CCD）：在动画位姿之上按 ikWeight 混合求解结果
  if (p.ikEnabled) {
    const preRot = state.poseIK.rot.slice();
    solveCCD(state.skeleton.bones, state.poseIK, p.ikTarget, n - 1, 14, 1e-3);
    if (p.ikWeight < 1) {
      for (let i = 0; i < n; i++) {
        const a = [preRot[i * 4], preRot[i * 4 + 1], preRot[i * 4 + 2], preRot[i * 4 + 3]];
        const b = [state.poseIK.rot[i * 4], state.poseIK.rot[i * 4 + 1], state.poseIK.rot[i * 4 + 2], state.poseIK.rot[i * 4 + 3]];
        // 简单 nlerp 混合 IK 强度
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        const s = dot < 0 ? -1 : 1;
        const w = p.ikWeight;
        const r = [
          a[0] + (b[0] * s - a[0]) * w,
          a[1] + (b[1] * s - a[1]) * w,
          a[2] + (b[2] * s - a[2]) * w,
          a[3] + (b[3] * s - a[3]) * w,
        ];
        const len = Math.hypot(r[0], r[1], r[2], r[3]) || 1;
        state.poseIK.rot.set([r[0] / len, r[1] / len, r[2] / len, r[3] / len], i * 4);
      }
    }
  }

  // 4) 世界矩阵 + 关节位置（调试骨骼线用）
  const world = computeWorldMatrices(state.skeleton.bones, state.poseIK);
  const jointPositions = new Float32Array((n + 1) * 3);
  for (let i = 0; i < n; i++) {
    jointPositions[i * 3] = world[i][12];
    jointPositions[i * 3 + 1] = world[i][13];
    jointPositions[i * 3 + 2] = world[i][14];
  }
  const tip = [world[n - 1][12], world[n - 1][13], world[n - 1][14]];
  const tipLocal = [0, SEGMENT_LENGTH, 0];
  jointPositions[n * 3] = tip[0] + world[n - 1][4] * tipLocal[1];
  jointPositions[n * 3 + 1] = tip[1] + world[n - 1][5] * tipLocal[1];
  jointPositions[n * 3 + 2] = tip[2] + world[n - 1][6] * tipLocal[1];

  const msg = { type: 'frame', jointPositions, simTime: 0, degraded: state.degraded };
  const transfers = [jointPositions.buffer];

  if (state.degraded) {
    // 降级路径：CPU 刚性蒙皮（Worker 内计算，主线程不卡）
    const skin = computeSkinMatrices(world, state.skeleton.inverseBind);
    const count = state.mesh.vertexCount;
    const pos = new Float32Array(count * 3);
    const nrm = new Float32Array(count * 3);
    cpuSkin(state.mesh, skin, pos, nrm);
    msg.cpuPositions = pos;
    msg.cpuNormals = nrm;
    transfers.push(pos.buffer, nrm.buffer);
  } else {
    // 正常路径：蒙皮矩阵 → GPU 顶点着色器 LBS
    msg.skinMatrices = computeSkinMatrices(world, state.skeleton.inverseBind);
    transfers.push(msg.skinMatrices.buffer);
  }

  msg.simTime = performance.now() - t0;
  postMessage(msg, transfers);
}

onmessage = (e) => {
  const d = e.data;
  if (d.type === 'init') {
    init(d.boneCount).then(() => postMessage({ type: 'ready' }));
  } else if (d.type === 'params') {
    Object.assign(state.params, d.params);
  } else if (d.type === 'tick') {
    tick(d.dt);
  } else if (d.type === 'rebuild') {
    init(d.boneCount).then(() => postMessage({ type: 'ready' }));
  }
};
