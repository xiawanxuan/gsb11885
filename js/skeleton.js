// 骨骼系统：程序化骨骼生成、关键帧动画、采样插值、动画混合、CCD 逆运动学。
import { vec3, quat, mat4 } from './math.js';

export const SEGMENT_LENGTH = 0.35;

// ---------------------------------------------------------------------------
// 骨骼生成：沿 +Y 的链式骨骼（脊椎/触手状），bone0 为根。
// ---------------------------------------------------------------------------
export function createSkeleton(boneCount) {
  const bones = [];
  for (let i = 0; i < boneCount; i++) {
    bones.push({
      name: `bone_${i}`,
      parent: i - 1,
      restPos: i === 0 ? [0, 0, 0] : [0, SEGMENT_LENGTH, 0],
      restRot: [0, 0, 0, 1],
    });
  }
  const restWorld = computeWorldMatrices(bones, restPose(boneCount));
  const inverseBind = restWorld.map((m) => mat4.invert(m));
  return { bones, inverseBind, boneCount };
}

export function restPose(boneCount) {
  const pos = new Float32Array(boneCount * 3);
  const rot = new Float32Array(boneCount * 4);
  for (let i = 0; i < boneCount; i++) {
    pos[i * 3 + 1] = i === 0 ? 0 : SEGMENT_LENGTH;
    rot[i * 4 + 3] = 1;
  }
  return { pos, rot };
}

// 由局部位姿（pos/rot 扁平数组）计算世界矩阵
export function computeWorldMatrices(bones, pose) {
  const world = new Array(bones.length);
  for (let i = 0; i < bones.length; i++) {
    const local = mat4.compose(
      [pose.pos[i * 3], pose.pos[i * 3 + 1], pose.pos[i * 3 + 2]],
      [pose.rot[i * 4], pose.rot[i * 4 + 1], pose.rot[i * 4 + 2], pose.rot[i * 4 + 3]],
    );
    const p = bones[i].parent;
    world[i] = p < 0 ? local : mat4.multiply(world[p], local);
  }
  return world;
}

// 蒙皮矩阵 = world * inverseBind，输出扁平 Float32Array（可直接传 uniform）
export function computeSkinMatrices(world, inverseBind) {
  const out = new Float32Array(world.length * 16);
  for (let i = 0; i < world.length; i++) {
    out.set(mat4.multiply(world[i], inverseBind[i]), i * 16);
  }
  return out;
}

// ---------------------------------------------------------------------------
// 关键帧动画剪辑（程序化生成）
// 每条轨道：times: Float32Array(K)，rot: Float32Array(K*4)，pos: Float32Array(K*3)
// ---------------------------------------------------------------------------
export function createClip(name, boneCount) {
  const duration = 2.0;
  const keys = 17; // 每 0.125s 一个关键帧，采样正弦曲线
  const tracks = [];
  for (let b = 0; b < boneCount; b++) {
    const times = new Float32Array(keys);
    const rot = new Float32Array(keys * 4);
    const pos = new Float32Array(keys * 3);
    for (let k = 0; k < keys; k++) {
      const t = (k / (keys - 1)) * duration;
      times[k] = t;
      const phase = (t / duration) * Math.PI * 2;
      let q = [0, 0, 0, 1];
      let p = [0, b === 0 ? 0 : SEGMENT_LENGTH, 0];
      if (name === 'wave') {
        // 波动：逐骨骼相位偏移的 Z 轴摆动
        const amp = 0.35 * Math.min(1, b / 3 + 0.25);
        q = quat.fromAxisAngle([0, 0, 1], amp * Math.sin(phase * 2 - b * 0.55));
      } else if (name === 'twist') {
        // 扭转：绕自身 Y 轴渐进扭转 + 轻微弯曲
        q = quat.fromAxisAngle([0, 1, 0], 0.5 * Math.sin(phase) * (b / boneCount));
        const bend = quat.fromAxisAngle([1, 0, 0], 0.12 * Math.sin(phase * 2 - b * 0.3));
        q = quat.mul(bend, q);
      } else {
        // idle：缓慢呼吸摇摆 + 根部上下浮动
        q = quat.fromAxisAngle([1, 0, 0], 0.06 * Math.sin(phase - b * 0.2));
        if (b === 0) p = [0, 0.05 * Math.sin(phase * 2), 0];
      }
      rot.set(q, k * 4);
      pos.set(p, k * 3);
    }
    tracks.push({ times, rot, pos });
  }
  return { name, duration, tracks, boneCount };
}

// 剪辑序列化（IndexedDB 存储用，TypedArray 可被结构化克隆，直接返回即可）
export function clipToStorable(clip) { return clip; }

// 采样单骨骼在 time 处的局部位姿（循环播放；旋转 slerp，位移 lerp）
export function sampleTrack(track, duration, time, outRot, outPos, offset) {
  const t = ((time % duration) + duration) % duration;
  const times = track.times;
  const n = times.length;
  let hi = 1;
  while (hi < n - 1 && times[hi] < t) hi++;
  const lo = hi - 1;
  const span = times[hi] - times[lo] || 1;
  const f = Math.min(1, Math.max(0, (t - times[lo]) / span));

  const rA = [track.rot[lo * 4], track.rot[lo * 4 + 1], track.rot[lo * 4 + 2], track.rot[lo * 4 + 3]];
  const rB = [track.rot[hi * 4], track.rot[hi * 4 + 1], track.rot[hi * 4 + 2], track.rot[hi * 4 + 3]];
  const r = quat.slerp(rA, rB, f);
  outRot[offset * 4] = r[0]; outRot[offset * 4 + 1] = r[1];
  outRot[offset * 4 + 2] = r[2]; outRot[offset * 4 + 3] = r[3];

  for (let c = 0; c < 3; c++) {
    outPos[offset * 3 + c] =
      track.pos[lo * 3 + c] + (track.pos[hi * 3 + c] - track.pos[lo * 3 + c]) * f;
  }
}

// 采样整条剪辑 → pose {pos, rot}
export function sampleClip(clip, time, pose) {
  for (let b = 0; b < clip.boneCount; b++) {
    sampleTrack(clip.tracks[b], clip.duration, time, pose.rot, pose.pos, b);
  }
  return pose;
}

// 动画混合：poseOut = mix(poseA, poseB, w)（旋转 slerp / 位移 lerp）
export function blendPose(poseA, poseB, w, out) {
  const n = poseA.rot.length / 4;
  for (let i = 0; i < n; i++) {
    const a = [poseA.rot[i * 4], poseA.rot[i * 4 + 1], poseA.rot[i * 4 + 2], poseA.rot[i * 4 + 3]];
    const b = [poseB.rot[i * 4], poseB.rot[i * 4 + 1], poseB.rot[i * 4 + 2], poseB.rot[i * 4 + 3]];
    const r = quat.slerp(a, b, w);
    out.rot.set(r, i * 4);
    for (let c = 0; c < 3; c++) {
      out.pos[i * 3 + c] = poseA.pos[i * 3 + c] + (poseB.pos[i * 3 + c] - poseA.pos[i * 3 + c]) * w;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 逆运动学：CCD（Cyclic Coordinate Descent）求解器
// 在动画位姿基础上迭代调整链上各骨骼的局部旋转，使末端效应器逼近目标点。
// ---------------------------------------------------------------------------
export function solveCCD(bones, pose, target, tipIndex, iterations = 12, tolerance = 1e-3) {
  const count = bones.length;
  for (let iter = 0; iter < iterations; iter++) {
    const world = computeWorldMatrices(bones, pose);
    const tipPos = mat4.transformPoint(world[tipIndex], [0, SEGMENT_LENGTH, 0]);
    const toTarget = vec3.sub(target, tipPos);
    if (vec3.length(toTarget) < tolerance) break;

    for (let j = tipIndex; j >= 0; j--) {
      const jointPos = mat4.transformPoint(world[j], [0, 0, 0]);
      const vEnd = vec3.normalize(vec3.sub(tipPos, jointPos));
      const vGoal = vec3.normalize(vec3.sub(target, jointPos));
      if (vec3.length(vEnd) < 1e-8 || vec3.length(vGoal) < 1e-8) continue;

      const delta = quat.fromTo(vEnd, vGoal); // 世界空间旋转增量
      const parent = bones[j].parent;
      const parentWorldRot = parent < 0 ? [0, 0, 0, 1] : mat4.toQuat(world[parent]);
      const worldRot = mat4.toQuat(world[j]);
      // newWorldRot = delta * worldRot；换算回局部：local = parentWorldRot⁻¹ * newWorldRot
      const newLocal = quat.normalize(
        quat.mul(quat.conjugate(parentWorldRot), quat.mul(delta, worldRot)),
      );
      pose.rot.set(newLocal, j * 4);

      // 前向更新该骨骼及其后代的世界矩阵，供下一轮使用
      const local = mat4.compose(
        [pose.pos[j * 3], pose.pos[j * 3 + 1], pose.pos[j * 3 + 2]],
        newLocal,
      );
      world[j] = parent < 0 ? local : mat4.multiply(world[parent], local);
      for (let k = j + 1; k < count; k++) {
        const lk = mat4.compose(
          [pose.pos[k * 3], pose.pos[k * 3 + 1], pose.pos[k * 3 + 2]],
          [pose.rot[k * 4], pose.rot[k * 4 + 1], pose.rot[k * 4 + 2], pose.rot[k * 4 + 3]],
        );
        world[k] = mat4.multiply(world[bones[k].parent], lk);
      }
      const newTip = mat4.transformPoint(world[tipIndex], [0, SEGMENT_LENGTH, 0]);
      tipPos[0] = newTip[0]; tipPos[1] = newTip[1]; tipPos[2] = newTip[2];
    }
  }
  return pose;
}
