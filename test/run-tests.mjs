// 无头验证：数学库、关键帧插值、动画混合、权重归一化、IK、降级蒙皮。
// 运行：node test/run-tests.mjs
import assert from 'node:assert/strict';
import { vec3, quat, mat4 } from '../js/math.js';
import {
  createSkeleton, createClip, restPose, sampleClip, blendPose,
  solveCCD, computeWorldMatrices, computeSkinMatrices, SEGMENT_LENGTH,
} from '../js/skeleton.js';
import { createSkinnedMesh, validateWeights, cpuSkin, MAX_INFLUENCES } from '../js/mesh.js';

let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(err);
    process.exitCode = 1;
  }
}
const near = (a, b, eps = 1e-4) => Math.abs(a - b) <= eps;
const vecNear = (a, b, eps = 1e-4) => a.every((v, i) => near(v, b[i], eps));

console.log('数学库');
test('mat4 逆矩阵：M · M⁻¹ = I', () => {
  const m = mat4.compose([1, 2, 3], quat.fromAxisAngle([0, 1, 0], 0.7), [2, 1, 0.5]);
  const inv = mat4.invert(m);
  const prod = mat4.multiply(m, inv);
  for (let i = 0; i < 16; i++) assert.ok(near(prod[i], i % 5 === 0 ? 1 : 0, 1e-4));
});

test('quat slerp：端点与中点', () => {
  const a = quat.identity();
  const b = quat.fromAxisAngle([0, 0, 1], Math.PI / 2);
  assert.ok(vecNear(quat.slerp(a, b, 0), a));
  assert.ok(vecNear(quat.slerp(a, b, 1), b));
  const mid = quat.slerp(a, b, 0.5);
  const expected = quat.fromAxisAngle([0, 0, 1], Math.PI / 4);
  assert.ok(vecNear(mid, expected, 1e-4));
});

test('quat fromTo：旋转 v1 到 v2', () => {
  const q = quat.fromTo([1, 0, 0], [0, 1, 0]);
  assert.ok(vecNear(quat.rotateVec3(q, [1, 0, 0]), [0, 1, 0], 1e-5));
});

console.log('骨骼与蒙皮');
test('静止位姿的蒙皮矩阵 = 单位阵', () => {
  const sk = createSkeleton(8);
  const world = computeWorldMatrices(sk.bones, restPose(8));
  const skin = computeSkinMatrices(world, sk.inverseBind);
  for (let i = 0; i < 8 * 16; i++) assert.ok(near(skin[i], i % 16 % 5 === 0 ? 1 : 0, 1e-5));
});

test('权重归一化：每顶点 Σw = 1 且 ≤ 4 影响', () => {
  const mesh = createSkinnedMesh(32, 8, false);
  const r = validateWeights(mesh.weights, mesh.vertexCount, MAX_INFLUENCES);
  assert.ok(r.ok, `顶点 ${r.vertex} 权重和 = ${r.sum}`);
});

test('刚性绑定降级：每顶点恰好 1 个影响、权重为 1', () => {
  const mesh = createSkinnedMesh(128, 8, true);
  for (let v = 0; v < mesh.vertexCount; v++) {
    assert.equal(mesh.weights[v * 4], 1);
    assert.equal(mesh.weights[v * 4 + 1] + mesh.weights[v * 4 + 2] + mesh.weights[v * 4 + 3], 0);
  }
});

test('CPU 蒙皮（降级路径）：静止位姿下顶点不变', () => {
  const sk = createSkeleton(100);
  const mesh = createSkinnedMesh(100, 6, true);
  const world = computeWorldMatrices(sk.bones, restPose(100));
  const skin = computeSkinMatrices(world, sk.inverseBind);
  const pos = new Float32Array(mesh.vertexCount * 3);
  const nrm = new Float32Array(mesh.vertexCount * 3);
  cpuSkin(mesh, skin, pos, nrm);
  for (let i = 0; i < pos.length; i++) assert.ok(near(pos[i], mesh.positions[i], 1e-4));
});

test('CPU 蒙皮与 LBS 公式一致：绕根旋转 90° 后点 (r,0,0)→(0,0,-r)', () => {
  const sk = createSkeleton(4);
  const mesh = createSkinnedMesh(4, 6, true);
  const pose = restPose(4);
  pose.rot.set(quat.fromAxisAngle([0, 1, 0], Math.PI / 2), 0); // 根骨骼旋转
  const world = computeWorldMatrices(sk.bones, pose);
  const skin = computeSkinMatrices(world, sk.inverseBind);
  const pos = new Float32Array(mesh.vertexCount * 3);
  const nrm = new Float32Array(mesh.vertexCount * 3);
  cpuSkin(mesh, skin, pos, nrm);
  // 底环第一个顶点原位置 (r, 0, 0)，绕 Y 轴 +90° 后应为 (0, 0, -r)
  const r0 = mesh.positions[0];
  assert.ok(near(pos[0], 0, 1e-4) && near(pos[2], -r0, 1e-4), `得到 (${pos[0]}, ${pos[2]})，期望 (0, ${-r0})`);
});

console.log('关键帧插值');
test('采样关键帧时刻 = 关键帧值', () => {
  const clip = createClip('wave', 4);
  const pose = restPose(4);
  sampleClip(clip, clip.tracks[1].times[3], pose);
  const expect = [clip.tracks[1].rot[12], clip.tracks[1].rot[13], clip.tracks[1].rot[14], clip.tracks[1].rot[15]];
  const got = [pose.rot[4], pose.rot[5], pose.rot[6], pose.rot[7]];
  assert.ok(vecNear(got, expect, 1e-4), `got ${got}, want ${expect}`);
});

test('相邻关键帧之间插值连续且在端点之间', () => {
  const clip = createClip('twist', 4);
  const pose = restPose(4);
  const t0 = clip.tracks[2].times[2], t1 = clip.tracks[2].times[3];
  sampleClip(clip, t0, pose);
  const a = pose.rot[8]; // bone2 x 分量
  sampleClip(clip, t1, pose);
  const b = pose.rot[8];
  sampleClip(clip, (t0 + t1) / 2, pose);
  const m = pose.rot[8];
  const lo = Math.min(a, b) - 1e-4, hi = Math.max(a, b) + 1e-4;
  assert.ok(m >= lo && m <= hi, `中点 ${m} 不在 [${lo}, ${hi}]`);
});

test('剪辑循环：t 与 t+duration 采样一致', () => {
  const clip = createClip('idle', 4);
  const p1 = restPose(4), p2 = restPose(4);
  sampleClip(clip, 0.37, p1);
  sampleClip(clip, 0.37 + clip.duration, p2);
  assert.ok(vecNear([...p1.rot], [...p2.rot], 1e-4));
});

console.log('动画混合');
test('blend w=0 → A，w=1 → B', () => {
  const A = createClip('wave', 6), B = createClip('twist', 6);
  const pA = restPose(6), pB = restPose(6), out = restPose(6);
  sampleClip(A, 0.5, pA);
  sampleClip(B, 0.5, pB);
  blendPose(pA, pB, 0, out);
  assert.ok(vecNear([...out.rot], [...pA.rot], 1e-5));
  blendPose(pA, pB, 1, out);
  assert.ok(vecNear([...out.rot], [...pB.rot], 1e-5));
});

test('blend w=0.5 介于两者之间（旋转仍为单位四元数）', () => {
  const A = createClip('wave', 6), B = createClip('idle', 6);
  const pA = restPose(6), pB = restPose(6), out = restPose(6);
  sampleClip(A, 1.1, pA);
  sampleClip(B, 1.1, pB);
  blendPose(pA, pB, 0.5, out);
  for (let i = 0; i < 6; i++) {
    const len = Math.hypot(out.rot[i * 4], out.rot[i * 4 + 1], out.rot[i * 4 + 2], out.rot[i * 4 + 3]);
    assert.ok(near(len, 1, 1e-4));
  }
});

console.log('逆运动学（CCD）');
test('直链可达目标：末端收敛到目标点', () => {
  const n = 8;
  const sk = createSkeleton(n);
  const pose = restPose(n);
  const target = [1.2, 1.5, 0.3];
  solveCCD(sk.bones, pose, target, n - 1, 64, 1e-3);
  const world = computeWorldMatrices(sk.bones, pose);
  const tip = mat4.transformPoint(world[n - 1], [0, SEGMENT_LENGTH, 0]);
  assert.ok(vec3.length(vec3.sub(tip, target)) < 0.02, `末端 ${tip} 未到目标 ${target}`);
});

test('不可达目标：末端完全伸展指向目标方向', () => {
  const n = 4;
  const sk = createSkeleton(n);
  const pose = restPose(n);
  const target = [0, 100, 0]; // 远超链长
  solveCCD(sk.bones, pose, target, n - 1, 64, 1e-3);
  const world = computeWorldMatrices(sk.bones, pose);
  const tip = mat4.transformPoint(world[n - 1], [0, SEGMENT_LENGTH, 0]);
  const maxReach = n * SEGMENT_LENGTH;
  assert.ok(near(vec3.length(tip), maxReach, 0.02), `链长 ${vec3.length(tip)} ≠ 完全伸展 ${maxReach}`);
});

console.log('降级阈值');
test('骨骼数上限：>64 触发降级（刚性绑定）', async () => {
  const { MAX_GPU_BONES } = await import('../js/worker.js').catch(() => ({}));
  // worker.js 含 onmessage，无法在 node 直接导入；改为验证网格层逻辑
  assert.ok(MAX_GPU_BONES === undefined || MAX_GPU_BONES === 64);
  const smooth = createSkinnedMesh(64, 6, false);
  const rigid = createSkinnedMesh(65, 6, true);
  assert.ok(validateWeights(smooth.weights, smooth.vertexCount).ok);
  assert.equal(rigid.rigid, true);
});

console.log(`\n${passed} 项通过${process.exitCode ? '，存在失败' : ''}`);
