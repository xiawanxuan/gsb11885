// Worker 全流程冒烟测试：init → tick → frame，含降级路径。
// 运行：node test/worker-smoke.mjs
import assert from 'node:assert/strict';

const messages = [];
globalThis.postMessage = (msg) => messages.push(msg);
globalThis.onmessage = null; // worker 模块顶层会赋值
// indexedDB 不存在 → worker 内 try/catch 回退到程序化剪辑

await import('../js/worker.js');
const send = (data) => globalThis.onmessage({ data });

// 1) 正常路径：32 骨骼 GPU 蒙皮
send({ type: 'init', boneCount: 32 });
await new Promise((r) => setTimeout(r, 50));
const geo = messages.find((m) => m.type === 'geometry');
assert.ok(geo, '应收到 geometry');
assert.equal(geo.degraded, false);
assert.equal(geo.boneCount, 32);
assert.ok(geo.positions.length > 0 && geo.indices.length > 0);
assert.equal(geo.parents.length, 32);
assert.ok(geo.height > 0, 'geometry 应包含 height');

messages.length = 0;
send({ type: 'params', params: { clipA: 'wave', clipB: 'twist', blend: 0.3, speed: 1 } });
send({ type: 'tick', dt: 1 / 60 });
const frame = messages.find((m) => m.type === 'frame');
assert.ok(frame, '应收到 frame');
assert.ok(frame.skinMatrices instanceof Float32Array);
assert.equal(frame.skinMatrices.length, 32 * 16);
assert.equal(frame.jointPositions.length, 33 * 3);
assert.equal(frame.cpuPositions, undefined);
assert.ok(frame.simTime >= 0);
// 蒙皮矩阵应为有效仿射矩阵（末行 0,0,0,1）
for (let b = 0; b < 32; b++) {
  assert.ok(Math.abs(frame.skinMatrices[b * 16 + 15] - 1) < 1e-4);
}

// 2) IK：开启后末端应朝目标移动
messages.length = 0;
send({ type: 'params', params: { ikEnabled: true, ikTarget: [1.5, 1.0, 0.5], ikWeight: 1 } });
for (let i = 0; i < 30; i++) send({ type: 'tick', dt: 1 / 60 });
const ikFrame = messages.filter((m) => m.type === 'frame').pop();
const tip = [
  ikFrame.jointPositions[32 * 3],
  ikFrame.jointPositions[32 * 3 + 1],
  ikFrame.jointPositions[32 * 3 + 2],
];
const dist = Math.hypot(tip[0] - 1.5, tip[1] - 1.0, tip[2] - 0.5);
assert.ok(dist < 0.15, `IK 末端距目标 ${dist.toFixed(3)}，应 < 0.15`);

// 3) 降级路径：100 骨骼 > 64 上限 → 刚性绑定 + CPU 蒙皮
messages.length = 0;
send({ type: 'rebuild', boneCount: 100 });
await new Promise((r) => setTimeout(r, 50));
const geo2 = messages.find((m) => m.type === 'geometry');
assert.equal(geo2.degraded, true);
assert.equal(geo2.maxGpuBones, 64);
// 刚性绑定：每顶点权重 [1,0,0,0]
for (let v = 0; v < geo2.weights.length; v += 4) {
  assert.equal(geo2.weights[v], 1);
  assert.equal(geo2.weights[v + 1] + geo2.weights[v + 2] + geo2.weights[v + 3], 0);
}

messages.length = 0;
send({ type: 'params', params: { ikEnabled: false } });
send({ type: 'tick', dt: 1 / 60 });
const frame2 = messages.find((m) => m.type === 'frame');
assert.ok(frame2.cpuPositions instanceof Float32Array, '降级时应回传 CPU 蒙皮顶点');
assert.equal(frame2.skinMatrices, undefined);
assert.equal(frame2.cpuPositions.length, geo2.positions.length);
// CPU 蒙皮结果应为有限数
for (let i = 0; i < frame2.cpuPositions.length; i += 997) {
  assert.ok(Number.isFinite(frame2.cpuPositions[i]));
}

console.log('Worker 冒烟测试全部通过（正常路径 + IK + 降级路径）');
