// 程序化蒙皮网格：围绕骨骼链的锥形管状网格。
// 每顶点最多 4 个骨骼影响，权重归一化（Σw = 1）；支持降级为刚性绑定（1 影响）。
import { SEGMENT_LENGTH } from './skeleton.js';

export const MAX_INFLUENCES = 4;

// 单顶点权重归一化：截断到 maxInfluences 个最大权重并归一化到和为 1。
export function normalizeWeightSet(entries, maxInfluences = MAX_INFLUENCES) {
  entries.sort((a, b) => b.weight - a.weight);
  const kept = entries.filter((e) => e.weight > 1e-6).slice(0, maxInfluences);
  let sum = 0;
  for (const e of kept) sum += e.weight;
  if (sum < 1e-12) return [{ joint: 0, weight: 1 }];
  for (const e of kept) e.weight /= sum;
  return kept;
}

// 校验整个网格权重是否归一化（测试与调试使用）
export function validateWeights(weights, vertexCount, influences = MAX_INFLUENCES, eps = 1e-4) {
  for (let v = 0; v < vertexCount; v++) {
    let sum = 0;
    for (let k = 0; k < influences; k++) sum += weights[v * influences + k];
    if (Math.abs(sum - 1) > eps) return { ok: false, vertex: v, sum };
  }
  return { ok: true };
}

// 半径轮廓：底部粗、顶端细的“触手”形
function radiusAt(t) {
  return 0.34 * (1 - t * 0.72) + 0.05 * Math.sin(t * Math.PI);
}

/**
 * 生成蒙皮网格。
 * @param boneCount 骨骼数
 * @param radialSegments 圆周分段
 * @param rigid 为 true 时生成刚性绑定（每顶点 1 个骨骼、权重 1）
 */
export function createSkinnedMesh(boneCount, radialSegments = 14, rigid = false) {
  const ringCount = boneCount + 1; // 每个关节一个环 + 顶端封口环
  const positions = [];
  const normals = [];
  const joints = [];
  const weights = [];
  const indices = [];

  const height = boneCount * SEGMENT_LENGTH;

  for (let r = 0; r < ringCount; r++) {
    const t = r / boneCount;
    const y = r * SEGMENT_LENGTH;
    const radius = r === ringCount - 1 ? 0.02 : radiusAt(t);
    for (let s = 0; s < radialSegments; s++) {
      const a = (s / radialSegments) * Math.PI * 2;
      const nx = Math.cos(a), nz = Math.sin(a);
      positions.push(nx * radius, y, nz * radius);
      normals.push(nx, 0, nz);

      if (rigid) {
        // 刚性绑定：绑定到最近骨骼
        const j = Math.min(boneCount - 1, r);
        joints.push(j, 0, 0, 0);
        weights.push(1, 0, 0, 0);
      } else {
        // 平滑蒙皮：到相邻骨骼的距离衰减权重（最多 4 个影响）
        const entries = [];
        for (let b = Math.max(0, r - 2); b <= Math.min(boneCount - 1, r + 1); b++) {
          const d = Math.abs(y - b * SEGMENT_LENGTH) / SEGMENT_LENGTH;
          const w = Math.max(0, 1 - d) ** 2;
          if (w > 0) entries.push({ joint: b, weight: w });
        }
        if (entries.length === 0) entries.push({ joint: Math.min(boneCount - 1, r), weight: 1 });
        const norm = normalizeWeightSet(entries);
        const j = [0, 0, 0, 0];
        const w = [0, 0, 0, 0];
        norm.forEach((e, i) => { j[i] = e.joint; w[i] = e.weight; });
        joints.push(...j);
        weights.push(...w);
      }
    }
  }

  // 侧面索引
  for (let r = 0; r < ringCount - 1; r++) {
    for (let s = 0; s < radialSegments; s++) {
      const a = r * radialSegments + s;
      const b = r * radialSegments + ((s + 1) % radialSegments);
      const c = (r + 1) * radialSegments + s;
      const d = (r + 1) * radialSegments + ((s + 1) % radialSegments);
      indices.push(a, c, b, b, c, d);
    }
  }

  // 底盖（扇形，绑定 bone0）
  const baseCenter = positions.length / 3;
  positions.push(0, 0, 0);
  normals.push(0, -1, 0);
  joints.push(0, 0, 0, 0);
  weights.push(1, 0, 0, 0);
  for (let s = 0; s < radialSegments; s++) {
    indices.push(baseCenter, (s + 1) % radialSegments, s);
  }

  const vertexCount = positions.length / 3;
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    joints: new Float32Array(joints),
    weights: new Float32Array(weights),
    indices: new Uint16Array(indices),
    vertexCount,
    height,
    rigid,
  };
}

// CPU 刚性蒙皮（降级路径）：v' = Σ wk * Mk * v，刚性时只有 1 个影响。
// 直接在 Worker 中执行，结果通过 Transferable 传回主线程，主线程零计算。
export function cpuSkin(mesh, skinMatrices, outPositions, outNormals) {
  const pos = mesh.positions;
  const nrm = mesh.normals;
  const joints = mesh.joints;
  const weights = mesh.weights;
  const n = mesh.vertexCount;
  for (let v = 0; v < n; v++) {
    let px = 0, py = 0, pz = 0, nx = 0, ny = 0, nz = 0;
    for (let k = 0; k < MAX_INFLUENCES; k++) {
      const w = weights[v * MAX_INFLUENCES + k];
      if (w === 0) continue;
      const j = joints[v * MAX_INFLUENCES + k] | 0;
      const m = j * 16;
      const x = pos[v * 3], y = pos[v * 3 + 1], z = pos[v * 3 + 2];
      px += w * (skinMatrices[m] * x + skinMatrices[m + 4] * y + skinMatrices[m + 8] * z + skinMatrices[m + 12]);
      py += w * (skinMatrices[m + 1] * x + skinMatrices[m + 5] * y + skinMatrices[m + 9] * z + skinMatrices[m + 13]);
      pz += w * (skinMatrices[m + 2] * x + skinMatrices[m + 6] * y + skinMatrices[m + 10] * z + skinMatrices[m + 14]);
      const dx = nrm[v * 3], dy = nrm[v * 3 + 1], dz = nrm[v * 3 + 2];
      nx += w * (skinMatrices[m] * dx + skinMatrices[m + 4] * dy + skinMatrices[m + 8] * dz);
      ny += w * (skinMatrices[m + 1] * dx + skinMatrices[m + 5] * dy + skinMatrices[m + 9] * dz);
      nz += w * (skinMatrices[m + 2] * dx + skinMatrices[m + 6] * dy + skinMatrices[m + 10] * dz);
    }
    outPositions[v * 3] = px; outPositions[v * 3 + 1] = py; outPositions[v * 3 + 2] = pz;
    const len = Math.hypot(nx, ny, nz) || 1;
    outNormals[v * 3] = nx / len; outNormals[v * 3 + 1] = ny / len; outNormals[v * 3 + 2] = nz / len;
  }
}
