// 数学库：vec3 / quat / mat4（列主序，与 WebGL uniformMatrix4fv 兼容）
// 纯 ES Module，可同时被主线程、Web Worker 和 Node 测试复用。

export const vec3 = {
  create(x = 0, y = 0, z = 0) { return [x, y, z]; },
  add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; },
  sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; },
  scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; },
  dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; },
  cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  },
  length(a) { return Math.hypot(a[0], a[1], a[2]); },
  normalize(a) {
    const len = Math.hypot(a[0], a[1], a[2]);
    if (len < 1e-12) return [0, 0, 0];
    return [a[0] / len, a[1] / len, a[2] / len];
  },
  lerp(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  },
};

export const quat = {
  identity() { return [0, 0, 0, 1]; },

  fromAxisAngle(axis, angle) {
    const n = vec3.normalize(axis);
    const half = angle * 0.5;
    const s = Math.sin(half);
    return [n[0] * s, n[1] * s, n[2] * s, Math.cos(half)];
  },

  mul(a, b) {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ];
  },

  conjugate(q) { return [-q[0], -q[1], -q[2], q[3]]; },

  normalize(q) {
    const len = Math.hypot(q[0], q[1], q[2], q[3]);
    if (len < 1e-12) return [0, 0, 0, 1];
    return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
  },

  // 球面线性插值，自动处理最短路径（点积为负时取反）
  slerp(a, b, t) {
    let [bx, by, bz, bw] = b;
    let cosHalf = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw;
    if (cosHalf < 0) {
      cosHalf = -cosHalf;
      bx = -bx; by = -by; bz = -bz; bw = -bw;
    }
    if (cosHalf > 0.9995) {
      return quat.normalize([
        a[0] + (bx - a[0]) * t,
        a[1] + (by - a[1]) * t,
        a[2] + (bz - a[2]) * t,
        a[3] + (bw - a[3]) * t,
      ]);
    }
    const half = Math.acos(cosHalf);
    const sinHalf = Math.sin(half);
    const ra = Math.sin((1 - t) * half) / sinHalf;
    const rb = Math.sin(t * half) / sinHalf;
    return [
      a[0] * ra + bx * rb,
      a[1] * ra + by * rb,
      a[2] * ra + bz * rb,
      a[3] * ra + bw * rb,
    ];
  },

  // 两个单位向量之间的最短旋转
  fromTo(a, b) {
    const va = vec3.normalize(a);
    const vb = vec3.normalize(b);
    const d = vec3.dot(va, vb);
    if (d > 0.999999) return [0, 0, 0, 1];
    if (d < -0.999999) {
      // 反向：任选一个垂直轴旋转 180°
      let axis = vec3.cross([1, 0, 0], va);
      if (vec3.length(axis) < 1e-6) axis = vec3.cross([0, 1, 0], va);
      axis = vec3.normalize(axis);
      return [axis[0], axis[1], axis[2], 0];
    }
    const c = vec3.cross(va, vb);
    const w = Math.sqrt((1 + d) * 2);
    const inv = 1 / w;
    return [c[0] * inv, c[1] * inv, c[2] * inv, w * 0.5];
  },

  rotateVec3(q, v) {
    const [qx, qy, qz, qw] = q;
    const [x, y, z] = v;
    // t = 2 * cross(q.xyz, v)
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    return [
      x + qw * tx + (qy * tz - qz * ty),
      y + qw * ty + (qz * tx - qx * tz),
      z + qw * tz + (qx * ty - qy * tx),
    ];
  },
};

export const mat4 = {
  identity() {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
  },

  // 由平移/旋转/缩放组合（列主序）
  compose(pos, q, scale = [1, 1, 1]) {
    const [x, y, z, w] = q;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    const [sx, sy, sz] = scale;
    const m = new Float32Array(16);
    m[0] = (1 - (yy + zz)) * sx;
    m[1] = (xy + wz) * sx;
    m[2] = (xz - wy) * sx;
    m[3] = 0;
    m[4] = (xy - wz) * sy;
    m[5] = (1 - (xx + zz)) * sy;
    m[6] = (yz + wx) * sy;
    m[7] = 0;
    m[8] = (xz + wy) * sz;
    m[9] = (yz - wx) * sz;
    m[10] = (1 - (xx + yy)) * sz;
    m[11] = 0;
    m[12] = pos[0];
    m[13] = pos[1];
    m[14] = pos[2];
    m[15] = 1;
    return m;
  },

  // out = a * b（列主序，glMatrix 约定）
  multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[r] * b[c * 4] +
          a[4 + r] * b[c * 4 + 1] +
          a[8 + r] * b[c * 4 + 2] +
          a[12 + r] * b[c * 4 + 3];
      }
    }
    return out;
  },

  invert(m) {
    // 高斯-若尔当消元（列主序存储，按行主序视图处理）
    const a = Array.from({ length: 4 }, (_, r) => [
      m[r], m[4 + r], m[8 + r], m[12 + r],
      r === 0 ? 1 : 0, r === 1 ? 1 : 0, r === 2 ? 1 : 0, r === 3 ? 1 : 0,
    ]);
    for (let col = 0; col < 4; col++) {
      let pivot = col;
      for (let r = col + 1; r < 4; r++) {
        if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
      }
      if (Math.abs(a[pivot][col]) < 1e-12) return null;
      if (pivot !== col) [a[pivot], a[col]] = [a[col], a[pivot]];
      const inv = 1 / a[col][col];
      for (let c = 0; c < 8; c++) a[col][c] *= inv;
      for (let r = 0; r < 4; r++) {
        if (r === col) continue;
        const f = a[r][col];
        if (f === 0) continue;
        for (let c = 0; c < 8; c++) a[r][c] -= f * a[col][c];
      }
    }
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) out[c * 4 + r] = a[r][4 + c];
    }
    return out;
  },

  transformPoint(m, v) {
    const [x, y, z] = v;
    const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1;
    return [
      (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
      (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
      (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
    ];
  },

  transformDir(m, v) {
    const [x, y, z] = v;
    return [
      m[0] * x + m[4] * y + m[8] * z,
      m[1] * x + m[5] * y + m[9] * z,
      m[2] * x + m[6] * y + m[10] * z,
    ];
  },

  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1;
    m[14] = (2 * far * near) / (near - far);
    return m;
  },

  lookAt(eye, center, up) {
    const z = vec3.normalize(vec3.sub(eye, center));
    const x = vec3.normalize(vec3.cross(up, z));
    const y = vec3.cross(z, x);
    const m = new Float32Array(16);
    m[0] = x[0]; m[1] = y[0]; m[2] = z[0]; m[3] = 0;
    m[4] = x[1]; m[5] = y[1]; m[6] = z[1]; m[7] = 0;
    m[8] = x[2]; m[9] = y[2]; m[10] = z[2]; m[11] = 0;
    m[12] = -vec3.dot(x, eye);
    m[13] = -vec3.dot(y, eye);
    m[14] = -vec3.dot(z, eye);
    m[15] = 1;
    return m;
  },

  // 提取世界旋转四元数（假设无负缩放，先归一化各基向量）
  toQuat(m) {
    const sx = Math.hypot(m[0], m[1], m[2]);
    const sy = Math.hypot(m[4], m[5], m[6]);
    const sz = Math.hypot(m[8], m[9], m[10]);
    const r00 = m[0] / sx, r01 = m[4] / sy, r02 = m[8] / sz;
    const r10 = m[1] / sx, r11 = m[5] / sy, r12 = m[9] / sz;
    const r20 = m[2] / sx, r21 = m[6] / sy, r22 = m[10] / sz;
    const trace = r00 + r11 + r22;
    let q;
    if (trace > 0) {
      const s = Math.sqrt(trace + 1) * 2;
      q = [(r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, s / 4];
    } else if (r00 > r11 && r00 > r22) {
      const s = Math.sqrt(1 + r00 - r11 - r22) * 2;
      q = [s / 4, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s];
    } else if (r11 > r22) {
      const s = Math.sqrt(1 + r11 - r00 - r22) * 2;
      q = [(r01 + r10) / s, s / 4, (r12 + r21) / s, (r02 - r20) / s];
    } else {
      const s = Math.sqrt(1 + r22 - r00 - r11) * 2;
      q = [(r02 + r20) / s, (r12 + r21) / s, s / 4, (r10 - r01) / s];
    }
    return quat.normalize(q);
  },
};
