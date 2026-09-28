// WebGL 渲染器：GPU 线性混合蒙皮（LBS）+ 降级（CPU 蒙皮）渲染 + 调试线条。
import { mat4 } from './math.js';

const SKIN_VS = `
precision highp float;
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec4 aJoints;
attribute vec4 aWeights;
uniform mat4 uViewProj;
uniform mat4 uBones[MAX_BONES];
varying vec3 vNormal;
varying vec3 vWorldPos;
void main() {
  ivec4 j = ivec4(aJoints);
  // 线性混合蒙皮：v' = Σ wᵢ · Mᵢ · v
  mat4 skin = uBones[j.x] * aWeights.x + uBones[j.y] * aWeights.y
            + uBones[j.z] * aWeights.z + uBones[j.w] * aWeights.w;
  vec4 wp = skin * vec4(aPosition, 1.0);
  vWorldPos = wp.xyz;
  vNormal = normalize(mat3(skin) * aNormal);
  gl_Position = uViewProj * wp;
}`;

const SKIN_FS = `
precision mediump float;
varying vec3 vNormal;
varying vec3 vWorldPos;
uniform vec3 uLightDir;
uniform vec3 uColor;
void main() {
  vec3 n = normalize(vNormal);
  float diff = max(dot(n, normalize(uLightDir)), 0.0);
  vec3 col = uColor * (0.25 + 0.75 * diff);
  // 简单菲涅尔边缘光，突出形体
  float rim = pow(1.0 - abs(n.z), 2.0) * 0.25;
  gl_FragColor = vec4(col + rim, 1.0);
}`;

const RIGID_VS = `
precision highp float;
attribute vec3 aPosition;
attribute vec3 aNormal;
uniform mat4 uViewProj;
varying vec3 vNormal;
void main() {
  vNormal = aNormal;
  gl_Position = uViewProj * vec4(aPosition, 1.0);
}`;

const LINE_VS = `
precision highp float;
attribute vec3 aPosition;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * vec4(aPosition, 1.0); }`;

const LINE_FS = `
precision mediump float;
uniform vec4 uColor;
void main() { gl_FragColor = uColor; }`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('Shader error: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

function link(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('Link error: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

export class Renderer {
  constructor(canvas, maxBones) {
    const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
    if (!gl) throw new Error('WebGL 不可用');
    this.gl = gl;
    this.canvas = canvas;

    this.skinProg = link(gl, SKIN_VS.replace(/MAX_BONES/g, String(maxBones)), SKIN_FS);
    this.rigidProg = link(gl, RIGID_VS, SKIN_FS);
    this.lineProg = link(gl, LINE_VS, LINE_FS);

    this.loc = {
      skin: {
        aPosition: gl.getAttribLocation(this.skinProg, 'aPosition'),
        aNormal: gl.getAttribLocation(this.skinProg, 'aNormal'),
        aJoints: gl.getAttribLocation(this.skinProg, 'aJoints'),
        aWeights: gl.getAttribLocation(this.skinProg, 'aWeights'),
        uViewProj: gl.getUniformLocation(this.skinProg, 'uViewProj'),
        uBones: gl.getUniformLocation(this.skinProg, 'uBones'),
        uLightDir: gl.getUniformLocation(this.skinProg, 'uLightDir'),
        uColor: gl.getUniformLocation(this.skinProg, 'uColor'),
      },
      rigid: {
        aPosition: gl.getAttribLocation(this.rigidProg, 'aPosition'),
        aNormal: gl.getAttribLocation(this.rigidProg, 'aNormal'),
        uViewProj: gl.getUniformLocation(this.rigidProg, 'uViewProj'),
        uLightDir: gl.getUniformLocation(this.rigidProg, 'uLightDir'),
        uColor: gl.getUniformLocation(this.rigidProg, 'uColor'),
      },
      line: {
        aPosition: gl.getAttribLocation(this.lineProg, 'aPosition'),
        uViewProj: gl.getUniformLocation(this.lineProg, 'uViewProj'),
        uColor: gl.getUniformLocation(this.lineProg, 'uColor'),
      },
    };

    // 网格缓冲
    this.buffers = {
      position: gl.createBuffer(),
      normal: gl.createBuffer(),
      joints: gl.createBuffer(),
      weights: gl.createBuffer(),
      index: gl.createBuffer(),
      cpuPosition: gl.createBuffer(),
      cpuNormal: gl.createBuffer(),
      line: gl.createBuffer(),
    };
    this.indexCount = 0;
    this.degraded = false;

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.clearColor(0.07, 0.08, 0.11, 1);
  }

  uploadGeometry(g) {
    const gl = this.gl;
    this.indexCount = g.indices.length;
    this.degraded = g.degraded;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
    gl.bufferData(gl.ARRAY_BUFFER, g.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
    gl.bufferData(gl.ARRAY_BUFFER, g.normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.joints);
    gl.bufferData(gl.ARRAY_BUFFER, g.joints, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.weights);
    gl.bufferData(gl.ARRAY_BUFFER, g.weights, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, g.indices, gl.STATIC_DRAW);
    if (g.degraded) {
      this.cpuVertexCount = g.positions.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.cpuPosition);
      gl.bufferData(gl.ARRAY_BUFFER, g.positions, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.cpuNormal);
      gl.bufferData(gl.ARRAY_BUFFER, g.normals, gl.DYNAMIC_DRAW);
    }
  }

  updateCpuMesh(positions, normals) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.cpuPosition);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.cpuNormal);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(this.canvas.clientWidth * dpr);
    const h = Math.floor(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.gl.viewport(0, 0, w, h);
    return w / Math.max(1, h);
  }

  render(frame) {
    const gl = this.gl;
    const aspect = this.resize();
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const { camera, skinMatrices, jointPositions, parents, ikTarget, showBones, height } = frame;
    const proj = mat4.perspective(Math.PI / 4, aspect, 0.1, 100);
    const view = mat4.lookAt(camera.eye, camera.center, [0, 1, 0]);
    const viewProj = mat4.multiply(proj, view);

    // 地面网格
    this.drawGrid(viewProj, height);

    // 角色网格
    if (this.degraded && frame.cpuPositions) {
      this.updateCpuMesh(frame.cpuPositions, frame.cpuNormals);
      gl.useProgram(this.rigidProg);
      const L = this.loc.rigid;
      gl.uniformMatrix4fv(L.uViewProj, false, viewProj);
      gl.uniform3f(L.uLightDir, 0.5, 0.8, 0.6);
      gl.uniform3f(L.uColor, 0.9, 0.55, 0.35);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.cpuPosition);
      gl.enableVertexAttribArray(L.aPosition);
      gl.vertexAttribPointer(L.aPosition, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.cpuNormal);
      gl.enableVertexAttribArray(L.aNormal);
      gl.vertexAttribPointer(L.aNormal, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
      gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    } else if (!this.degraded && skinMatrices) {
      gl.useProgram(this.skinProg);
      const L = this.loc.skin;
      gl.uniformMatrix4fv(L.uViewProj, false, viewProj);
      gl.uniformMatrix4fv(L.uBones, false, skinMatrices);
      gl.uniform3f(L.uLightDir, 0.5, 0.8, 0.6);
      gl.uniform3f(L.uColor, 0.45, 0.7, 0.95);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
      gl.enableVertexAttribArray(L.aPosition);
      gl.vertexAttribPointer(L.aPosition, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
      gl.enableVertexAttribArray(L.aNormal);
      gl.vertexAttribPointer(L.aNormal, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.joints);
      gl.enableVertexAttribArray(L.aJoints);
      gl.vertexAttribPointer(L.aJoints, 4, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.weights);
      gl.enableVertexAttribArray(L.aWeights);
      gl.vertexAttribPointer(L.aWeights, 4, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.index);
      gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    }

    // 骨骼调试线
    if (showBones && jointPositions && parents) {
      const lines = [];
      for (let i = 0; i < parents.length; i++) {
        const p = parents[i];
        if (p >= 0) {
          lines.push(
            jointPositions[p * 3], jointPositions[p * 3 + 1], jointPositions[p * 3 + 2],
            jointPositions[i * 3], jointPositions[i * 3 + 1], jointPositions[i * 3 + 2],
          );
        }
      }
      const last = parents.length - 1;
      lines.push(
        jointPositions[last * 3], jointPositions[last * 3 + 1], jointPositions[last * 3 + 2],
        jointPositions[(last + 1) * 3], jointPositions[(last + 1) * 3 + 1], jointPositions[(last + 1) * 3 + 2],
      );
      this.drawLines(viewProj, new Float32Array(lines), [1, 0.85, 0.2, 1]);
    }

    // IK 目标标记（十字线）
    if (ikTarget) {
      const s = 0.12;
      const [x, y, z] = ikTarget;
      const cross = new Float32Array([
        x - s, y, z, x + s, y, z,
        x, y - s, z, x, y + s, z,
        x, y, z - s, x, y, z + s,
      ]);
      this.drawLines(viewProj, cross, [1, 0.25, 0.3, 1]);
    }
  }

  drawGrid(viewProj, height) {
    const lines = [];
    const size = 4, step = 0.5;
    for (let i = -size; i <= size; i += step) {
      lines.push(i, 0, -size, i, 0, size);
      lines.push(-size, 0, i, size, 0, i);
    }
    this.drawLines(viewProj, new Float32Array(lines), [0.3, 0.32, 0.38, 1]);
  }

  drawLines(viewProj, data, color) {
    const gl = this.gl;
    gl.useProgram(this.lineProg);
    const L = this.loc.line;
    gl.uniformMatrix4fv(L.uViewProj, false, viewProj);
    gl.uniform4fv(L.uColor, color);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.line);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(L.aPosition);
    gl.vertexAttribPointer(L.aPosition, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, data.length / 3);
  }
}
