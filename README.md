# WebGL 骨骼动画 · 线性混合蒙皮（LBS）

纯 WebGL1 + 原生 ES Module 实现的骨骼动画演示：程序化生成骨骼 / 权重 / 关键帧，
支持动画播放、混合、CCD 逆运动学，并实时显示帧率与骨骼数。

## 运行

```bash
npm run serve        # python3 -m http.server 8080
# 打开 http://localhost:8080
```

## 测试

```bash
npm test             # 数学库 / 插值 / 混合 / 归一化 / IK / 降级 + Worker 冒烟
```

## 架构

- **主线程**（`js/main.js` + `js/renderer.js`）：只做渲染与输入。
  `PerformanceObserver` 监听 longtask 验证主线程不卡；HUD 显示 FPS / 帧时间 / Worker 耗时。
- **Web Worker**（`js/worker.js`）：承担全部重计算 —— 关键帧采样（slerp/lerp）、
  动画混合、CCD IK、蒙皮矩阵计算；结果以 Transferable `Float32Array` 回传。
- **蒙皮**：≤64 骨骼走 GPU 顶点着色器 LBS（4 影响，uniform 矩阵数组）；
  超过上限自动**降级为刚性绑定**（每顶点 1 骨骼）并在 Worker 内做 CPU 蒙皮，
  主线程仍然零计算。
- **IndexedDB**（`js/db.js`）：持久化程序化生成的动画剪辑与 UI 设置，刷新后恢复。
- **TypedArray**：所有位姿 / 矩阵 / 顶点数据均为 `Float32Array`/`Uint16Array`。

## 验收标准对照

| 标准 | 实现 | 验证 |
| --- | --- | --- |
| 骨骼动画正确 | 链式骨骼 + 世界矩阵求值 | `静止位姿的蒙皮矩阵 = 单位阵` |
| 蒙皮正确 | 着色器 LBS / CPU 蒙皮同公式 | `CPU 蒙皮与 LBS 公式一致` |
| 关键帧插值正确 | 旋转 slerp、位移 lerp、循环采样 | `采样关键帧时刻 = 关键帧值` 等 3 项 |
| 动画混合正确 | 双剪辑 slerp/lerp 交叉淡化 | `blend w=0/1/0.5` 2 项 |
| 逆运动学正确 | CCD 迭代求解 | `直链可达目标` / `不可达完全伸展` |
| 骨骼数上限有降级 | >64 → 刚性绑定 + CPU 蒙皮 | `降级阈值` + Worker 冒烟 |
| 权重归一化正确 | 截断 4 影响并归一化 Σw=1 | `权重归一化` / `刚性绑定` |
| 帧率可观测 | FPS / 帧时间 / Worker 耗时 HUD | 界面实时显示 |
| 降级方案可用 | 降级模式 UI 橙色标识 | Worker 冒烟降级路径 |
| 主线程不卡 | 计算全在 Worker + longtask 监控 | `长任务` 计数保持 0 |

## 操作

- 左键拖动：旋转相机；滚轮：缩放
- 勾选 IK 后，Shift+拖动或右键拖动红色十字目标
- 骨骼数滑块超过 64 可观察自动降级
