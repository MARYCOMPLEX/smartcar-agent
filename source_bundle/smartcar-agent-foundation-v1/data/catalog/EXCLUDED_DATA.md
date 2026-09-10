# 明确排除的数据类别

以下内容没有进入清洗后的算法输入：

- `online-run-evidence/**`：历史运行结果、shell/chassis/assembly/validation；
- `hardware-kit/geometry-source/**`：原有坐标布局和 PR311 生成逻辑；
- `hardware-kit/repo-pr311/connection-modules/**`：历史车型连接模块；
- `connection-patterns.v1.json` / `reusable-interfaces.v1.json`：包含旧结构工作流/适配策略，不作为“新通用算法输入”；
- `generated-proxies/meshes/shell.stl`、`chassis.stl`、motor support、battery tray、boss 等：全部属于既有设计结果；
- optional LED/servo/magnet：不属于当前基础移动硬件输入包；
- historical design defaults、vehicle-specific placements、quality scores。

轮子没有使用任何历史 preview mesh。由于缺少官方 wheel STEP，清洗包仅根据 `hardware-inventory.json` 给出的 37 mm 直径和 13 mm 宽度生成圆柱碰撞代理；该代理不是制造几何，未来应由官方 CAD 替换。
