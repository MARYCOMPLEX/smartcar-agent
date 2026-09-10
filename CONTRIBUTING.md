# 开发与验证约定

先运行 `setup.cmd` 和 `test.cmd -q`。源码在 `src/smartcar/`；物理制造参数在 `config/manufacturing.json`，车辆比例与外观政策在 `config/vehicle_design.json`。默认输入均采用仓库相对路径。

修改几何能力时，先复现正式 pipeline 的失败，再修改对应模块，加入能触发原错误的几何回归，并使用新 run-id 重新运行。不得手工修改输出 STL、硬件姿态、layout 或 validation report 来制造成功；不得按文件名、哈希或当前车型专用坐标决定布局，也不得放宽验收阈值掩盖失败。

Agent 只选择策略和调度模块；坐标、碰撞、距离、轴线、墙厚和装配路径必须由算法计算。保持 Layout、Structure、Assembly 和 Validator 的职责边界。

每轮运行保存源码快照与输入摘要。旧运行是不可变证据，新修改不应声称已经获得旧批次验证。`runs/`、`cache/`、`.venv/`、批量结果和本机记录均由 Git 忽略；提交源码、配置、真正必要的输入以及可复现测试。

升级依赖时，明确更新 `pyproject.toml` 中相关版本约束，使用 uv 0.11.14 执行 `uv lock`，在全新环境运行安装检查、回归和完整设计。不能只在已有原生库环境中做 import 后就宣称兼容。当前平台锁定为 Windows x64 / Python 3.12。

不要将 GitHub 凭据、`.env`、对话记录或无关文件提交到仓库。保留原硬件 CAD 字节、哈希和来源说明。
