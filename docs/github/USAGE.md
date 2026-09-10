# 使用与排错

| 命令 | 行为 |
|---|---|
| `setup.cmd` | 准备锁定环境，检查输入哈希和原生几何工具 |
| `run.cmd` | 运行默认真实小车的完整设计 |
| `run.cmd --stage analyze` | 输入、坐标、内部空间和硬件检查 |
| `test.cmd -q` | 全量几何回归 |
| `batch.cmd --help` | 批量测试命令说明 |

首次联网安装 uv 到仓库内 `.tools/`，校验官方压缩包 SHA256；已有同版本 uv 时直接使用。Python 3.12 存在则复用解释器，否则由 uv 下载。第三方包只安装在本项目 `.venv/`。启动器使用单次 PowerShell 执行策略，不修改系统执行策略或 PATH。

第二次运行会快速检查依赖是否齐全。准备完环境后，如需完全跳过依赖同步，可直接运行 `.venv/Scripts/python.exe -m smartcar.pipeline`。

## 配置与自定义硬件

```powershell
.\run.cmd --appearance "C:\models\car.stl" --kit "C:\kit\kit_manifest.json" --profile config/manufacturing.json --vehicle-policy config/vehicle_design.json
```

新增套件应遵循随附硬件定义的数据结构；当前算法围绕两个驱动模块和两个从动轮工作，修改数量或机械拓扑需要扩展正式算法。支持更换输入路径不等于支持任意硬件架构。

## 批量输入

```powershell
.\batch.cmd prepare --archive "C:\models\cars.zip" --suite benchmarks/new-cars
.\batch.cmd run --suite benchmarks/new-cars --label initial --jobs 2
```

suite 和 label 选择新名称。每批冻结源码和配置，避免运行途中编辑代码使结果混用实现。加入 `--stage analyze` 只做分析；`--indices` 可选部分模型，具体格式见 `batch.cmd run --help`。最多两个完整设计并行适用于原 32 GB 测试机器，低内存机器用 `--jobs 1`。

## 常见问题

- **下载失败**：确认可访问 GitHub、PyPI 和 Python 下载地址后重跑命令；不需要清空目录。首次运行并非离线安装包。
- **锁文件不同步**：普通使用先 `git pull` 获取对应版本的配置与 `uv.lock`。开发修改依赖后应重新锁定和验证，不能靠删除锁文件让环境随意升级。
- **INPUT_SCALE_TOO_SMALL / EMPTY_REPAIRED_VOLUME**：检查 STL 实际尺寸；使用正确 `--input-unit`，或明确给出 `--target-length-mm`。不应降低壁厚。
- **VOXEL_GRID_LIMIT / RESOURCE_LIMIT**：检查单位和尺寸是否误放大；内存预算失败与布局不可能是两种不同原因。
- **INFEASIBLE**：查看 `agent_trace.json`、`scale_search.json` 和 `05_wheel_candidates/diagnostics-scale-*.json`；表示已尝试范围内没有有效解，不是全局不可能的数学证明。
- **RUN_ALREADY_EXISTS**：使用新 run-id 或省略该参数。旧结果不会覆盖。
- **完成但有 FAIL**：查看具体数值和中间几何。这是有效的失败报告，不能当作可打印验收通过。
- **初次 CAD 较慢**：首次从真实 STEP 提取固有接口；后续可复用硬件哈希缓存。缓存不包含车型布局。

程序对每轮运行保留 `execution.log` 和 `run_status.json`。原生库进程异常退出会记录为 ERROR，避免留下假的 RUNNING 状态。系统不需要 GitHub 登录或 API Key 来做几何设计；GitHub 身份仅用于克隆私有代码。
