# 硬件感知自动机械设计 Agent

从小车外观 STL 和固定真实硬件数据出发，自动计算坐标系、轮位和内部布局，生成外壳、底盖、安装结构、装配模型与独立验证报告。包含可运行的 Python 工程、原始示例 STL、硬件 STEP/STP、制造配置和几何回归测试。运行不需要 API Key。

**软件可运行，生成件需单独验收。** 随附默认外观目前会触发已知的上壳局部薄壁 FAIL，适合检查完整流程和失败反馈，不能作为已通过验收的打印样件。模型验收状态和实测证据见 [验证说明](docs/github/VERIFICATION.md)。

## 拉取后直接运行（Windows x64）

```powershell
git clone https://github.com/MARYCOMPLEX/smartcar-agent.git
cd smartcar-agent
.\run.cmd
```

私有仓库需要用有访问权限的 GitHub 账号拉取。**首次运行需要联网**访问 GitHub 和 PyPI：启动脚本自动准备 uv、Python 3.12 和项目专用 `.venv`，按 `uv.lock` 安装固定依赖，然后处理仓库自带的小车。无需手工安装 CadQuery、配置系统 PATH 或复制原电脑的缓存。

支持范围为 Windows 10/11 x64；建议 16 GB 以上内存、至少 8 GB 可用磁盘，批量设计建议 32 GB 内存。首次依赖下载较大；完整几何设计需要分钟到几十分钟，复杂输入及多次重试可能更久。Linux/macOS 尚未验证，不承诺使用这份 Windows 锁文件直接运行。

如果只想先确认安装和输入：

```powershell
.\setup.cmd
.\run.cmd --stage analyze
```

`setup.cmd` 会检查 8 份原始几何文件的 SHA256，并实际执行 CAD STEP 往返、布尔几何、射线碰撞、约束求解和 PNG 渲染。`analyze` 会分析原模型、建立坐标和内部空间、导入真实硬件，不进行完整布局与结构设计。

## 输入自己的模型

```powershell
.\run.cmd --appearance "C:\models\my car.stl" --stage analyze
.\run.cmd --appearance "C:\models\my car.stl"
```

STL 不可靠地保存单位。默认按毫米解释；已知源单位时用 `--input-unit mm|cm|m|in`。不知道导出单位但知道目标初始车长时，可以明确指定：

```powershell
.\run.cmd --appearance "C:\models\my car.stl" --target-length-mm 200
```

这里 200 mm 是初始车长，硬件尺寸保持真实毫米值。硬件无法容纳时仍可能继续放大外观；加 `--max-scale 1` 可禁止后续放大。`--input-unit` 与 `--target-length-mm` 不能同时使用，程序不会按文件名猜测单位。

所有命令均可从其他目录通过启动器的完整路径调用；输入的相对路径以调用时的目录为准。默认输入、配置、缓存及输出则定位到当前克隆目录。

更多操作见 [使用与排错](docs/github/USAGE.md)，固定硬件信息见 [数据说明](docs/github/DATA.md)。

## 输出位置

每次运行写入新的 `runs/<时间戳>/`，原 STL 和已有结果不会被覆盖。控制台打印完整输出路径。先看 `run_status.json` 和 `output/validation_report.json`，再检查模型。

| 文件 | 内容 |
|---|---|
| `output/body.stl`、`bottom_cover.stl` | 上壳与底盖 |
| 其他 STL | 电机压板、紧固试片等独立打印件 |
| `assembly.glb`、`exploded_assembly.glb` | 装配模型与爆炸图 |
| `printable.3mf` | 打印件集合，未进行切片 |
| `rigid_hardware_assembly.step` | 有真实 CAD 的硬件装配；不包含网格打印件 |
| `layout.json`、`assembly_plan.json` | 硬件姿态与已验证装配路径 |
| `validation_report.json`、`bom.json`、`design_report.md` | 测量、验收状态、物料与设计报告 |

STL/JSON 使用毫米，GLB 使用米。全过程中间 STL、PNG、CSV、JSON 保存在编号阶段目录中。进程成功结束只说明流程执行完毕，设计是否合格仍取决于报告里的 FAIL / WARNING。

## 开发与测试

```powershell
.\test.cmd -q
.\batch.cmd prepare --archive "C:\models\cars.zip" --suite benchmarks/my-suite
.\batch.cmd run --suite benchmarks/my-suite --label first-pass --jobs 2
```

已有 uv 0.11.14 时也可以使用：

```powershell
uv sync --locked
uv run --locked python -m smartcar.pipeline --stage analyze
uv run --locked pytest -q
```

`pyproject.toml` 和 `uv.lock` 固定了已验证的 Windows 依赖；保留 CasADi 3.6.7 以避免曾复现的原生库退出问题。启动器使用 `--locked`，依赖声明与锁文件不一致时会报错，不会静默升级。[uv 锁定行为说明](https://docs.astral.sh/uv/concepts/projects/sync/)。

GitHub Actions 在 Windows 上执行同一安装脚本、完整单元回归、默认模型分析与真实硬件导入。完整整车设计的验证记录及历史十模型结论见 [验证说明](docs/github/VERIFICATION.md)。

## 已知边界

当前为可审计策略队列驱动的工程原型，没有接入运行时大模型。几何坐标、碰撞和间隙由算法计算。不能保证所有形状都适配。

历史十模型 `release-v14` 全部通过轮位检查，其中 7 例无正式 FAIL，5、6、10.stl 仍有局部薄壁。真实轮毂轴向接口、连接器与线束语义、打印机孔配合、夹持强度和实物试装仍不完整；不能把生成结果直接视为已认证可打印、可装配的成品。

[架构与模块](docs/github/ARCHITECTURE.md) · [开发规则](CONTRIBUTING.md) · [输入来源](docs/github/DATA.md)
