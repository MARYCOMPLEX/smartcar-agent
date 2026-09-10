# 随附输入与来源

仓库保留 `source_bundle/smartcar-agent-foundation-v1/data/` 中的原始字节。默认小车 `appearance/vehicle_appearance.stl` 约 1.35 MB，原始网格不是闭合体，pipeline 会执行修复及空间分析。

`hardware/kit_manifest.json` 定义固定套件：两个 LA024/LA016 驱动单元、一个控制板、一个电池、一个开关、两个驱动轮、两个从动轮和按设计计算数量的紧固件。组件定义使用各自目录的相对路径引用 STEP/STP 或碰撞代理。

原始来源说明、上游 commit/release、几何 SHA256 位于每份 `definition.json`、`appearance.json`、`data/provenance/` 和 `data/catalog/`。`setup.cmd` 会逐个验证实际文件字节；整个输入数据约 16 MB，无需 Git LFS 或额外下载链接。

轮胎采用来自已给定 37 × 13 mm 尺寸的圆柱碰撞代理。没有把代理当成完整轮毂 CAD。柔性电缆显示姿态与硬件刚性包络分别处理，不拿已有电缆外形当作已完成走线。

上游溯源记录来自用户提供的数据包，仓库不改变其授权或所有权，也未另行声明开源许可。附带旧 prompts、旧流程合同、历史布局和旧车壳不作为设计输入。

仍需补齐：真实轮毂轴向配合与限位、控制板连接器特征坐标、插头及线束空间、质量、打印机孔配合和实物夹持/强度数据。当前验证报告保留这些缺口。
