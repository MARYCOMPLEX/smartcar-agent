"""Read-only corpus report and measured wheel-placement comparisons."""
from pathlib import Path
import argparse
import collections
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle,Rectangle
from smartcar.io import read_json,write_json,write_csv
from smartcar.batch import summarize_run
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.understanding.axle_anchors import detect_axle_anchors
from smartcar.validation.axle_layout import validate_axle_positions


def inspect(run,profile):
    summary=summarize_run(run)
    if not (run/'output/layout.json').exists():return summary
    layout=read_json(run/'output/layout.json');wheels=[i for i in layout['instances'] if 'wheel' in i['role']]
    centers=np.asarray([np.asarray(i['bounding_box']).mean(0) for i in wheels])
    if len(centers)==4:
        paired=centers[np.argsort(centers[:,1])]
        summary['rear_front_tracks_mm']=[float(np.ptp(paired[:2,0])),float(np.ptp(paired[2:,0]))]
    evidence_path=run/'03_coordinate_frame/axle_evidence.json'
    if evidence_path.exists():evidence=read_json(evidence_path)
    elif (run/'03_coordinate_frame/normalized.stl').exists():
        mesh=trimesh.load(run/'03_coordinate_frame/normalized.stl',force='mesh')
        evidence=detect_axle_anchors(mesh,read_json(run/'03_coordinate_frame/semantic_regions.json'))
    else:
        summary['supplementary_audit_unavailable']='historical compact evidence has no source mesh or saved axle evidence'
        return summary
    checks=validate_axle_positions([np.asarray(i['bounding_box']).mean(0) for i in wheels],[np.ptp(np.asarray(i['bounding_box']),axis=0)[1]/2 for i in wheels],
            evidence['source_bounds_mm'],evidence['axles'],layout['scale'],profile)
    summary['posthoc_axle_audit']=dict(kind='separate current diagnostic; original report is unchanged',checks=checks)
    correspondence=next((c for c in checks if c['check']=='axle_layout:source_correspondence'),{})
    summary['source_axle_max_error_mm']=correspondence.get('measurements',{}).get('maximum_error_mm')
    summary['posthoc_axle_failures']=[c['check'] for c in checks if c['status']=='FAIL']
    report_path=run/'output/validation_report.json'
    if report_path.exists():
        checks=read_json(report_path).get('checks',[])
        body_check=next((c for c in checks if c['check']=='exported_STL:body'),None)
        if body_check is not None:
            summary['printed_body_dimensions_mm']=body_check['measurements']['dimensions_mm']
    if 'printed_body_dimensions_mm' not in summary and (run/'output/body.stl').exists():
        summary['printed_body_dimensions_mm']=trimesh.load(run/'output/body.stl',force='mesh').extents.tolist()
    return summary


def draw_run(run,axes,title,include_source=True):
    if run is None:
        for ax in axes:ax.axis('off');ax.text(.5,.5,'No completed design',ha='center')
        return
    mesh=trimesh.load(run/'03_coordinate_frame/normalized.stl',force='mesh')
    frame=read_json(run/'03_coordinate_frame/frame.json')
    design=read_json(run/'03_coordinate_frame/design_frame.json')
    mesh.apply_transform(np.asarray(design['input_to_design'])@np.linalg.inv(frame['input_to_vehicle']))
    layout=read_json(run/'output/layout.json')
    body=trimesh.load(run/'output/body.stl',force='mesh')
    previews=[]
    for shape,color,alpha in ([(mesh,'#c3c9cd',.2)] if include_source else [])+[(body,'#46758b',.18)]:
        if len(shape.faces)>35000:
            preview=trimesh.Trimesh(np.array(shape.vertices,dtype=np.float64,order='C',copy=True),
                np.array(shape.faces,dtype=np.int64,order='C',copy=True),process=False)
            faces=preview.simplify_quadric_decimation(face_count=35000).triangles
        else:faces=shape.triangles
        previews.append((faces,color,alpha))
    for ax,coords in zip(axes,[[1,2],[1,0]]):
        for faces,color,alpha in previews:
            # Preserve the surface in previews. Striding triangle indices
            # visually tears holes into an otherwise closed printable mesh.
            ax.add_collection(PolyCollection(faces[:,:,coords],facecolor=color,edgecolor='none',alpha=alpha))
        for inst in layout['instances']:
            b=np.asarray(inst['bounding_box']);center=b.mean(0);d=b[1]-b[0]
            if 'wheel' in inst['role']:
                if coords==[1,2]:ax.add_patch(Circle(center[coords],d[1]/2,facecolor='#e3e9ee',edgecolor='#202c35',lw=1.4))
                else:ax.add_patch(Rectangle(b[0,coords],d[1],d[0],fill=False,color='#202c35',lw=1.4))
            elif coords==[1,0]:
                color={'drive_unit':'#c76839','battery':'#d7b54c','main_controller':'#469276','power_switch':'#9966b5'}.get(inst['role'],'#777')
                ax.add_patch(Rectangle(b[0,coords],d[1],d[0],fill=False,color=color,lw=1))
        epath=run/'03_coordinate_frame/axle_evidence.json'
        evidence=read_json(epath) if epath.exists() else detect_axle_anchors(trimesh.load(run/'03_coordinate_frame/normalized.stl',force='mesh'),read_json(run/'03_coordinate_frame/semantic_regions.json'))
        for axle in evidence['axles']:ax.axvline(axle['y_mm']*layout['scale'],color='#cc4b36',ls='--',lw=1.2)
        ax.autoscale_view();ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('Y / mm');ax.set_ylabel(('Z' if coords==[1,2] else 'X')+' / mm')
    wheels=[np.asarray(i['bounding_box']).mean(0) for i in layout['instances'] if 'wheel' in i['role']]
    ratio=np.ptp(np.asarray(wheels)[:,1])/(frame['dimensions_mm'][1]*layout['scale'])
    axes[0].set_title(title+f" | scale {layout['scale']:.2f} | WB/L {ratio:.3f}")
    if len(axes)>1:axes[1].set_title('Top: actual hardware; red lines = measured source axles')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True,type=Path);parser.add_argument('--render',action='store_true')
    parser.add_argument('--label',help='Use this exact cohort for the comparison table; retain every historical run in the issue log.')
    args=parser.parse_args();suite=args.suite.resolve();root=Path(__file__).resolve().parents[1]
    out=suite/'reports';out.mkdir(exist_ok=True)
    manifest=read_json(suite/'manifest.json');profile=ManufacturingProfile.load(root/'config/manufacturing.json')
    input_quality=[]
    for model in manifest['models']:
        source_report=root/f'runs/{suite.name}-baseline-v2-m{model["index"]:02d}/01_input/mesh_analysis.json'
        if source_report.exists():
            stats=read_json(source_report)
            # The baseline used no declared conversion. Keep these coordinates
            # as source units, including the five explicitly unconfirmed files.
            input_quality.append(dict(index=model['index'],model=model['source_entry'],
                source_coordinate_dimensions=stats['dimensions_mm'],triangles=stats['triangles'],
                component_count=stats['component_count'],watertight=stats['watertight'],manifold=stats['manifold'],
                nonmanifold_edges=stats['nonmanifold_edges'],boundary_edges=stats['boundary_edges'],
                degenerate_triangles=stats['degenerate_triangles'],surface_area_source_units2=stats['surface_area_mm2'],
                volume_source_units3=stats['volume_mm3'],evidence=str(source_report.relative_to(root))))
    write_json(out/'input_quality.json',dict(kind='original baseline mesh analysis; source coordinates are not confirmed physical units',models=input_quality))
    write_csv(out/'input_quality.csv',input_quality)
    records=[];latest={};baseline={};cancellations=[];wall_audits={}
    for path in out.glob('*-critical-wall-audit.json'):
        for row in read_json(path).get('models',[]):wall_audits[row['run']]=dict(row,source_report=path.name)
    folders=sorted((suite/'batches').iterdir(),key=lambda p:p.stat().st_ctime)
    for folder in folders:
        if not (folder/'summary.json').exists():continue
        summary=read_json(folder/'summary.json')
        if summary['stage']!='all':continue
        entries=list(summary['models']);known={r['index'] for r in entries}
        if (folder/'cancellation.json').exists():
            cancellations.append(read_json(folder/'cancellation.json'))
            for model in manifest['models']:
                run=root/f'runs/{suite.name}-{folder.name}-m{model["index"]:02d}'
                if model['index'] not in known and (run/'run_status.json').exists():
                    entries.append(dict(index=model['index'],model=model['source_entry'],input_sha256=model['sha256'],run=str(run)))
        for record in entries:
            run=root/'runs'/Path(record['run'].replace('\\','/')).name;measured=inspect(run,profile)
            record=dict(record,**measured,label=folder.name)
            if run.name in wall_audits:record['supplementary_wall_audit']=wall_audits[run.name]
            records.append(record)
            if folder.name=='baseline-v2':baseline[record['index']]=record
            elif (args.label is None or folder.name==args.label) and record.get('run_status',{}).get('status') not in ['RUNNING','CANCELLED']:
                latest[record['index']]=record
    write_json(out/'all_runs.json',records)
    rows=[]
    for r in records:
        rows.append(dict(model=r['model'],batch=r['label'],status=r.get('run_status',{}).get('status'),seconds=r.get('seconds'),scale=r.get('scale'),
            **r.get('counts',{}),wheelbase_fraction=r.get('wheel_metrics',{}).get('wheelbase_to_input_length'),
            source_axle_max_error_mm=r.get('source_axle_max_error_mm'),posthoc_axle_failures='; '.join(r.get('posthoc_axle_failures',[])),
            supplementary_wall_status=r.get('supplementary_wall_audit',{}).get('additional_status'),
            supplementary_wall_min_mm=r.get('supplementary_wall_audit',{}).get('measurements',{}).get('minimum_mm'),
            blockers='; '.join(r.get('blocking_checks',[])),run=r['run']))
    write_csv(out/'all_runs.csv',rows)
    write_json(out/'selected_results.json',dict(cohort=args.label or 'latest completed per model',models=[latest.get(m['index'],dict(index=m['index'],model=m['source_entry'],status='PENDING')) for m in manifest['models']]))
    timings=[dict(index=r['index'],model=r['model'],seconds=r['seconds'],
        minutes=r['seconds']/60,geometry_evaluations=r.get('run_status',{}).get('geometry_evaluations'),
        selected_iteration=r.get('run_status',{}).get('selected_iteration'),
        status=r.get('run_status',{}).get('status')) for r in latest.values() if r.get('seconds') is not None]
    def time_statistics(rows):
        values=[r['seconds'] for r in rows]
        return dict(models=len(values),minimum_seconds=min(values),median_seconds=float(np.median(values)),
            maximum_seconds=max(values),sum_of_case_wall_seconds=sum(values)) if values else dict(models=0)
    timing_report=dict(cohort=args.label or 'latest completed per model',completed_models=len(timings),
        corpus_models=len(manifest['models']),cases=timings,all_completed=time_statistics(timings),
        single_geometry_evaluation=time_statistics([r for r in timings if r['geometry_evaluations']==1]),
        multiple_geometry_evaluations=time_statistics([r for r in timings if r['geometry_evaluations'] and r['geometry_evaluations']>1]),
        whole_batch_elapsed_seconds=None,
        note='Public batch measures each case with perf_counter. Cases ran with two workers and concurrent diagnostic work. '
        'Per-case wall times cannot be summed or divided by two to recover the exact whole-batch duration. '
        'This is observed runtime, not a next-run guarantee or CPU time.')
    write_json(out/'runtime_summary.json',timing_report)
    write_csv(out/'runtime_summary.csv',timings)
    verification=[]
    categories={
        'mesh':('exported_STL:',),
        'wheel_layout':('axle_layout:',),
        'drive':('wheel_ground:','wheel_rotation:','shaft_alignment:'),
        'collision_clearance':('hardware_clearance:','hardware_structure:','containment:'),
        'assembly':('assembly_path:',),
        'mounting':('pcb_mounting','PCB_mounting','battery_mounting','motor_retention','passive_wheel_mounting','fastener_engagement:','mating:'),
        'access':('switch_accessibility','opening_accessibility','wheel_side_access:','fastener_tool_access:','connector_accessibility'),
        'manufacturing':('wall_thickness:','fdm:','self_intersection:','closure_fit','closure_intersection','closure_clearance','fastener_retention')}
    for model in manifest['models']:
        record=latest.get(model['index']);path=Path(record['run'])/'output/validation_report.json' if record else None
        report=read_json(path) if path and path.exists() else {};checks=report.get('checks',[])
        groups={}
        for name,prefixes in categories.items():
            selected=[c for c in checks if c['check'].startswith(prefixes)]
            counts=dict(collections.Counter(c['status'] for c in selected))
            state='MISSING' if not selected else 'FAIL' if counts.get('FAIL') else 'WARNING' if counts.get('WARNING') else 'PASS'
            groups[name]=dict(status=state,counts=counts,checks=selected)
        wall=next((c.get('measurements',{}) for c in checks if c['check']=='wall_thickness:body'),{})
        grouped_names={c['check'] for g in groups.values() for c in g['checks']}
        verification.append(dict(model=model['source_entry'],run=record['run'] if record else None,
            release_ready=report.get('release_ready'),groups=groups,body_sampled_minimum_wall_mm=wall.get('minimum_mm'),
            body_wall_rays=wall.get('samples'),body_wall_below_minimum=wall.get('below_minimum'),
            other_checks=[c for c in checks if c['check'] not in grouped_names]))
    write_json(out/'verification_matrix.json',dict(cohort=args.label,models=verification,scope='aggregation of original measured checks; missing checks never count as passing'))
    write_csv(out/'verification_matrix.csv',[dict(model=v['model'],**{k:g['status'] for k,g in v['groups'].items()},
        body_sampled_minimum_wall_mm=v['body_sampled_minimum_wall_mm'],body_wall_rays=v['body_wall_rays'],
        body_wall_below_minimum=v['body_wall_below_minimum'],release_ready=v['release_ready']) for v in verification])
    families=collections.defaultdict(list)
    for r in records:
        report_path=Path(r['run'])/'output/validation_report.json'
        if r.get('run_status',{}).get('status')=='ERROR':
            families['input_units_or_repair'].append(dict(model=r['model'],batch=r['label'],error=r['run_status']))
        if r.get('supplementary_wall_audit',{}).get('additional_status')=='FAIL':
            families['posthoc_wall_thickness'].append(dict(model=r['model'],batch=r['label'],evidence=r['supplementary_wall_audit']))
        # Native export repair may fail before the complete validator returns.
        # Preserve that evidence even for a cancelled diagnostic cohort.
        for structure_path in (Path(r['run'])/'design_iterations').glob('*/09_structure/structure.json'):
            structure=read_json(structure_path)
            for part,check in structure.get('print_mesh_stabilization',{}).items():
                if check.get('status')=='FAIL':
                    families['print_mesh_repair'].append(dict(model=r['model'],batch=r['label'],part=part,
                        evaluation=str(structure_path.relative_to(Path(r['run']))),evidence=check))
        trace_path=Path(r['run'])/'agent_trace.json'
        if trace_path.exists():
            for event_index,event in enumerate(read_json(trace_path)):
                # Synthesis exceptions can happen before a validator has a
                # mesh to inspect. Preserve these as issues too. Ordinary
                # sampled infeasibility at a smaller scale remains in the
                # source trace and is not mislabeled as a software defect.
                if event.get('design_iteration') and event.get('failure'):
                    code=str(event.get('error_code',event['failure']))
                    families['synthesis:'+code].append(dict(model=r['model'],batch=r['label'],
                        evaluation=event['design_iteration'],trace_event_index=event_index,
                        evidence={k:event[k] for k in ['failure','error_code','scale','topology','action','measurements'] if k in event}))
        report_paths=list((Path(r['run'])/'design_iterations').glob('*/12_validation/validation_report.json'))
        if report_path.exists():report_paths.append(report_path)
        seen_checks=set()
        for evidence_path in report_paths:
            for check in read_json(evidence_path)['checks']:
                if check['status']=='FAIL':
                    signature=json.dumps(check,sort_keys=True)
                    if signature in seen_checks:continue
                    seen_checks.add(signature)
                    key=check['check'].split(':')[0]
                    families[key].append(dict(model=r['model'],batch=r['label'],evaluation=str(evidence_path.relative_to(Path(r['run']))),evidence=check))
        for check in r.get('posthoc_axle_audit',{}).get('checks',[]):
            if check['status']=='FAIL':families['posthoc_'+check['check']].append(dict(model=r['model'],batch=r['label'],evidence=check))
    write_json(out/'issue_catalog.json',dict(kind='measured failures grouped across immutable runs',issues=families))
    write_csv(out/'issue_catalog.csv',[dict(issue=k,models='; '.join(sorted(set(v['model'] for v in values))),observations=len(values),batches='; '.join(sorted(set(v['batch'] for v in values)))) for k,values in sorted(families.items())])
    def fmt(v,d=2):return '—' if v is None else f'{v:.{d}f}'
    lines=['# 归档小车模型：批量测试与轮位修复记录','',
        f"输入压缩包含 {len(manifest['models'])} 个 STL；SHA256 `{manifest['archive_sha256']}`。原文件和历次运行产物均保留。",'',
        '五个最长边约为 2 个坐标单位的 STL，在复测中采用明确标注的初始车长 200 mm。此为诊断测试假设，尚未确认成品尺寸；其余模型保持原始毫米尺寸，再由正式求解器搜索放大比例。','',
        '本报告的“补充轴距审计”由当前独立测量程序重新计算，单独保存，未修改旧版 validation_report.json。旧版零 FAIL 不代表轮位合理。','',
        '另一次加强壁厚复核发现，旧版 1200 条随机表面射线会漏掉小连接薄片；补充复核单独记录在 *-critical-wall-audit.json。凡该复核为 FAIL 的历史模型，即使原报告零 FAIL，也不能据此验收。','',
        '轴距比例使用归一化输入车长×设计比例为分母；打印外壳的实际尺寸另存于 all_runs.json 的 printed_body_dimensions_mm。','',
        '## 原始输入质量','',
        '| 模型 | 源坐标尺寸 | 三角面 | 连通组件 | 水密 / 流形 | 非流形边 |',
        '|---|---|---:|---:|---|---:|']
    for q in input_quality:
        yn=lambda value:'是' if value else '否'
        lines.append(f"| {q['model']} | {' × '.join(fmt(v,3) for v in q['source_coordinate_dimensions'])} | {q['triangles']} | {q['component_count']} | {yn(q['watertight'])} / {yn(q['manifold'])} | {q['nonmanifold_edges']} |")
    lines += ['', '以上来自原始基线的正式网格分析，未将无单位坐标冒充已确认毫米。表面积、可计算体积和更多质量指标见 input_quality.json / CSV；原网格不闭合并不等于最终结构无解。','',
        '## 每个模型的基线和'+('统一批次 '+args.label if args.label else '最近一次已完成复测'),'',
        '| 模型 | 基线 | 最近复测 | 比例 | 轴距/车长 | 原轴线最大偏移 mm | FAIL 数 |','|---|---|---|---:|---:|---:|---:|']
    for model in manifest['models']:
        b=baseline.get(model['index'],{});r=latest.get(model['index'],{})
        ratio=r.get('wheel_metrics',{}).get('wheelbase_to_input_length');error=r.get('source_axle_max_error_mm')
        lines.append(f"| {model['source_entry']} | {b.get('run_status',{}).get('status','未完成')} | {r.get('label','未完成')} / {r.get('run_status',{}).get('status','')} | {fmt(r.get('scale'))} | {fmt(ratio,3)} | {fmt(error)} | {r.get('counts',{}).get('FAIL','—')} |")
    lines += ['', '## 已确认的根因与正式实现修改','',
        '1. 原模型轮胎的分割结果没有传入布局，驱动轮和从动轮分别优化。现改为测量原轴线区域、联合搜索两排轮子，轴距、轮拱对应、前后悬和轮距为独立约束。',
        '2. 仅识别独立圆轮会漏掉融合轮胎；大平面有时会被误认成车底。现用多个低位水平截面的双侧接地区域补充坐标系和轴线证据。',
        '3. 旧轮拱空洞迫使电机远离原轴线。现依据实测轮胎区域、相邻车身侧面和车腹高度重建局部设计包络，保存增减体积；超出外观改动预算的提案不直接采用。',
        '4. 硬件本体不碰撞仍可能出现电机压盖、螺丝柱重叠。新增结构占位检查，并在候选去重之前应用硬约束，避免有效方向和位置被丢弃。',
        '5. 底盘截面较窄时，PCB 柱或从动轮支架可能悬空。新增在车身投影范围内生成底部连接梁的算法，并避让实际轮胎和开口切除体。',
        '6. 原闭合筋连向原外观表面，而该处可能已经开了轮孔。现连向切孔后的实际壳体，布尔检查连通性，并在壳体离散重建时恢复连接筋。',
        '7. 原轮位只预留车底离地间隙，遗漏外露螺丝头高度。候选现同时考虑螺丝头和刚性间隙。',
        '8. 电池不碰轮胎仍可能使轮拱上方的托盘底过薄。现按轮拱实际包络为托盘保留名义壁厚；其他支撑柱也预先避让轮胎运动区域。',
        '9. 壳体重建后恢复连接筋，可能同时恢复外表面细薄碎边。现只恢复位于最小壁厚安全区域内的筋；底板边缘细条采用保持原轮孔边界的几何整形。',
        '10. 抬高的电池托盘可能在绑带孔上留下 1.1 mm 薄顶。现贯穿实际托盘高度，并同步切除冲突壳体；独立竖向穿孔回归在旧代码失败、修改后通过。',
        '11. 开关的手指开口可能切薄相邻托盘。现依据功能特征的实际变换和手指间隙预留开口占位，再让布局求解器选择组件位置。',
        '12. 最后添加的从动轮支架会重新遮挡局部绑带孔。现所有支架并入底盘后，再统一保持真实开口；独立结构回归覆盖此生成顺序。',
        '13. 连接底盘支架的梁若高出底盘面，会侵占分壳装配间隙。现限制连接梁位于底盘厚度内，保持实际支架连接，并单独测量其与上壳的距离。',
        '14. 上壳的离散重建可能侵入已经切出的绑带通道。现为通道预留边界误差；最终仍用真实孔体积对所有生成件作独立相交检查。',
        '15. 螺丝柱与壳体原料重叠 98% 不能保证完整圆柱恢复后的间隙。7.stl 在修正梁高后仍有 0.174039 mm 间隙，最近点定位到柱与从动轮支架；现闭合选位直接检查完整柱到实际底盘的距离。',
        '16. 支架下端恰在底盘面时，重叠体积为零仍可能有完整连接面。现以布尔边界面积差测量共享面，并验证连通性；点或边接触仍不接受。',
        '17. 固定数量的随机壁厚采样可能漏掉壳体与柱交界处的小薄片。现对连接区域内每个最终三角面的中心补充法向射线；保留原随机采样和最小壁厚要求。连接区域经截面及三维重建处理，新增和移除材料实测，最终重新检查几何和装配。',
        '18. 导出前的全局网格简化会在已经修好的细曲面上重新形成异常小面。现保留实际打印曲面，仅对预览副本作粗化；正式 STL 稳定化和几何改变量限制仍保留。倾斜墙反例包含旧简化负对照及新 STL 读回后的逐面壁厚验证。',
        '19. STL 的单精度坐标会把近重合底板接缝合并，或留下极小非零面积三角形。现只在实际编码检测失败时，以浮点坐标精度清理原双精度实体，再验证拓扑和增减体积；不再用粗网格简化处理所有打印面。',
        '20. 高斯平滑二值实体可能将邻近壁面连接成不足一个喷嘴宽的薄片。最终连接区域改为内部球心距离场重建，独立双柱反例检查间隙保持和逐面壁厚。该处理不能保证消除所有真实细颈，5.stl 的 v13 实测仍有残留，最终状态以本轮测量为准。',
        '21. 闭合孔深曾错误地从上壳接缝向上增加整个螺丝长度，遗漏螺丝实际从底盖底面安装。初次钻孔和重建后恢复现在共享实际承压面＋螺丝长度＋端部间隙的计算，孔径与啮合要求不变。',
        '22. 直接把薄顶盲孔改为贯穿孔，在斜壳出口处仍可能生成不合格薄边。v13 的实际失败被保留；修正真实螺丝基准后避免了不必要的贯穿。独立倾斜屋顶反例对加密但未改变几何的三角面测量，错误贯穿失败、正确盲孔通过。','',
        '## 本轮验收矩阵','',
        '| 模型 | STL 网格 | 轮位 | 接地/转动 | 碰撞/间隙 | 装配路径 | 固定/配合 | 访问 | 制造 | 上壳采样最小壁厚 mm | 壁厚射线数 |',
        '|---|---|---|---|---|---|---|---|---|---:|---:|']
    for v in verification:
        states=[v['groups'][key]['status'] for key in categories]
        lines.append('| '+v['model']+' | '+' | '.join(states)+f" | {fmt(v['body_sampled_minimum_wall_mm'],3)} | {v['body_wall_rays']} |")
    lines += ['', 'WARNING 保留原检查的限制，例如全局壁厚、自交、连接器资料和实物试装尚未确认；不能解释为已经可投产。MISSING 表示无对应证据。完整分组检查和测量保存在 verification_matrix.json / CSV。','',
        '## 复测尺寸与耗时','',
        '| 模型 | 初始车长 mm | 设计比例 | 打印上壳 X×Y×Z mm | 实际装配轴距 mm | 两排轮胎边缘间隙 mm | 后/前轮距 mm | 单例耗时 min |','|---|---:|---:|---|---:|---:|---|---:|']
    for model in manifest['models']:
        r=latest.get(model['index'],{});w=r.get('wheel_metrics',{});dims=r.get('printed_body_dimensions_mm')
        check=next((c for c in r.get('posthoc_axle_audit',{}).get('checks',[]) if c['check']=='axle_layout:wheelbase'),{})
        gap=check.get('measurements',{}).get('tire_edge_gap_mm')
        tracks=r.get('rear_front_tracks_mm',[])
        lines.append(f"| {model['source_entry']} | {fmt(r.get('input_dimensions_mm',[None,None,None])[1])} | {fmt(r.get('scale'))} | {' × '.join(fmt(v,1) for v in dims) if dims else '—'} | {fmt(w.get('wheelbase_mm'))} | {fmt(gap)} | {' / '.join(fmt(v,1) for v in tracks) if tracks else '—'} | {fmt(r['seconds']/60) if r.get('seconds') is not None else '—'} |")
    lines += ['', '耗时是在本机并行运行和诊断工作存在时测得；不是独占运行速度或下次必需时间的保证。放大比例是有界离散搜索的结果，不是全局最小尺寸证明。','',
        '## 全部运行记录','',
        '| 模型 | 批次 | 状态 | PASS / FAIL / WARNING | 秒 | 补充轴距审计失败项 | 加强壁厚复核 |','|---|---|---|---|---:|---|---|']
    for r in records:
        c=r.get('counts',{});audit=r.get('supplementary_wall_audit',{})
        wall=f"{audit['additional_status']} / {audit['measurements']['minimum_mm']:.3f} mm" if audit else '—'
        lines.append(f"| {r['model']} | {r['label']} | {r.get('run_status',{}).get('status')} | {c.get('PASS','—')} / {c.get('FAIL','—')} / {c.get('WARNING','—')} | {r.get('seconds','—')} | {'; '.join(r.get('posthoc_axle_failures',[]))} | {wall} |")
    if cancellations:
        lines += ['', '## 停止的过时计算','']
        for event in cancellations:lines.append(f"- {event['label']}：{event['timestamp']} 停止。已完成和失败的产物全部保留；被中止的个例标为 CANCELLED，不计入通过。后续批次的结果独立记录。")
    lines += ['', '## 尚未完成的工程验收','',
        '必须以各次正式运行的实际 FAIL 项和输出文件为准。本批测试期间报告会更新，未完成运行不被视为成功。',
        '轮毂轴向定位、从动轮轮毂内部接口、PCB 连接器语义和线缆路径仍缺少真实数据；装配路径和打印验证为几何模型检查，未做真实硬件装配或切片试印。全局自交、全局最小壁厚和结构强度也未获得完整证明。所有结果保持 release_ready=false。',
        '尺寸、源文件校验值、命令、逐例运行耗时、实际失败项和模型路径见 all_runs.json、all_runs.csv 以及各批次日志。','']
    if args.render:
        for model in manifest['models']:
            b=baseline.get(model['index']);r=latest.get(model['index'])
            brun=Path(b['run']) if b and (Path(b['run'])/'output/body.stl').exists() else None
            rrun=Path(r['run']) if r and (Path(r['run'])/'output/body.stl').exists() else None
            if brun is None and rrun is None:continue
            fig,axes=plt.subplots(2,2,figsize=(15,9))
            draw_run(brun,axes[:,0],'Baseline');draw_run(rrun,axes[:,1],r['label'] if r else 'Pending')
            fig.suptitle(model['source_entry']+' | Source appearance + generated geometry + physical wheel positions')
            fig.tight_layout();name=f'm{model["index"]:02d}-wheel-comparison.png';fig.savefig(out/name,dpi=130);plt.close(fig)
            lines += [f"### {model['source_entry']}",'',f'![轮位比较]({name})','']
        models=sorted(manifest['models'],key=lambda m:int(Path(m['source_entry']).stem) if Path(m['source_entry']).stem.isdigit() else m['index'])
        fig,axes=plt.subplots((len(models)+1)//2,2,figsize=(16,3.5*((len(models)+1)//2)),squeeze=False)
        for ax,model in zip(axes.flat,models):
            record=latest.get(model['index']);run=Path(record['run']) if record and (Path(record['run'])/'output/layout.json').exists() else None
            draw_run(run,[ax],model['source_entry'],include_source=False)
        for ax in list(axes.flat)[len(models):]:ax.axis('off')
        fig.suptitle('Final generated bodies and physical wheel poses | Red dashed: measured source axles | mm',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'wheel-layout-overview.png',dpi=140);plt.close(fig)
        lines += ['## 十模型轮位总览','','蓝色为生成外壳，黑色圆为真实轮胎代理，红线为实测源轴线；各图坐标单位为 mm。','',
                  '![最终轮位总览](wheel-layout-overview.png)','']
    entry=['# 成果入口','',f"批次：{args.label or '每例最近完成运行'}。所有结果为工程原型，release_ready=false。",'',
           '[执行摘要](执行摘要.md) · [批量分析](批量问题分析.md) · [逐项验收](verification_matrix.csv) · [机器可读结果](selected_results.json) · [全部运行 CSV](all_runs.csv) · [问题目录](issue_catalog.json)','',
           '| 模型 | 状态 | 预览 | 装配 GLB | 打印 3MF | 上壳 STL | 底盘 STL | 验收 |','|---|---|---|---|---|---|---|---|']
    for model in sorted(manifest['models'],key=lambda m:int(Path(m['source_entry']).stem) if Path(m['source_entry']).stem.isdigit() else m['index']):
        record=latest.get(model['index'])
        if not record or not (Path(record['run'])/'output/layout.json').exists():
            entry.append(f"| {model['source_entry']} | 未完成 | | | | | | |");continue
        run=Path(record['run']);prefix='../../../runs/'+run.name+'/output/'
        c=record.get('counts',{});state=f"{c.get('PASS')} / {c.get('FAIL')} / {c.get('WARNING')}"
        links=[f'[{title}]({prefix}{filename})' for title,filename in [('预览','assembly.png'),('装配','assembly.glb'),('3MF','printable.3mf'),('上壳','body.stl'),('底盘','bottom_cover.stl'),('报告','design_report.md')]]
        entry.append(f"| {model['source_entry']} | {state} | "+' | '.join(links)+' |')
    entry += ['', '状态数字依次为 PASS / FAIL / WARNING。每例目录还含两个电机压盖、爆炸装配、装配步骤、布局、BOM、精确刚性硬件 STEP 和紧固配合试片。',
              '五个无单位模型使用诊断初始车长 200 mm；实际制作尺寸尚未确认。完整接口、线束、强度和物理试装未完成。','']
    (out/'成果入口.md').write_text('\n'.join(entry),encoding='utf-8')
    completed=[v for v in verification if v['release_ready'] is not None]
    accepted=[r for r in latest.values() if r.get('counts',{}).get('FAIL')==0]
    wheel_ok=[v for v in completed if v['groups']['wheel_layout']['status']=='PASS']
    summary_lines=['# 十模型测试执行摘要','',f"统一批次：{args.label or '每例最近完成运行'}。已完成 {len(completed)}/{len(manifest['models'])} 个模型；{len(accepted)} 个没有检测到正式 FAIL；{len(wheel_ok)} 个轮位检查通过。所有结果仍为工程原型，不能直接视为量产或实物装配认证。",'',
        '原流程分别布置两排轮子，仅靠不碰撞约束会让轮子挤在一起。现在先从原始外观测量双侧轮位，再联合搜索前后轴；独立验证轴距、原轮拱对应、轮距、接地、旋转扫掠和装入路径。没有通过文件名、模型哈希或手写坐标选择轮心。','',
        '| 模型 | 轴距 mm | 两排轮胎边缘间隔 mm | 轴距/车长 | 原轴线最大偏移 mm | 正式 FAIL | 上壳壁厚采样最小值 mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for model in manifest['models']:
        r=latest.get(model['index'],{});w=r.get('wheel_metrics',{})
        c=next((c for c in r.get('posthoc_axle_audit',{}).get('checks',[]) if c['check']=='axle_layout:wheelbase'),{})
        v=next(v for v in verification if v['model']==model['source_entry'])
        summary_lines.append(f"| {model['source_entry']} | {fmt(w.get('wheelbase_mm'))} | {fmt(c.get('measurements',{}).get('tire_edge_gap_mm'))} | {fmt(w.get('wheelbase_to_input_length'),3)} | {fmt(r.get('source_axle_max_error_mm'))} | {r.get('counts',{}).get('FAIL','—')} | {fmt(v['body_sampled_minimum_wall_mm'],3)} |")
    summary_lines += ['', '毫米值均来自实际生成模型和正式测量。最小壁厚要求为 1.6 mm，表内为射线采样最小值，不能当作全局下界。源轴线误差不是对语义正确性的保证，外观对照仍需要审查。','',
        '## 本轮残留硬失败','']
    residual_count=0
    for model in manifest['models']:
        r=latest.get(model['index'])
        if not r:continue
        path=Path(r['run'])/'output/validation_report.json'
        if not path.exists():continue
        failed=[c for c in read_json(path)['checks'] if c['status']=='FAIL']
        for check in failed:
            residual_count+=1;measurement=check.get('measurements',{})
            if check['check'].startswith('wall_thickness:'):
                detail=f"采样最小 {fmt(measurement.get('minimum_mm'),4)} mm，要求 {fmt(check.get('required_mm'))} mm，{measurement.get('below_minimum')} 条射线低于要求。"
            elif 'degenerate_triangles' in measurement:
                detail=f"退化三角面 {measurement['degenerate_triangles']}，非流形边 {measurement.get('nonmanifold_edges')}。"
            else:detail=check.get('reason','完整测量见原验收报告。')
            link='../../../runs/'+Path(r['run']).name+'/output/validation_report.json'
            summary_lines.append(f"- **{model['source_entry']} / {check['check']}**：{detail}[原报告]({link})")
    if not residual_count:summary_lines.append('已完成的模型没有检测到正式 FAIL；未完成模型不计入此结论，WARNING 和实物验收限制仍保留。')
    summary_lines += ['',
        '已处理的共性问题还包括：输入单位不明和空体素体积、融合轮胎坐标识别、轮拱影响托盘壁厚、悬空安装柱、闭合柱间隙、绑带/开关开口被支架堵住、局部薄片、盲孔薄顶及 STL 单精度拓扑问题。具体失败证据、正式修复及旧版反例见[批量分析](批量问题分析.md)与[回归证据](../../../docs/REGRESSION_EVIDENCE.md)。','',
        '五个无单位的小尺寸文件（1、2、6、8、9）采用初始车长 200 mm 的诊断假设，尚未确认成品尺寸。固定轮胎、电机、PCB、电池和开关保持真实毫米尺寸。不能承诺任意形状都适配；三轴车、履带、极端不对称或开放骨架仍需要新的设计政策和算法。','',
        '尚未完成：真实轮毂及轴向限位、连接器和线缆语义、打印机孔配合、物理夹持/强度、全局自交与壁厚证明、切片试印和硬件试装。3MF 是模型，不是已经配置支撑的切片文件。','',
        '各模型的 STL、3MF、装配/爆炸 GLB、布局、装配步骤、BOM 与报告链接集中在[成果入口](成果入口.md)；下一模型和批量运行命令见[使用说明](../../../docs/WHEEL_LAYOUT_BATCH.md)。硬件数据缺口和实物验收顺序见[后续验收](../../../docs/后续验收与输入补全.md)。','',
        '## 实测运行时间','']
    if timings:
        t=timing_report['all_completed']
        summary_lines.append(f"已完成 {len(timings)} 例，单例最短 {t['minimum_seconds']/60:.1f} 分钟、中位数 {t['median_seconds']/60:.1f} 分钟、最长 {t['maximum_seconds']/60:.1f} 分钟。各例计时包含该例内部的布局和结构重试、验证与导出。")
        summary_lines.append('')
        for key,label in [('single_geometry_evaluation','一次结构评估'),('multiple_geometry_evaluations','多次结构评估')]:
            t=timing_report[key]
            if t['models']:summary_lines.append(f"- {label}：{t['models']} 例，中位数 {t['median_seconds']/60:.1f} 分钟。")
    else:summary_lines.append('当前统一批次尚无完成单例，不能给出实测运行时间。')
    summary_lines += ['', '这些是在本机两个模型并行、同时存在诊断工作的条件下测得，不能保证下次耗时。单例时间相加或除以二也不是整批实际墙钟时间。详细数据见 [runtime_summary.json](runtime_summary.json) / [CSV](runtime_summary.csv)。','']
    if args.render:summary_lines += ['![十模型轮位总览](wheel-layout-overview.png)','']
    (out/'执行摘要.md').write_text('\n'.join(summary_lines),encoding='utf-8')
    (out/'批量问题分析.md').write_text('\n'.join(lines),encoding='utf-8')
    print(str(out),flush=True)


if __name__=='__main__':main()
