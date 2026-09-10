"""Package source, immutable input data and completed runs without modifying them."""
from pathlib import Path
import argparse,hashlib,json,zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+');parser.add_argument('--name',default='smartcar-agent-v1-review.zip')
    parser.add_argument('--suite',type=Path,action='append',default=[],help='Include a benchmark corpus, its reports, and compact evidence from superseded runs.')
    args=parser.parse_args()
    names=[Path(name) for name in args.runs]
    required=['body.stl','bottom_cover.stl','assembly.glb','exploded_assembly.glb','layout.json','assembly_plan.json','validation_report.json','bom.json','design_report.md']
    run_paths=[];summaries=[]
    for name in names:
        run=(root/'runs'/name).resolve()
        if not run.is_relative_to(root/'runs'):raise ValueError('Run outside project')
        status=json.loads((run/'run_status.json').read_text(encoding='utf-8-sig'))
        if status['status'] not in ['FAIL','WARNING']:raise ValueError(f'Run not a completed design: {name}')
        for item in required:
            if not (run/'output'/item).is_file():raise ValueError(f'Missing artifact: {name}/{item}')
        report=json.loads((run/'output/validation_report.json').read_text())
        summaries.append(dict(run=str(name),status=report['status'],counts=report['counts'],release_ready=report['release_ready']))
        run_paths.append(run)
    folders=[root/p for p in ['src','config','tests','tools','docs','reports','benchmarks/inputs','source_bundle/smartcar-agent-foundation-v1/data']]+run_paths
    files=[root/p for p in ['README.md','CLAUDE.md','pyproject.toml','requirements-lock.txt','.gitignore']]
    for suite in args.suite:
        suite=suite.resolve()
        if not suite.is_relative_to(root/'benchmarks'):raise ValueError('Suite outside project benchmarks')
        if not (suite/'manifest.json').is_file():raise ValueError('Suite manifest missing')
        folders.append(suite)
        # Keep original failure evidence reviewable without duplicating every
        # obsolete heavy mesh. Completed selected runs above are included fully.
        for summary_path in (suite/'batches').glob('*/summary.json'):
            records=json.loads(summary_path.read_text(encoding='utf-8-sig'))['models']
            evidence_runs={root/'runs'/Path(record['run'].replace('\\','/')).name for record in records}
            # Cancelled native failures may never have reached the batch's
            # completed-case summary. Their original JSON/source/log evidence
            # still belongs in the review package.
            evidence_runs.update(p for p in (root/'runs').glob(f'{suite.name}-{summary_path.parent.name}-m*') if p.is_dir())
            for old in sorted(evidence_runs):
                if not old.is_relative_to(root/'runs'):raise ValueError('Evidence run outside project')
                if old in run_paths:continue
                for sub in ['01_input/source_snapshot','03_coordinate_frame']:
                    for f in (old/sub).rglob('*'):
                        if f.is_file() and f.suffix in ['.json','.py']:files.append(f)
                files.extend(f for f in old.rglob('*.json') if 'source_snapshot' not in f.parts)
                files.extend(f for f in old.glob('*.log'))
                files.extend(f for f in (old/'output').glob('*.md'))
    for folder in folders:
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts)
                     and not p.is_relative_to(root/'docs/history'))
    dest=root/'deliverables'/args.name
    if dest.parent.resolve()!=(root/'deliverables').resolve():raise ValueError('Archive name must be a filename')
    if dest.exists():raise ValueError('Delivery archive already exists')
    dest.parent.mkdir(exist_ok=True)
    digests={}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(set(files)):
            relative=file.relative_to(root).as_posix();archive.write(file,'smartcar-agent/'+relative)
            digests[relative]=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()
        archive.writestr('smartcar-agent/delivery_manifest.json',json.dumps(dict(runs=summaries,files_sha256=digests),indent=2))
    with zipfile.ZipFile(dest) as archive:
        bad=archive.testzip()
        if bad:raise ValueError('Archive CRC failed: '+bad)
    with dest.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    dest.with_suffix(dest.suffix+'.sha256').write_text(digest+'  '+dest.name+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,files=len(digests),sha256=digest,runs=summaries),indent=2))


if __name__=='__main__':main()
