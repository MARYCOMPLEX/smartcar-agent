"""Persist native worker failures which cannot reach Python exception handlers."""
import json
from smartcar.io import write_json


def record_worker_exit(run,returncode):
    path=run/'run_status.json'
    if returncode==0 or not path.exists():return
    previous=json.loads(path.read_text(encoding='utf-8-sig'))
    if previous.get('status')=='ERROR':
        previous['worker_exit_code']=returncode;write_json(path,previous);return
    write_json(path,dict(status='ERROR',error_type='WorkerProcessExit',worker_exit_code=returncode,
                         previous_status=previous.get('status'),release_ready=False,
                         error='Worker ended without a successful process exit; inspect execution.log and retained intermediate evidence.'))
