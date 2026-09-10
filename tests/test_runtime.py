import json
from smartcar.agent.runtime import record_worker_exit


def test_native_worker_exit_cannot_leave_false_running_status(tmp_path):
    p=tmp_path/'run_status.json';p.write_text(json.dumps(dict(status='RUNNING',pid=123,release_ready=False)))
    record_worker_exit(tmp_path,-1073741819)
    result=json.loads(p.read_text());assert result['status']=='ERROR'
    assert result['worker_exit_code']==-1073741819
    assert result['previous_status']=='RUNNING'


def test_python_error_detail_is_retained_after_worker_exit(tmp_path):
    p=tmp_path/'run_status.json';p.write_text(json.dumps(dict(status='ERROR',error='diagnosed failure')))
    record_worker_exit(tmp_path,1)
    assert json.loads(p.read_text())['error']=='diagnosed failure'
