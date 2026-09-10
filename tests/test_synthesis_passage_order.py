from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.assembly import placed
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,to_mesh
from smartcar.structure import synthesis
from smartcar.validation.battery_access import validate_battery_passages


def test_passive_bracket_added_later_keeps_real_strap_path(tmp_path,monkeypatch):
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    battery=SimpleNamespace(role='battery',bounding_box=np.array([[-8.,-25,-8],[8,25,8]]),features={},raw={})
    wheel=SimpleNamespace(role='passive_wheel',bounding_box=np.array([[-6.5,-18.5,-18.5],[6.5,18.5,18.5]]),features={},
        raw={'rigid_body':{'nominal_diameter_mm':37.,'nominal_width_mm':13.}})
    instances=[placed(battery,np.eye(3),[18,0,15.4],'battery','flat'),placed(wheel,np.eye(3),[42,0,18.5],'passive','axial')]
    for name in ['09_structure','05_wheel_candidates','10_openings']:(tmp_path/name).mkdir()
    # This regression exercises real cavity, tray, axle and final Boolean
    # synthesis. Upper-shell voxel smoothing is irrelevant to this bottom cut.
    monkeypatch.setattr(synthesis,'regularize_shell',lambda body,*args:(body,{}))
    vehicle=to_mesh(box([[-50,-80,0],[50,80,55]]))
    parts,records=synthesis.synthesize(vehicle,instances,5.,p,tmp_path)
    checks=validate_battery_passages(parts,records['battery'],p)
    assert all(c['status']=='PASS' for c in checks),checks
    assert len(parts['bottom_cover'].decompose())==1
