from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,union,intersection_volume
from smartcar.structure.battery_tray import battery_tray
from smartcar.structure.opening import reconstruction_aperture
from smartcar.structure.regularize import regularize_shell


def test_real_strap_volume_survives_shell_boundary_reconstruction():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=p.bottom_thickness
    shell=box([[-25,-20,floor+p.sliding_clearance],[25,20,20]])-box([[-22.6,-17.6,2],[22.6,17.6,21]])
    closure=dict(seam_z=floor,centers_xy=[])
    obstructed=[]
    for phase in [0.,.1,.2]:
        battery=SimpleNamespace(id='battery',bounds=np.array([[-12+phase,1.5,4],[12+phase,17.5,16.]]))
        _,slots,record=battery_tray(battery,floor,p)
        old,_=regularize_shell(shell-slots,closure,p)
        obstructed.append(intersection_volume(old,slots))
        reserved=union([reconstruction_aperture(b,p) for b in record['strap_passage_bounds']])
        final,_=regularize_shell(shell-reserved,closure,p)
        assert intersection_volume(final,slots)<p.collision_volume_tolerance
    # The negative control has the same physical opening and reconstruction;
    # without boundary reserve its original exact cut is not sufficient.
    assert max(obstructed)>p.collision_volume_tolerance,obstructed
