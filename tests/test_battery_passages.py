from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,to_mesh
from smartcar.structure.battery_tray import battery_tray
from smartcar.validation.geometry import wall_measurements
from smartcar.validation.battery_access import validate_battery_passages


def test_low_battery_platform_keeps_strap_passages_open_after_chassis_union():
    profile=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=profile.bottom_thickness
    # A legitimate low pose leaves a sub-minimum unsupported membrane if the
    # tray is added AFTER cutting slots in the underlying base plate.
    bottom=floor+profile.nozzle_mm*2
    battery=SimpleNamespace(id='test_battery',bounds=np.array([[-10.,-24,bottom],[10,24,bottom+12]]))
    tray,slots,record=battery_tray(battery,floor,profile)
    chassis=box([[-30,-50,0],[30,50,floor]])
    assembled=(chassis-slots)+tray
    assert (assembled^slots).volume()<profile.collision_volume_tolerance
    # The openings must not be obtained by erasing their surrounding tray.
    assert assembled.volume()>chassis.volume()
    measured=wall_measurements(to_mesh(assembled),profile)
    assert measured['below_minimum']==0
    assert all(c['status']=='PASS' for c in validate_battery_passages({'bottom_cover':assembled},[record],profile))
    for bound in record['strap_passage_bounds']:
        assert np.isclose(bound[1,1]-bound[0,1],profile.battery_strap_width+2*profile.sliding_clearance)


def test_final_strap_validator_catches_a_later_part_covering_a_passage():
    profile=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    battery=SimpleNamespace(id='test_battery',bounds=np.array([[-10.,-24,3.2],[10,24,15.2]]))
    tray,_,record=battery_tray(battery,2.4,profile)
    membrane=np.array(record['strap_passage_bounds'][0]).copy();membrane[:,2]=[2.5,2.8]
    checks=validate_battery_passages({'tray':tray,'obstructing_support':box(membrane)},[record],profile)
    assert checks[0]['status']=='FAIL' and checks[0]['intersection_mm3']>1
    assert checks[1]['status']=='PASS'
