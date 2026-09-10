from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.assembly import placed
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.layout.solver import solve_layout
from smartcar.structure.battery_tray import battery_tray
from smartcar.geometry.solid import box, intersection_volume


def test_finger_aperture_cannot_shave_neighbouring_tray_wall():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    switch=SimpleNamespace(role='power_switch',bounding_box=np.array([[-10.,-8,0],[10,8,14]]),features={},
        raw={'functional_features':[{'id':'actuator','external_access_required':True,'local_bounds_xyz_mm':[[-3,-4,0],[3,8,12]]}]})
    battery=SimpleNamespace(role='battery',bounding_box=np.array([[-15.,-10,0],[15,10,16]]),features={},raw={})
    sw=placed(switch,np.eye(3),[0,0,12],'switch','down')
    # Body gap 4.4 mm satisfies the former pair rule, but a 3 mm finger
    # extension physically removes 1.2 mm from the tray's retaining edge.
    bad=placed(battery,np.eye(3),[0,22.4,14],'battery','flat')
    good=placed(battery,np.eye(3),[0,25,14],'battery','flat')
    sw.score={};bad.score={};good.score={'height':1.}
    actual_opening=box([[-6,-7,-1],[6,11,6]])
    bad_tray,_,_=battery_tray(bad,2.4,p)
    good_tray,_,_=battery_tray(good,2.4,p)
    assert intersection_volume(bad_tray,actual_opening)>1
    assert intersection_volume(good_tray,actual_opening)<p.collision_volume_tolerance
    selected,info=solve_layout({'switch':[sw],'battery':[bad,good]},p)
    assert selected is not None,info
    assert selected[1] is good
