from types import SimpleNamespace
from pathlib import Path
import numpy as np
from smartcar.layout.hardware_candidates import generate_candidates
from smartcar.domain.assembly import placed
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.layout.reservations import structural_bounds
from smartcar.geometry.collision import overlaps


def test_common_tray_rejects_battery_stack_over_motor():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/"config/manufacturing.json")
    battery=SimpleNamespace(role="battery",bounding_box=np.array([[-14,-25,-8],[14,25,8]]),raw={},features={})
    motor=SimpleNamespace(role="drive_unit",bounding_box=np.array([[-9,-27,-11],[9,27,11]]),raw={},features={})
    anchor=placed(motor,np.eye(3),[20,0,16],"motor","fixed")
    class FakeVolume:
        pitch=.6
        def feasible_centers(self,*a,**k):return np.array([[20,0,40]]),np.array([10.])
    vehicle=SimpleNamespace(bounds=np.array([[-50,-100,0],[50,100,80]]),extents=np.array([100,200,80]))
    cs,_=generate_candidates(battery,FakeVolume(),vehicle,[anchor],3,p,"spatial_free")
    assert not cs, "a tray support would fill the motor even if the battery itself is above it"


def test_motor_screw_post_is_reserved_outside_rigid_body():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/"config/manufacturing.json")
    motor=SimpleNamespace(role="drive_unit",bounding_box=np.array([[-9,-27,-11],[9,27,11]]),raw={},features={})
    inst=placed(motor,np.eye(3),[25,0,20],"drive","fixed")
    b=structural_bounds(inst,p)
    required=p.rigid_clearance+2*p.standoff_radius
    assert inst.bounds[0,0]-b[0,0]>=required-1e-9
