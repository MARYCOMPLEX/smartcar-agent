from pathlib import Path
from types import SimpleNamespace
import numpy as np
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.sdf import DesignableVolume
from smartcar.geometry.solid import box,cylinder
from smartcar.geometry.wheel import wheel_side_exposure
from smartcar.structure.wheel_arch import wheel_arch
from smartcar.validation.wheel_access import validate_wheel_access


def profile():
    return ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')


def test_shallow_side_aperture_is_distinct_from_a_deeply_buried_tire():
    p=profile()
    mask=np.zeros((190,130,120),bool);mask[5:185,5:125,5:115]=True
    volume=DesignableVolume(mask,[-47.5,-32.5,-2.5],.5,p.nominal_wall)
    shallow=wheel_side_exposure(volume,[37.5,0,25],18.5,13,1,maximum_recess=p.nominal_wall)
    deep=wheel_side_exposure(volume,[30,0,25],18.5,13,1,maximum_recess=p.nominal_wall)
    assert shallow['initially_exposed_fraction']==0
    assert shallow['exposed_fraction']==1
    assert deep['exposed_fraction']==0


def recessed_case():
    p=profile();bounds=np.array([[-45.,-60,0],[45,60,65]])
    body=box(bounds);c=np.array([37.,0,25.])
    wheel=SimpleNamespace(id='test_wheel',bounds=np.array([c-[6.5,18.5,18.5],c+[6.5,18.5,18.5]]),
                          definition=SimpleNamespace(role='drive_wheel',raw={'rigid_body':{'nominal_diameter_mm':37,'nominal_width_mm':13}}))
    return p,bounds,body,c,wheel


def test_outward_arch_opens_actual_axial_insertion_sweep():
    p,bounds,body,c,_=recessed_case()
    old=body-wheel_arch(c,18.5,13,-1,p)
    opened=body-wheel_arch(c,18.5,13,-1,p,exterior_bounds=bounds)
    sweep=cylinder(18.5,c[0]-6.5,bounds[1,0]+20,c[1:],axis=0)
    assert (old^sweep).volume()>1
    assert (opened^sweep).volume()<p.collision_volume_tolerance


def test_final_access_validator_detects_a_retained_outer_membrane():
    p,bounds,body,c,wheel=recessed_case()
    old=body-wheel_arch(c,18.5,13,-1,p)
    opened=body-wheel_arch(c,18.5,13,-1,p,exterior_bounds=bounds)
    assert validate_wheel_access({'body':old},[wheel],p)[0]['status']=='FAIL'
    result=validate_wheel_access({'body':opened},[wheel],p)[0]
    assert result['status']=='PASS' and result['blocked_rays']==0
