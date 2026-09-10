from types import SimpleNamespace
import numpy as np
from smartcar.layout.wheel_candidates import drive_instances,rotation_for_shaft


def test_all_discrete_motor_rolls_preserve_the_fixed_shaft_and_wheel_interface():
    motor=SimpleNamespace(role='drive_unit',bounding_box=np.array([[-11,-3,-32],[11,15,22]]),
                          features={'shaft':dict(axis=[0,1,0],tip=[0,23,11],engagement=4.7)})
    wheel=SimpleNamespace(role='drive_wheel',bounding_box=np.array([[-6.5,-18.5,-18.5],[6.5,18.5,18.5]]),
                          raw={'rigid_body':{'nominal_width_mm':13}},features={})
    for axis in [1,2]:
        for sign in [-1,1]:
            for side in [-1,1]:
                center=np.array([side*40,20,18.5])
                inst,wh=drive_instances(motor,wheel,center,sign,side,axis)
                r=inst.transform[:3,:3]
                tip=r@motor.features['shaft']['tip']+inst.transform[:3,3]
                direction=r@motor.features['shaft']['axis']
                assert np.allclose(r.T@r,np.eye(3)) and np.isclose(np.linalg.det(r),1)
                assert np.allclose(direction,[side,0,0])
                assert np.allclose(tip-center,-direction*(13/2-4.7))
                assert np.allclose(wh.bounds.mean(0),center)
                expected=np.array([18,54,22] if axis==1 else [18,22,54])
                assert np.allclose(np.ptp(inst.bounds,axis=0),expected)


def test_search_finds_upright_drive_when_longitudinal_motor_cannot_fit():
    from pathlib import Path
    import trimesh
    from smartcar.domain.manufacturing import ManufacturingProfile
    from smartcar.geometry.sdf import DesignableVolume
    from smartcar.geometry.solid import cylinder,to_mesh
    from smartcar.layout.wheel_candidates import wheel_candidates
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    motor=SimpleNamespace(role='drive_unit',bounding_box=np.array([[-11.,-3,-32],[11,15,22]]),
                          features={'shaft':dict(axis=[0,1,0],tip=[0,23,11],engagement=4.7)})
    wheel=SimpleNamespace(role='drive_wheel',bounding_box=np.array([[-6.5,-18.5,-18.5],[6.5,18.5,18.5]]),
                          raw={'rigid_body':{'nominal_width_mm':13,'nominal_diameter_mm':37}},
                          mesh=to_mesh(cylinder(18.5,-6.5,6.5,axis=0)),features={})
    # A short, tall internal region surrounded by a larger exterior boundary.
    # This describes a cabin-like volume without using any vehicle file/pose.
    mask=np.zeros((75,95,90),bool);mask[3:72,35:70,3:87]=True
    volume=DesignableVolume(mask,[-37,-47,-3],1.,p.nominal_wall)
    vehicle=trimesh.creation.box([70,90,85]);vehicle.apply_translation([0,0,42.5])
    diagnostics={}
    candidates,rows=wheel_candidates(volume,vehicle,{'drive_unit':motor,'drive_wheel':wheel},p,2.4,diagnostics)
    assert candidates and rows
    assert all(r['long_axis']==2 for r in rows)
    horizontal=[r for r in diagnostics['orientations'] if r['long_axis']==1]
    assert all(r['inside_centers_before_floor']==0 for r in horizontal)
    for record in diagnostics['orientations']:
        rejected=sum(v for k,v in record.items() if k.endswith('_rejected'))
        assert record['sampled_centers']==rejected+record['accepted']


def test_high_wide_fenders_do_not_forbid_exposed_wheels_at_narrow_lower_body():
    from pathlib import Path
    from smartcar.domain.manufacturing import ManufacturingProfile
    from smartcar.geometry.sdf import DesignableVolume
    from smartcar.geometry.solid import cylinder,to_mesh
    from smartcar.layout.wheel_candidates import wheel_candidates
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    motor=SimpleNamespace(role='drive_unit',bounding_box=np.array([[-11.,-3,-32],[11,15,22]]),
                          features={'shaft':dict(axis=[0,1,0],tip=[0,23,11],engagement=4.7)})
    wheel=SimpleNamespace(role='drive_wheel',bounding_box=np.array([[-6.5,-18.5,-18.5],[6.5,18.5,18.5]]),
                          raw={'rigid_body':{'nominal_width_mm':13,'nominal_diameter_mm':37}},
                          mesh=to_mesh(cylinder(18.5,-6.5,6.5,axis=0)),features={})
    mask=np.zeros((145,185,85),bool);mask[32:113,2:183,2:73]=True;mask[2:143,32:153,62:83]=True
    volume=DesignableVolume(mask,[-72,-92,-2],1.,p.nominal_wall)
    vehicle=SimpleNamespace(bounds=np.array([[-70.,-90,0],[70,90,80]]),extents=np.array([140.,180,80]))
    candidates,rows=wheel_candidates(volume,vehicle,{'drive_unit':motor,'drive_wheel':wheel},p,2.4)
    assert candidates
    assert all(r['center'][0]<vehicle.bounds[1,0]-.65*13 for r in rows)
    assert all(r['score']['side_exposure_fraction']>=p.minimum_wheel_exposed_fraction for r in rows)
