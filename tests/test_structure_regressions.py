from pathlib import Path
import numpy as np
import trimesh
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import cylinder,to_mesh,box


def test_wheel_cutter_polygon_contains_required_radial_clearance():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/"config/manufacturing.json")
    wheel=cylinder(18.5,-6.5,6.5,(0,0),axis=0)
    radius=(18.5+p.moving_clearance+p.mesh_tolerance)/np.cos(np.pi/64)
    surrounding=box([[-10,-24,-24],[10,24,24]])-cylinder(radius,-8,8,(0,0),axis=0)
    assert wheel.min_gap(surrounding,2)>=p.moving_clearance


def test_boolean_print_mesh_preserves_watertight_index_topology(tmp_path):
    a=box([[0,0,0],[30,50,2.4]])+cylinder(3.2,2.3,15,(10,10))
    a=a-cylinder(1.35,8,16,(10,10))
    m=to_mesh(a.simplify(.01))
    assert m.is_watertight
    p=tmp_path/'part.stl';m.export(p)
    assert trimesh.load(p,force='mesh').is_watertight


def test_open_bottom_wheel_arch_has_constant_plate_cutout():
    from smartcar.structure.wheel_arch import wheel_arch
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    cut=wheel_arch([20,0,18.5],18.5,13,-1,p)
    # A circular-only cut varies across bottom plate height and leaves wedges.
    assert abs(cut.slice(3.).area()-cut.slice(5.4).area())<1e-6


def test_float32_collapsed_face_pair_cancels_without_changing_solid():
    from smartcar.geometry.solid import cancel_collapsed_faces
    cube=trimesh.creation.box([10,10,3])
    verts=np.vstack([cube.vertices,[[5,0,-1.5],[5,0,1.5],[5,1,1.5]]])
    faces=np.vstack([cube.faces,[[8,9,10],[10,9,8]]])
    repaired=cancel_collapsed_faces(trimesh.Trimesh(verts,faces,process=False))
    assert repaired.is_watertight
    assert len(repaired.faces)==len(cube.faces)
    assert abs(repaired.volume-cube.volume)<1e-8


def test_wheel_candidate_reserve_preserves_inset_chassis_end_web():
    from smartcar.geometry.wheel import wheel_end_margin
    from smartcar.structure.wheel_arch import wheel_arch
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    radius=18.5;edge=55.;y=edge-radius-wheel_end_margin(radius,p)
    plate=box([[-30,-edge,0],[30,edge-p.sliding_clearance,p.bottom_thickness]])
    cut=wheel_arch([25,y,radius],radius,13,-1,p)
    remaining=plate-cut
    web=box([[24,edge-p.sliding_clearance-p.nominal_wall,.1],[29,edge-p.sliding_clearance,p.bottom_thickness-.1]])
    assert abs((web-remaining).volume())<p.collision_volume_tolerance
