from pathlib import Path
import numpy as np
import trimesh
from smartcar.domain.hardware import fastener_datum
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.structure.fasteners import fastener_records,head_solid,upper_body_head_pockets
from smartcar.geometry.solid import box,intersection_volume


def test_fastener_datum_ignores_arbitrary_step_origin():
    origin=np.array([317.,-46.,99.]);axis=np.array([0.,1.,0.])
    m=trimesh.creation.box([4.7,7.5,4.7]);m.apply_translation(origin+axis*2.25)
    raw={"fastener":{"thread_diameter_mm":2.5,"thread_length_mm":6.,"head_height_mm":1.5}}
    features=dict(cylinders=[dict(point=origin-axis*1.5,axis=-axis,radius=1.25)],planes=[dict(center=origin,normal=axis)])
    result=fastener_datum(raw,features,m)
    assert np.allclose(result['head_bearing_center'],origin)
    assert np.allclose(result['thread_axis'],axis)


def test_retainer_head_is_reserved_beyond_the_cap_surface():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    records=dict(closure=dict(centers_xy=[]),pcb=[],switch=[],motor=[dict(hardware='drive',screws=[dict(center_xy=[3,7],cap_top_z=30,engagement_mm=4)])])
    screws=fastener_records(records,p)
    head=head_solid(screws[0]);solid=box([[-10,-10,6],[20,20,40]])
    assert intersection_volume(head,solid)>0
    reserved=solid-upper_body_head_pockets(screws,6,p)
    assert intersection_volume(head,reserved)<p.collision_volume_tolerance
    assert head.min_gap(reserved,2)>=p.rigid_clearance
