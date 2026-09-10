from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,cylinder,to_mesh
from smartcar.structure.closure_bores import closure_bores,closure_pilot_end_z


def test_blind_bore_with_thin_cap_is_extended_through_material():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    seam=-p.sliding_clearance
    end=closure_pilot_end_z(seam,p)
    body=cylinder(4,0,end+.5)
    cuts,records=closure_bores(body,dict(seam_z=seam,centers_xy=[[0,0]]),p)
    assert records[0]['type']=='through'
    assert abs(records[0]['measured_minimum_vertical_cap_mm']-.5)<p.numerical_tolerance
    result=to_mesh(body-cuts)
    assert not result.contains([[0,0,end+.25]])[0]
    assert result.contains([[3,0,end+.25]])[0]
    assert result.is_watertight


def test_sufficient_blind_cap_is_retained():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    end=closure_pilot_end_z(-p.sliding_clearance,p)
    body=box([[-5,-5,0],[5,5,end+3]])
    cuts,records=closure_bores(body,dict(seam_z=-p.sliding_clearance,centers_xy=[[0,0]]),p)
    assert records[0]['type']=='blind_or_ends_in_existing_void'
    assert to_mesh(body-cuts).contains([[0,0,end+1]])[0]


def test_pilot_uses_actual_fastener_bearing_datum_and_tip_clearance():
    from smartcar.structure.fasteners import fastener_records
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    seam=8.7
    closure=dict(centers_xy=[[1,2]],seam_z=seam,engagement_mm=p.screw_fit['minimum_engagement'])
    screw=fastener_records(dict(closure=closure,pcb=[],switch=[],motor=[]),p)[0]
    actual_tip=screw['head_bearing_center'][2]+screw['thread_length_mm']
    assert abs(closure_pilot_end_z(seam,p)-actual_tip-p.rigid_clearance-p.numerical_tolerance)<1e-10


def test_correct_depth_preserves_inclined_roof_instead_of_unnecessary_through_hole():
    import itertools
    import numpy as np
    import trimesh
    from smartcar.geometry.solid import from_mesh
    from smartcar.validation.geometry import wall_measurements
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    vertices=[[x,y,z] for x,y in itertools.product([-8,8],repeat=2) for z in [0,7.1+np.tan(np.deg2rad(20))*x]]
    body=from_mesh(trimesh.convex.convex_hull(vertices))
    closure=dict(seam_z=-p.sliding_clearance,centers_xy=[[0,0]])
    cuts,records=closure_bores(body,closure,p)
    assert records[0]['type']=='blind_or_ends_in_existing_void'
    final=to_mesh(body-cuts)
    region=np.array([[-3,-3,-1],[3,3,12]])
    # Subdivision leaves the analytic planes unchanged and covers small facets
    # beside the bore, as in the full pipeline's reconstructed curved shell.
    for _ in range(3):final=final.subdivide()
    assert wall_measurements(final,p,samples=0,regions=[region])['below_minimum']==0
    # The old depth datum would falsely request a through hole. An inclined
    # exterior intersecting that bore creates short material chords at the rim.
    wrong=to_mesh(body-cylinder(p.screw_fit['vertical_pilot']/2,-1,12))
    for _ in range(3):wrong=wrong.subdivide()
    assert wall_measurements(wrong,p,samples=0,regions=[region])['below_minimum']>0
