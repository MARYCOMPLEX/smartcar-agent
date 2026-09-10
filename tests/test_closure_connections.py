from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,to_mesh,intersection_volume
from smartcar.geometry.sdf import resample_designable_volume
from smartcar.structure.lightweight import hollow_body
from smartcar.structure.regularize import regularize_shell


def test_closure_rib_reaches_cut_shell_instead_of_source_opening(tmp_path):
    profile=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    source=box([[-30,-40,0],[30,40,30]])
    vehicle=to_mesh(source)
    volume,_=resample_designable_volume(vehicle,.6,profile.nominal_wall)
    floor=profile.bottom_thickness
    body=source^box([[-31,-41,floor+profile.sliding_clearance],[31,41,31]])
    opening=box([[23,-8,0],[35,8,13]])
    # Formal synthesis reserves shell reconstruction uncertainty around access
    # cutters; the final physical opening itself must remain completely clear.
    reserve=3**.5*profile.shell_regularization_pitch/2
    body=body-box([[23-reserve,-8-reserve,-reserve],[35+reserve,8+reserve,13+reserve]])
    closure=dict(centers_xy=[[18.5,0]],seam_z=floor)
    hollow,record=hollow_body(body,vehicle,volume,closure,profile)
    final,_=regularize_shell(hollow,closure,profile)
    assert len(final.decompose())==1
    assert intersection_volume(final,opening)<profile.collision_volume_tolerance
    assert to_mesh(final).is_watertight
    # A valid indexed Boolean boundary must also survive the physical STL
    # coordinate representation without a repair that fills the cavity.
    import trimesh
    from smartcar.geometry.print_mesh import stabilize_print_solid
    stable,repair=stabilize_print_solid(final,profile)
    assert repair.get('status','PASS')!='FAIL'
    path=tmp_path/'closure.stl';to_mesh(stable).export(path)
    exported=trimesh.load(path,force='mesh')
    assert exported.is_watertight and exported.is_winding_consistent
    assert exported.nondegenerate_faces().all()
    assert not exported.contains([[0,0,15]])[0]
