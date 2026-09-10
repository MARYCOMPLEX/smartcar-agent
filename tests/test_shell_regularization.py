from pathlib import Path
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.solid import box,to_mesh
from smartcar.structure.regularize import regularize_shell


def test_occupancy_contour_keeps_exact_lattice_before_physical_transform():
    import numpy as np
    from smartcar.geometry.repair import occupancy_mesh
    mask=np.ones((17,131,9),dtype=bool)
    origin=np.array([-43.527,-91.813,5.947]);pitch=.3
    mesh=occupancy_mesh(mask,origin,pitch)
    grid=(mesh.vertices-origin)/pitch
    assert np.max(np.abs(grid*2-np.round(grid*2)))<1e-10
    assert mesh.is_watertight


def test_manufacturing_opening_removes_attached_thin_fin_with_bounded_surface_change():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    bulk=box([[0,0,0],[12,12,6]])
    fin=box([[11,4,0],[17,8,.6]])
    source=bulk+fin
    result,record=regularize_shell(source,dict(centers_xy=[],seam_z=-p.sliding_clearance),p)
    assert result.volume()<source.volume()
    import numpy as np
    deviation=np.abs(to_mesh(result).bounds-to_mesh(source).bounds)
    # The intentional fin removal is larger; other outer limits stay within
    # the declared reconstruction resolution. Hardware clearances are separately
    # revalidated on the final reconstructed shape in the formal pipeline.
    assert max(deviation[0])<=p.shell_regularization_pitch
    assert max(deviation[1,1:])<=p.shell_regularization_pitch
    assert to_mesh(result).bounds[1,0]<14
    assert to_mesh(result).is_watertight


def test_hollow_enclosure_survives_final_print_mesh_repair(tmp_path):
    import trimesh
    from smartcar.geometry.print_mesh import stabilize_print_solid
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    source=box([[-15,-20,0],[15,20,12]])-box([[-12.6,-17.6,-1],[12.6,17.6,9.6]])
    opened,_=regularize_shell(source,dict(centers_xy=[],seam_z=-p.sliding_clearance),p)
    final,record=stabilize_print_solid(opened,p)
    assert record.get('status','PASS')=='PASS'
    target=tmp_path/'hollow.stl';to_mesh(final).export(target)
    mesh=trimesh.load(target,force='mesh')
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(mesh.split())==1
    assert not mesh.contains([[0,0,5]])[0]
    assert mesh.volume<source.volume()+p.collision_volume_tolerance


def test_restored_closure_boss_keeps_tip_clearance_and_a_printable_blind_cap():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    seam=.7;xy=[0,0];top=seam+p.sliding_clearance+p.screw_fit['thread_length']
    source=box([[-8,-8,seam+p.sliding_clearance],[8,8,top]])
    result,_=regularize_shell(source,dict(centers_xy=[xy],seam_z=seam),p)
    mesh=to_mesh(result)
    actual_tip=seam-p.bottom_thickness+p.screw_fit['thread_length']
    assert not mesh.contains([[0,0,actual_tip+p.rigid_clearance-.01]])[0]
    assert mesh.contains([[0,0,top-.1]])[0]
    assert mesh.contains([[3,0,top-.1]])[0]
    from smartcar.validation.geometry import wall_measurements
    assert wall_measurements(mesh,p,samples=0,regions=[mesh.bounds])['below_minimum']==0


def test_exact_nominal_flat_walls_survive_discrete_rolling_sphere():
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    outer=box([[-15,-30,0],[15,30,40]])
    inner=box([[-15+p.nominal_wall,-30+p.nominal_wall,-1],[15-p.nominal_wall,30-p.nominal_wall,40-p.nominal_wall]])
    source=outer-inner
    result,_=regularize_shell(source,dict(centers_xy=[],seam_z=-p.sliding_clearance),p)
    mesh=to_mesh(result)
    assert len(result.decompose())==1
    assert result.volume()>source.volume()*.85
    assert mesh.contains([[14,0,20],[-14,0,20],[0,29,20],[0,-29,20]]).all()
    assert not mesh.contains([[0,0,20]])[0]
