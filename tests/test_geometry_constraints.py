import numpy as np
import trimesh
from smartcar.geometry.sdf import DesignableVolume
from smartcar.geometry.solid import box,intersection_volume,to_mesh
from smartcar.geometry.repair import occupancy_mesh
from smartcar.geometry.collision import aabb_distance
from smartcar.understanding.coordinate_frame import establish_frame
from smartcar.assembly.motion_validation import linear_path
from smartcar.domain.manufacturing import ManufacturingProfile
from pathlib import Path


def profile():return ManufacturingProfile.load(Path(__file__).parents[1]/"config/manufacturing.json")


def test_full_volume_filter_rejects_concavity_between_corners():
    mask=np.ones((25,25,25),bool);mask[11:14,11:14,10:15]=False
    v=DesignableVolume(mask,np.zeros(3),1.,1.)
    assert v.box_clearance([[5,5,5],[20,20,20]])<0
    centers,_=v.feasible_centers([15,15,15],0)
    assert not any(np.linalg.norm(x-[12,12,12])<2 for x in centers)


def test_repair_closes_grid_boundary():
    m=occupancy_mesh(np.ones((5,5,5),bool),[0,0,0],.6)
    assert m.is_watertight and m.is_volume


def test_clearance_not_replaced_by_overlap_test():
    a=np.array([[0,0,0],[1,1,1]])
    b=np.array([[1.2,0,0],[2.2,1,1]])
    assert np.isclose(aabb_distance(a,b),.2)
    assert intersection_volume(box(a),box(b))==0
    assert aabb_distance(a,b)<profile().rigid_clearance


def test_assembly_detects_obstacle_between_final_and_open_pose():
    p=profile()
    moving=box([[0,0,0],[2,2,2]])
    obs=box([[-1,-1,4],[3,3,5]])
    r=linear_path(moving,[("barrier",obs)],[0,0,8],p)
    assert r["status"]=="FAIL"
    assert r["max_intersection_mm3"]>0


def test_frame_is_rigid_and_rotation_equivariant_in_dimensions():
    base=trimesh.creation.box([50,120,35])
    r=trimesh.transformations.rotation_matrix(.7,[1,2,3])
    base.apply_transform(r);base.apply_translation([33,-17,12])
    normalized,report=establish_frame(base)
    mat=np.array(report["input_to_vehicle"])
    assert np.isclose(np.linalg.det(mat[:3,:3]),1)
    assert np.allclose(normalized.extents,[50,120,35],atol=.2)
    assert abs(normalized.bounds[0,2])<1e-6


def test_continuous_minkowski_sweep_preserves_a_real_insertion_hole():
    from smartcar.geometry.solid import cylinder
    p=profile()
    ring=box([[-5,-5,2],[5,5,3]])-cylinder(2,1,4)
    base=box([[-6,-6,0],[6,6,2]])+cylinder(1.4,1,9)
    certificate=linear_path(ring,[("base_and_peg",base)],[0,0,10],p)
    assert certificate["continuous_enclosing_prism_intersection_mm3"]>1
    assert certificate["continuous_certificate"]
    assert certificate["continuous_method"]=="union of leading boundary triangle prisms"


def test_drive_shaft_translation_sweep_is_finite_and_contains_endpoints():
    from smartcar.geometry.solid import cylinder
    from smartcar.assembly.sweep import translation_sweep
    drive=box([[0,0,0],[18,50,22]])+cylinder(3,17,26,(40,11),axis=0)
    swept=translation_sweep(drive,[0,0,40])
    assert abs((drive-swept).volume())<profile().collision_volume_tolerance
    assert abs((drive.translate((0,0,40))-swept).volume())<profile().collision_volume_tolerance
    assert swept.volume()>drive.volume()


def test_scanline_voxelization_preserves_open_cavity():
    from smartcar.geometry.voxelize import voxelize_closed_solid
    hollow=box([[0,0,0],[10,20,8]])-box([[2,2,-1],[8,18,6]])
    mask,origin,record=voxelize_closed_solid(to_mesh(hollow),.5,batch_size=32)
    void=np.round((np.array([5,10,3])-origin)/.5).astype(int)
    wall=np.round((np.array([1,10,3])-origin)/.5).astype(int)
    assert not mask[tuple(void)] and mask[tuple(wall)]
    assert record['ray_axis']==1


def test_four_hole_board_gets_continuous_insertion_certificate():
    from smartcar.geometry.solid import cylinder,union
    p=profile()
    holes=[(-10,-12),(-10,12),(10,-12),(10,12)]
    board=box([[-14,-16,4],[14,16,5.6]])-union(cylinder(1.5,3,7,xy) for xy in holes)
    supports=box([[-16,-18,0],[16,18,2]])+union(cylinder(3.,1,4,xy)+cylinder(1.35,3,10,xy) for xy in holes)
    assert board.num_tri()>600
    result=linear_path(board,[('support_posts',supports)],[0,0,12],p)
    assert result['continuous_certificate'] and result['status']=='PASS'
    assert result['continuous_method']=='union of leading boundary triangle prisms'


def test_rotated_support_column_stays_open_for_continuous_sweep():
    from types import SimpleNamespace
    from smartcar.domain.assembly import placed
    from smartcar.assembly.sweep import translation_sweep
    bounds=np.array([[-15.,-11.799159976577,-20.],[15.,10.78,20.]])
    # CAD and proxy bounds can differ by a few floating-point ULPs at the
    # nominally open mouth; this must not sweep a fictitious closed membrane.
    mouth=bounds[0,1]+1e-13
    regions=[dict(point=[x,mouth,z],height=-.05-mouth,radius=3.22) for x in [-12,12] for z in [-17,17]]
    definition=SimpleNamespace(role='main_controller',bounding_box=bounds,features={'mount_support_sides':{'1':dict(normal_axis=1,valid=True,regions=regions)}})
    angle=np.pi/2;rotation=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
    inst=placed(definition,rotation,[0,-48,23.459159976577],'board','fixture')
    proxy=inst.proxy();offset=np.array([0,0,67.8]);sweep=translation_sweep(proxy,offset)
    cap=inst.bounds.copy();cap[0,2]=cap[1,2];cap[1,2]+=offset[2]
    expected=proxy+box(cap)
    assert (sweep-expected).volume()<profile().collision_volume_tolerance
