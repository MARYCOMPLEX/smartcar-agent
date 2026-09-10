from pathlib import Path
import itertools
import numpy as np
import trimesh
from smartcar.domain.manufacturing import ManufacturingProfile
from smartcar.geometry.mesh import mesh_stats
from smartcar.geometry.solid import from_mesh,to_mesh
from smartcar.geometry.print_mesh import finalize_print_parts,stabilize_print_solid


def test_subresolution_chamfer_is_removed_without_coarse_surface_simplification(tmp_path):
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    # An analytic box with one corner truncated at the float32 encoding scale.
    # This is a genuine closed tessellation with a tiny, nonzero-area facet;
    # height-based degeneracy checks alone miss it.
    vertices=np.array(list(itertools.product([-10.,10.],[-15.,15.],[-5.,5.])))
    corner=vertices[-1].copy();vertices=vertices[:-1]
    tiny=corner-np.eye(3)*8e-6
    mesh=trimesh.convex.convex_hull(np.vstack([vertices,tiny])+[123.,67.,31.])
    solid=from_mesh(mesh)
    old,_=stabilize_print_solid(solid,p)
    assert mesh_stats(to_mesh(old),include_components=False)['degenerate_triangles']>0
    parts,records=finalize_print_parts({'body':solid},p)
    assert records['body'].get('status','PASS')!='FAIL'
    path=tmp_path/'body.stl';to_mesh(parts['body']).export(path)
    exported=trimesh.load(path,force='mesh');stats=mesh_stats(exported)
    assert stats['degenerate_triangles']==0 and stats['watertight'] and stats['manifold']
    assert stats['component_count']==1
    assert abs(exported.volume-mesh.volume)<p.collision_volume_tolerance
    assert np.max(np.abs(exported.bounds-mesh.bounds))<p.numerical_tolerance


def test_coincident_bottom_layers_are_conditioned_before_float32_welding(tmp_path):
    from smartcar.geometry.solid import box,cylinder,cancel_collapsed_faces
    from smartcar.structure.bottom_regularization import regularize_bottom
    p=ManufacturingProfile.load(Path(__file__).parents[1]/'config/manufacturing.json')
    floor=5.3
    base=box([[-20,-25,floor-2.4],[20,25,floor]])-cylinder(7,floor-3,floor+3,[17,5])
    mount=cylinder(3.2,floor-.1,floor+20,[12,12])
    source,_=regularize_bottom(base+mount,floor,p)
    source+=box([[-25,-8,floor-2.4+1e-7],[25,8,floor+1e-7]])
    assert not cancel_collapsed_faces(to_mesh(source)).is_watertight
    parts,records=finalize_print_parts({'bottom_cover':source},p)
    assert records['bottom_cover'].get('status','PASS')!='FAIL'
    path=tmp_path/'bottom_cover.stl';to_mesh(parts['bottom_cover']).export(path)
    exported=trimesh.load(path,force='mesh');stats=mesh_stats(exported)
    assert stats['watertight'] and stats['manifold'] and stats['degenerate_triangles']==0
    assert stats['component_count']==1
    assert abs(exported.volume-source.volume())<p.collision_volume_tolerance
    # Planar base regularization has already trimmed the bottom 0.1 mm of the
    # fixture's mount; compare against the actual formal input to finalization.
    assert abs((parts['bottom_cover']^mount).volume()-(source^mount).volume())<p.collision_volume_tolerance
