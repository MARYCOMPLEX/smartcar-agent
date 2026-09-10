import numpy as np
import trimesh
from smartcar.geometry.conform_edges import conform_open_edges


def test_t_junction_subdivision_restores_closed_surface_without_changing_volume():
    source=trimesh.creation.box([18,130,30]);vertices=source.vertices.copy();faces=source.faces.copy()
    a,b,c=faces[0];middle=(vertices[a]*.27+vertices[b]*.73);mid=len(vertices)
    vertices=np.vstack([vertices,middle])
    faces=np.vstack([faces[1:],[[a,mid,c],[mid,b,c]]])
    broken=trimesh.Trimesh(vertices,faces,process=False)
    assert not broken.is_watertight
    repaired,record=conform_open_edges(broken,1e-5)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert abs(repaired.volume-source.volume)<1e-6
    assert record['split_vertices']==1
    assert repaired.nondegenerate_faces().all()


def test_open_enclosure_is_not_sealed_by_numerical_edge_repair():
    source=trimesh.creation.box([30,70,18]);source.update_faces(source.face_normals[:,2]<.5)
    repaired,record=conform_open_edges(source,1e-5)
    assert not repaired.is_watertight
    assert len(repaired.faces)==len(source.faces)
    assert record['split_vertices']==0


def test_native_mesh_conversion_provides_writable_buffers_for_preview():
    from smartcar.geometry.solid import box,to_mesh
    mesh=to_mesh(box([[0,0,0],[30,50,10]]))
    assert mesh.vertices.flags.writeable and mesh.faces.flags.writeable
    assert mesh.vertices.flags.c_contiguous and mesh.faces.flags.c_contiguous
    preview=mesh.simplify_quadric_decimation(face_count=10)
    assert len(preview.faces)>0
