"""Manifold solids for exact mesh boolean operations; CAD is tessellated once."""
from __future__ import annotations
import numpy as np
import trimesh
import manifold3d as mf


def from_mesh(mesh):
    m = mesh.copy()
    m.merge_vertices(digits_vertex=12)
    s = mf.Manifold(mf.Mesh64(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.uint64)))
    if s.status() != mf.Error.NoError:
        raise ValueError(f"NON_MANIFOLD_SOLID:{s.status()}")
    return s


def to_mesh(solid):
    m = solid.to_mesh64()
    vertices=np.array(np.asarray(m.vert_properties)[:,:3],dtype=np.float64,order='C',copy=True)
    faces=np.array(m.tri_verts,dtype=np.int64,order='C',copy=True)
    # Property seams may duplicate vertices even though Manifold's topology is
    # closed. Respect its authoritative merge mapping before handing to Trimesh.
    parent=np.arange(len(vertices))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in zip(m.merge_from_vert,m.merge_to_vert):parent[root(a)]=root(b)
    if len(m.merge_from_vert):
        remap=np.array([root(i) for i in range(len(vertices))]);faces=remap[faces]
    # Manifold stores a valid indexed topology. Trimesh's default vertex merge
    # can collapse sub-micron Boolean slivers and turn one solid into fragments.
    # Simplify below CAD tessellation tolerance before the STL float32 roundtrip.
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    mesh.remove_unreferenced_vertices()
    # Remove zero-area faces introduced by the float32 STL coordinate boundary,
    # while accepting the cleaned mesh only when its closed topology is intact.
    cleaned=trimesh.Trimesh(mesh.vertices,mesh.faces,process=True,validate=True)
    return cleaned if cleaned.is_watertight and cleaned.is_winding_consistent else mesh


def cancel_collapsed_faces(mesh):
    """Cancel equal opposite faces when float32 welding collapses a zero-volume fin.

    This is signed boundary cancellation, not removal of small physical parts.
    A remaining invalid boundary is retained and rejected by the validator.
    """
    welded=trimesh.Trimesh(np.asarray(mesh.vertices,dtype=np.float32),mesh.faces,process=True)
    welded.update_faces(welded.nondegenerate_faces())
    f=welded.faces
    _,inverse,counts=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True)
    keep=np.ones(len(f),bool)
    for group in np.flatnonzero(counts>1):
        ids=np.flatnonzero(inverse==group);faces=f[ids]
        parity=((faces[:,0]>faces[:,1]).astype(int)+(faces[:,0]>faces[:,2])+(faces[:,1]>faces[:,2]))%2
        positive=ids[parity==0];negative=ids[parity==1];pairs=min(len(positive),len(negative))
        keep[positive[:pairs]]=False;keep[negative[:pairs]]=False
    welded.update_faces(keep);welded.remove_unreferenced_vertices()
    return welded


def box(bounds):
    b = np.asarray(bounds, dtype=float)
    return mf.Manifold.cube(tuple(b[1]-b[0])).translate(tuple(b[0]))


def cylinder(radius, z0, z1, xy=(0,0), axis=2):
    s = mf.Manifold.cylinder(z1-z0, radius, circular_segments=64)
    if axis == 0: s = s.rotate((0,90,0))
    if axis == 1: s = s.rotate((-90,0,0))
    p = [0.,0.,0.]
    p[axis] = z0
    j = [i for i in range(3) if i != axis]
    p[j[0]], p[j[1]] = xy
    return s.translate(tuple(p))


def union(solids):
    solids = list(solids)
    return mf.Manifold.batch_boolean(solids, mf.OpType.Add) if solids else mf.Manifold()


def intersection_volume(a,b):
    return max(0., (a ^ b).volume())


def move(solid, matrix):
    return solid.transform(np.asarray(matrix)[:3,:4])
