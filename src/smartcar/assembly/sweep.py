"""Polyhedral translation sweep from leading boundary triangle prisms."""
import numpy as np
import manifold3d as mf
from smartcar.geometry.solid import to_mesh,union


def translation_sweep(solid,offset):
    vector=np.asarray(offset,dtype=float);mesh=to_mesh(solid)
    faces=np.array([[2,1,0],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]],dtype=np.uint64)
    prisms=[solid]
    for triangle,normal in zip(mesh.triangles,mesh.face_normals):
        if normal@vector<=1e-12:continue
        vertices=np.vstack([triangle,triangle+vector])
        prism=mf.Manifold(mf.Mesh64(np.ascontiguousarray(vertices),faces))
        if prism.status()!=mf.Error.NoError:raise ValueError('CONTINUOUS_PRISM_CONSTRUCTION_FAILED')
        prisms.append(prism)
    return union(prisms)
