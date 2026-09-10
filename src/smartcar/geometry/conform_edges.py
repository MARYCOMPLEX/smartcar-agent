"""Resolve numerical T junctions without filling physical holes or cavities."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def conform_open_edges(mesh,tolerance,max_rounds=100):
    result=mesh.copy();splits=0;maximum_deviation=0.
    for _ in range(max_rounds):
        counts=np.bincount(result.edges_unique_inverse,minlength=len(result.edges_unique))
        boundary=result.edges_unique[counts==1]
        if not len(boundary):break
        vertex_ids=np.unique(boundary);tree=cKDTree(result.vertices[vertex_ids])
        edge_to_face={}
        # Boundary edges have exactly one incident face.
        for edge,face,index in zip(result.edges,result.edges_face,result.edges_unique_inverse):
            if counts[index]==1:edge_to_face[tuple(sorted(edge))]=int(face)
        replacements={}
        for a,b in boundary:
            face_id=edge_to_face[(a,b)]
            if face_id in replacements:continue
            first,last=result.vertices[[a,b]];vector=last-first;length=np.linalg.norm(vector)
            if length<=2*tolerance:continue
            ids=vertex_ids[tree.query_ball_point((first+last)/2,length/2+tolerance)]
            parameter=(result.vertices[ids]-first)@vector/(length*length)
            projected=first+parameter[:,None]*vector
            distance=np.linalg.norm(result.vertices[ids]-projected,axis=1)
            valid=(parameter>tolerance/length)&(parameter<1-tolerance/length)&(distance<=tolerance)
            ids=ids[valid];parameter=parameter[valid]
            if not len(ids):continue
            maximum_deviation=max(maximum_deviation,float(distance[valid].max()))
            ordered=ids[np.argsort(parameter)].tolist()
            triangle=result.faces[face_id];ia=int(np.flatnonzero(triangle==a)[0])
            if triangle[(ia+1)%3]!=b:
                a,b=b,a;ordered=ordered[::-1];ia=int(np.flatnonzero(triangle==a)[0])
            third=triangle[(ia+2)%3];chain=[a,*ordered,b]
            replacements[face_id]=[[chain[i],chain[i+1],third] for i in range(len(chain)-1)]
            splits+=len(ordered)
        if not replacements:break
        keep=np.ones(len(result.faces),bool);keep[list(replacements)]=False
        faces=np.vstack([result.faces[keep],*[np.asarray(f) for f in replacements.values()]])
        result=trimesh.Trimesh(result.vertices,faces,process=False)
    return result,dict(split_vertices=splits,maximum_edge_deviation_mm=maximum_deviation,
                       tolerance_mm=tolerance,watertight=result.is_watertight)
