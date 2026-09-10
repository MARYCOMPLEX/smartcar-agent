from __future__ import annotations
import itertools
import numpy as np
from scipy.spatial import cKDTree
from smartcar.geometry.mesh import sample_surface, symmetry_error
from smartcar.understanding.wheel_evidence import wheel_component_evidence


def establish_frame(mesh,policy=None):
    """Evaluate OBB/PCA axes and both ground signs. No source-coordinate assumptions.

    Length is +Y, bilateral width +X. Front sign uses the upper silhouette centroid
    (cab/hood evidence) and is explicitly ambiguous without semantic annotation.
    """
    points = sample_surface(mesh, 20000)
    cov = np.cov(points.T)
    _, ev = np.linalg.eigh(cov)
    obb = mesh.bounding_box_oriented.primitive.transform[:3,:3]
    candidates = []
    components=mesh.split(only_watertight=False)
    for source, basis in [("obb",obb), ("pca",ev)]:
        q = points @ basis
        all_vertices=mesh.vertices@basis
        spans=np.ptp(all_vertices,axis=0)
        mids=(all_vertices.min(0)+all_vertices.max(0))/2
        tree=cKDTree(q);axis_symmetry={}
        for axis in range(3):
            reflected=q.copy();reflected[:,axis]=2*mids[axis]-reflected[:,axis]
            distances=tree.query(reflected)[0]
            axis_symmetry[axis]=dict(mean_mm=float(distances.mean()),p95_mm=float(np.quantile(distances,.95)))
        # Height is not necessarily the shortest dimension: tall/narrow cars
        # require all axis permutations, then geometry evidence chooses a frame.
        for (xidx,yidx,zidx),sign in itertools.product(itertools.permutations(range(3)),[-1,1]):
            z = basis[:,zidx] * sign
            y = basis[:,yidx]
            x = np.cross(y,z)
            r = np.stack([x,y,z])
            v = mesh.vertices @ r.T
            lo,hi = v.min(0), v.max(0)
            # Large horizontal low faces distinguish underbody from roof, with
            # lower-quarter footprint as additional evidence.
            centers = mesh.triangles_center @ r.T
            normals = mesh.face_normals @ r.T
            low = centers[:,2] < lo[2] + .23*(hi[2]-lo[2])
            flat = np.abs(normals[:,2]) > .94
            flat_area = float(mesh.area_faces[low & flat].sum())
            lowp = v[v[:,2] < lo[2]+.25*(hi[2]-lo[2])]
            footprint = float(np.prod(np.ptp(lowp[:,:2],axis=0))) if len(lowp) else 0.
            ground_score=flat_area/(mesh.area+1e-9)+.25*footprint/np.prod((hi-lo)[:2])
            tolerance=max(float(spans.max())*.005,np.finfo(float).eps)
            wheels=wheel_component_evidence(components,np.array([lo,hi]),tolerance,r)
            grounded=sum(w['wheel_count'] for w in wheels if w['ground_gap_mm']<=2*tolerance)
            symmetry=axis_symmetry[xidx]
            scores=dict(ground_surface=ground_score,
                        bilateral_symmetry=-4*symmetry['mean_mm']/spans[xidx],
                        grounded_wheels=.5*min(4,grounded),
                        length_prior=.2*spans[yidx]/spans.max(),
                        height_prior=.05*spans.min()/spans[zidx])
            candidates.append(dict(source=source,up_sign=sign,axis_indices=[xidx,yidx,zidx],
                                   score=float(sum(scores.values())),score_components=scores,
                                   low_flat_area_mm2=flat_area,rotation=r,bilateral_symmetry=symmetry,
                                   wheel_evidence=wheels,grounded_wheel_count=grounded))
    from smartcar.understanding.ground_frame import rescore_ground_frames
    rescore_ground_frames(mesh,candidates,policy)
    best = max(candidates, key=lambda c:c["score"])
    r = best["rotation"].copy()
    v = mesh.vertices @ r.T
    # Include ties: a planar roof can place the entire upper quantile exactly
    # at one height. A strict comparison produces an empty semantic sample.
    top = v[:,2] >= np.quantile(v[:,2], .8)
    front_signal = float(v[top,1].mean() - v[:,1].mean())
    # In the cab/hood hypothesis the high cabin lies toward the rear; the lower
    # extended hood lies in front. This remains explicitly ambiguous for vans.
    if front_signal > 0: r[:2] *= -1
    v = mesh.vertices @ r.T
    t = np.array([-(v[:,0].max()+v[:,0].min())/2, -(v[:,1].max()+v[:,1].min())/2, -v[:,2].min()])
    matrix = np.eye(4); matrix[:3,:3] = r; matrix[:3,3] = t
    normalized = mesh.copy().apply_transform(matrix)
    report = dict(input_to_vehicle=matrix, vehicle_to_input=np.linalg.inv(matrix),
                  convention={"X":"left", "Y":"front (inferred)", "Z":"up"}, ground_plane=[0,0,1,0],
                  selected=best, candidates=candidates, front_signal_mm=abs(front_signal),
                  semantic_confidence="front/rear ambiguous: upper-silhouette heuristic; axes scored by bilateral symmetry, ground and separate wheel/axle evidence",
                  symmetry=symmetry_error(normalized,0), dimensions_mm=normalized.extents)
    return normalized, report
