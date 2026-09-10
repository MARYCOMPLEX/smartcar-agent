import numpy as np
import trimesh
from smartcar.geometry.mesh import mesh_stats
from smartcar.geometry.rays import all_ray_hits


def wall_measurements(mesh,profile,samples=1200,regions=None):
    points,faces=trimesh.sample.sample_surface(mesh,samples,seed=profile.seed)
    extra=np.zeros(len(mesh.faces),dtype=bool)
    for bounds in regions or []:
        b=np.asarray(bounds)
        extra|=np.all((mesh.triangles_center>=b[0])&(mesh.triangles_center<=b[1]),axis=1)
    feature_faces=np.flatnonzero(extra)
    if len(feature_faces):
        points=np.vstack([points,mesh.triangles_center[feature_faces]])
        faces=np.concatenate([faces,feature_faces])
    directions=-mesh.face_normals[faces]
    eps=profile.numerical_tolerance
    origins=points+directions*eps
    lengths=np.full(len(points),np.inf)
    for start in range(0,len(points),512):
        hits,rays,triangles=all_ray_hits(mesh,origins[start:start+512],directions[start:start+512])
        if len(hits):
            d=np.linalg.norm(hits-origins[start+rays],axis=1)+eps
            valid=d>eps*2
            np.minimum.at(lengths,start+rays[valid],d[valid])
    finite=lengths[np.isfinite(lengths)]
    return dict(method="surface-normal rays: seeded surface samples plus every triangle center in critical junction regions; not a global thickness proof",samples=len(points),
                ray_backend=type(mesh.ray).__module__,random_surface_samples=samples,critical_feature_samples=len(feature_faces),
                resolved_samples=len(finite),minimum_mm=float(finite.min()) if len(finite) else None,
                p01_mm=float(np.quantile(finite,.01)) if len(finite) else None,
                below_minimum=int(sum(finite<profile.minimum_wall-profile.numerical_tolerance)),
                examples=points[np.argsort(lengths)[:12]])


def overhangs(mesh,profile):
    downward=mesh.face_normals[:,2]<-np.cos(np.deg2rad(profile.support_overhang_angle))
    # Exclude the build-plate contact; this estimate assumes exported orientation.
    elevated=mesh.triangles_center[:,2]>mesh.bounds[0,2]+profile.layer_height_mm*2
    return dict(unsupported_overhang_area_mm2=float(mesh.area_faces[downward & elevated].sum()),
                estimated_support_required=bool(np.any(downward & elevated)),method="face normal angle; slicer support plan still required")
