import numpy as np
import trimesh
from smartcar.geometry.rays import all_ray_hits


def test_full_parity_ray_retains_more_than_one_hundred_intersections():
    pieces=[trimesh.creation.box([.25,2,2]).apply_translation([i,0,0]) for i in range(60)]
    mesh=trimesh.util.concatenate(pieces)
    hits,rays,_=all_ray_hits(mesh,[[-2,.13,.17]],[[1,0,0]])
    expected=np.array([[i-.125,i+.125] for i in range(60)]).ravel()
    assert len(hits)==120 and np.all(rays==0)
    assert np.allclose(np.sort(hits[:,0]),expected,atol=1e-5)


def test_accelerated_thin_wall_distances_match_float64_triangle_backend():
    from trimesh.ray.ray_triangle import RayMeshIntersector
    mesh=trimesh.creation.box([.21,8,12]).apply_translation([123,67,-5])
    origins=np.array([[123,67,-5],[123,67.1,-4.5]])
    directions=np.tile([1.,0.,0.],(2,1))
    fast,ids,_=all_ray_hits(mesh,origins,directions)
    exact,eids,_=RayMeshIntersector(mesh).intersects_location(origins,directions,multiple_hits=True)
    assert np.array_equal(np.sort(ids),np.sort(eids))
    assert np.allclose(fast[np.argsort(ids)],exact[np.argsort(eids)],atol=1e-5)
