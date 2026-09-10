"""Full-boundary ray queries, including surfaces beyond the accelerator's default cap."""


def all_ray_hits(mesh, origins, directions):
    ray=mesh.ray
    if type(ray).__module__.endswith('ray_pyembree'):
        # The wrapper defaults to 100 intersections, which silently truncates
        # parity for a detailed multi-shell solid. A line can hit each triangle
        # at most once; allowance for duplicate-hit advancement is also bounded
        # by the actual geometry, not an appearance-specific limit.
        triangles,rays,points=ray.intersects_id(origins,directions,multiple_hits=True,
            max_hits=max(100,2*len(mesh.faces)),return_locations=True)
        return points,rays,triangles
    return ray.intersects_location(origins,directions,multiple_hits=True)
